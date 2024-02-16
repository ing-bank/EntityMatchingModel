# Copyright (c) 2023 ING Analytics Wholesale Banking
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from emm.base.module import Module
from emm.features.pandas_feature_extractor import PandasFeatureExtractor
from emm.loggers import Timer
from emm.loggers.logger import logger


class BaseSupervisedModel(Module):
    def __init__(self) -> None:
        super().__init__()


def create_new_model_pipeline(
    name_only: bool = True, feature_args: dict | None = None, xgb_args: dict | None = None
) -> Pipeline:
    default_feature_args = {
        "name1_col": "preprocessed",
        "name2_col": "gt_preprocessed",
        "uid_col": "uid",
        "gt_uid_col": "gt_uid",
        "score_columns": ["score_0"],
        "vocabulary": None,
        "extra_features": [] if name_only else ["country"],
        "without_rank_features": False,
        "with_legal_entity_forms_match": False,
        "drop_features": [],
    }
    feature_args = {k: v for k, v in feature_args.items() if v is not None} if feature_args is not None else {}
    default_feature_args.update(feature_args)

    enable_categorical = default_feature_args["with_legal_entity_forms_match"]

    default_xgb_args = {
        "objective": "binary:logistic",
        "learning_rate": 0.1,
        "eval_metric": "aucpr",
        "seed": 0,
        "enable_categorical": enable_categorical,
        "tree_method": "approx",
        "n_jobs": -1,
    }
    xgb_args = {k: v for k, v in xgb_args.items() if v is not None} if xgb_args is not None else {}
    default_xgb_args.update(xgb_args)

    return Pipeline(
        [("feat", PandasFeatureExtractor(**default_feature_args)), ("classifier", XGBClassifier(**default_xgb_args))]
    )


def calc_features_from_sm(sm: Pipeline, input: pd.DataFrame, features_name="feat"):
    res = pd.DataFrame(index=input.index)
    if not hasattr(sm, "named_steps"):
        logger.warning("calc_features_from_sm is supported only for new models (version > 0.0.4)")
        return res
    if features_name in sm.named_steps:
        feat_step = sm.named_steps[features_name]
        return feat_step.transform(input)
    return res


def features_schema_from_sm(sm: Pipeline, return_spark_types=False):
    if not hasattr(sm, "named_steps"):
        logger.warning("features_schema_from_sm is supported only for new models (version > 0.0.4)")
        return []
    feat_step = sm.named_steps["feat"]
    input_stub = pd.DataFrame(
        {
            "uid": [0],
            "gt_uid": [1],
            "name": ["a"],
            "gt_name": ["b"],
            "preprocessed": ["a"],
            "gt_preprocessed": ["b"],
            "country": ["NL"],
            "gt_country": ["NL"],
        }
    )
    for i in range(10):
        input_stub[f"score_{i}"] = 0.0
    output_stub = feat_step.transform(input_stub)
    res = list(output_stub.dtypes.items())
    if return_spark_types:
        import pyspark.sql.types as T

        mapping = {
            "int8": T.IntegerType(),
            "int64": T.IntegerType(),
            "float32": T.FloatType(),
            "float64": T.DoubleType(),
        }
        return [(name, mapping[str(dtype)]) for name, dtype in res]
    return res


def train_model(
    train_df,
    vocabulary=None,
    name_only=False,
    without_rank_features=False,
    positive_set_col="positive_set",
    custom_model=None,
    score_columns=None,
    with_legal_entity_forms_match=False,
    drop_features=None,
    n_jobs=-1,
    positive_only=False,
    extra_features=None,
    **feature_kws,
):
    """Train the supervised pipeline

    No testing. Input dataset contains 1 row per candidate

    Args:
        train_df: input name-pairs to train on. See prepare_name_pairs().
        vocabulary: vocabulary of common words. See create_vocabulary().
        name_only: use name-only features. Default is false.
        without_rank_features: without generated rank features, default is false.
        positive_set_col: name of positive_set column, default is 'positive_set'.
        custom_model: custom pipeline, default is None.
        score_columns: list of columns with raw scores from indexers to pass to classifier.
                          default is None, meaning all indexer scores (e.g. cosine similarity values).
        with_legal_entity_forms_match: if True, then add match of legal entity forms.
        drop_features: list of features to drop at end of feature calculation, before sm. default is None.
        n_jobs: number of parallel jobs passed on to model. Default -1.
        positive_only: if true, train on positive names only and reject negative ones. default is False.
        extra_features: list of columns (and possibly functions) used for extra features calculation,
                        e.g. country if name_only=False, default is None.
                        With ``name_only=False`` internally ``extra_features=['country']``.
        feature_kws: extra kwargs passed on to model init function.

    Returns:
        trained model
    """
    for col in ["correct", "no_candidate"]:
        if col not in train_df.columns:
            msg = f"column {col} not in dataset. Did you run prepare_dataset()?"
            raise ValueError(msg)

    if score_columns is None:
        score_columns = [c for c in train_df.columns if re.match(r"^(score)_\d+$", c)]

    if positive_only and positive_set_col in train_df.columns:
        logger.debug("train_on: positive names only")
        train_fit = train_df[train_df[positive_set_col]]  # Train only on positive
    else:
        logger.debug("train_on: all names")
        train_fit = train_df
    train_fit = train_fit[~train_fit.no_candidate]  # Keep only names-to-match that have a candidate for training

    if custom_model is None:
        feature_args = {
            "vocabulary": vocabulary,
            "score_columns": score_columns,
            "without_rank_features": without_rank_features,
            "with_legal_entity_forms_match": with_legal_entity_forms_match,
            "drop_features": drop_features,
            "extra_features": extra_features,
        }
        feature_args.update(feature_kws)
        xgb_args = {"n_jobs": n_jobs}
        model = create_new_model_pipeline(name_only=name_only, feature_args=feature_args, xgb_args=xgb_args)
    else:
        model = custom_model

    # The `train_fit` dataframe should contain at least `name1_col,name2_col,uid_col` and `score_columns`.
    # the rest is ignored in CalcFeatures module.
    with Timer("Fitting supervised model pipeline"):
        model.fit(X=train_fit, y=train_fit["correct"])

    return model


def train_test_model(
    dataset,
    vocabulary=None,
    name_only=False,
    without_rank_features=False,
    n_folds=8,
    account_col="account",
    uid_col="uid",
    random_state=42,
    positive_set_col="positive_set",
    benchmark_col="score_0",
    custom_model=None,
    score_columns=None,
    with_legal_entity_forms_match=False,
    drop_features=None,
    n_jobs=-1,
    positive_only=False,
    extra_features=None,
):
    """Train and test the supervised pipeline

    Input dataset contains 1 row per candidate

    Args:
        dataset: input name-pairs to train on and validate. See prepare_name_pairs().
        vocabulary: vocabulary of common words. See create_vocabulary().
        name_only: use name-only features. Default is false.
        without_rank_features: without generated rank features, default is false.
        n_folds: number of folds. One is used for validation.
        account_col: account column, default is "account".
        uid_col: uid column, default is "uid".
        random_state: random seed, default is 42.
        positive_set_col: name of positive_set column, default is 'positive_set'.
        benchmark_col: for benchmark validation, default score column is "score_0".
        custom_model: custom pipeline, default is None.
        score_columns: list of columns with raw scores from indexers to pass to classifier.
                          default is None, meaning all indexer scores (e.g. cosine similarity values).
        with_legal_entity_forms_match: if True, then add match of legal entity forms
        drop_features: list of features to drop at end of feature calculation, before sm. default is None.
        n_jobs: number of parallel jobs passed on to model. Default -1.
        positive_only: if true, train on positive names only and reject negative ones. default is False.
        extra_features: list of columns (and possibly functions) used for extra features calculation,
                        e.g. country if name_only=False, default is None.
                        With ``name_only=False`` internally ``extra_features=['country']``.

    Returns:
        tuple of trained model and scored dataset.
    """
    logger.info("Training the supervised model")

    for col in [uid_col, "correct", "no_candidate"]:
        if col not in dataset.columns:
            msg = f"column {col} not in dataset. Did you run prepare_dataset()?"
            raise ValueError(msg)
    group_col = account_col if account_col in dataset.columns else uid_col

    y = dataset["correct"].astype(str) + dataset["no_candidate"].astype(str)
    if positive_set_col in dataset.columns:
        y += dataset[positive_set_col].astype(str)

    # Train test split with consistent name-to-match account (group) and with approximately same class balance y (stratified)
    # it is important to have all the name in the same account, for account matching after aggregation
    # remark: we use StratifiedGroupKFold() not for cross-validation folds, but just to split in two: training/validation.
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_inds, valid_inds = next(cv.split(X=dataset, y=y, groups=dataset[group_col]))
    train_df, valid_df = (dataset.iloc[train_inds].copy(), dataset.iloc[valid_inds].copy())

    model = train_model(
        train_df,
        vocabulary=vocabulary,
        name_only=name_only,
        without_rank_features=without_rank_features,
        positive_set_col=positive_set_col,
        custom_model=custom_model,
        score_columns=score_columns,
        with_legal_entity_forms_match=with_legal_entity_forms_match,
        drop_features=drop_features,
        n_jobs=n_jobs,
        positive_only=positive_only,
        extra_features=extra_features,
    )

    # We score 'train_df' and 'valid_df', and not on 'dataset' to avoid leakage/issues
    for _label, df in [("train", train_df), ("valid", valid_df)]:
        df["nm_score"] = model.predict_proba(df)[:, 1]
        # need to manually fix score for no-candidate rows (to have same behaviour as in SparkSupervisedLayerEstimator)
        df.loc[df.no_candidate, "nm_score"] = 0.0

        # check nm_score for non-candidate rows
        assert (df[df.no_candidate]["nm_score"] == 0.0).all()

    train_df["fold"] = "train"
    valid_df["fold"] = "valid"
    dataset_scored = pd.concat([train_df, valid_df])

    # Compute rank column
    dataset_scored["nm_score_rank"] = dataset_scored.groupby("uid", group_keys=False)["nm_score"].apply(
        lambda x: x.rank(ascending=False, method="first", na_option="bottom")
    )
    dataset_scored[f"{benchmark_col}_rank"] = dataset_scored.groupby("uid", group_keys=False)[benchmark_col].apply(
        lambda x: x.rank(ascending=False, method="first", na_option="bottom")
    )

    return model, dataset_scored
