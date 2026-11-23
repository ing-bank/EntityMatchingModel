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

from copy import copy
from typing import Any, Mapping

import numpy as np
import polars as pl
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBModel

from emm.features.pandas_feature_extractor import PandasFeatureExtractor
from emm.features.polars_feature_extractor import PolarsFeatureExtractor
from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.supervised_model.base_supervised_model import BaseSupervisedModel


def _polars_feature_step(feature_step: Any) -> Any:
    if not isinstance(feature_step, PandasFeatureExtractor):
        return feature_step
    if type(feature_step) is not PandasFeatureExtractor:
        msg = "custom pandas feature extractor subclasses are not supported by the Polars supervised layer"
        raise TypeError(msg)

    default = PandasFeatureExtractor()
    if not feature_step.without_rank_features and (
        feature_step.rank_features != default.rank_features or feature_step.diff_features != default.diff_features
    ):
        msg = "custom pandas rank or difference features are not supported by the Polars supervised layer"
        raise ValueError(msg)

    feature = PolarsFeatureExtractor(
        name1_col=feature_step.name1_col,
        name2_col=feature_step.name2_col,
        uid_col=feature_step.uid_col,
        gt_uid_col=feature_step.gt_uid_col,
        score_columns=feature_step.score_columns,
        extra_features=feature_step.extra_features,
        vocabulary=feature_step.vocabulary,
        without_rank_features=feature_step.without_rank_features,
        with_legal_entity_forms_match=feature_step.with_legal_entity_forms_match,
        fillna_value=feature_step.fillna_value,
        drop_features=feature_step.drop_features,
    )
    feature.name_features = feature_step.name_features.copy()
    feature._fitted = feature_step._fitted
    return feature


def _update_fitted_pandas_feature(original: Any, feature: Any) -> None:
    if isinstance(original, PandasFeatureExtractor) and not original._fitted:
        original.vocabulary = feature.vocabulary
        original._fitted = feature._fitted


def _categoricals_to_enums(features: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(column).cast(pl.Enum(sorted(features[column].drop_nulls().unique().to_list())))
        for column, dtype in features.schema.items()
        if dtype == pl.Categorical
    ]
    return features.with_columns(columns)


def _polars_model(model: Any) -> Any:
    if not hasattr(model, "named_steps") or "feat" not in model.named_steps:
        return model
    feature = model.named_steps["feat"]
    adapt_feature = isinstance(feature, PandasFeatureExtractor)
    adapt_categories = isinstance(model, Pipeline) and isinstance(model.steps[-1][1], XGBModel)
    if not adapt_feature and not adapt_categories:
        return model
    if not isinstance(model, Pipeline):
        msg = "a model with a pandas feature extractor must be an sklearn Pipeline"
        raise TypeError(msg)
    adapted = copy(model)
    adapted.steps = []
    category_step = "polars_categories"
    while category_step in model.named_steps:
        category_step = f"_{category_step}"
    for name, step in model.steps:
        adapted.steps.append((name, _polars_feature_step(step) if name == "feat" else step))
        if name == "feat" and adapt_categories:
            adapted.steps.append((category_step, FunctionTransformer(_categoricals_to_enums)))
    return adapted


def _not_missing(df: pl.DataFrame, column: str) -> pl.Expr:
    value = pl.col(column)
    return value.is_not_null() & ~value.is_nan() if df.schema[column].is_float() else value.is_not_null()


def _sort_expression(df: pl.DataFrame, column: str) -> pl.Expr:
    value = pl.col(column)
    return value.fill_nan(None) if df.schema[column].is_float() else value


class PolarsSupervisedLayerTransformer(TransformerMixin, BaseSupervisedModel):
    """Score and rank candidate pairs using sklearn models on Polars frames.

    Models must accept Polars input. Standard sklearn pipelines containing a
    pandas feature extractor are adapted to the equivalent Polars extractor;
    their trained estimator and vocabulary are retained. Custom pandas feature
    extractor subclasses and custom rank or difference functions are not supported.
    """

    def __init__(
        self,
        supervised_models: Mapping[str, dict],
        best_score_col: str | None = "nm_score",
        return_features: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.supervised_models = supervised_models
        self.best_score_col = best_score_col
        self.return_features = return_features
        BaseSupervisedModel.__init__(self)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> PolarsSupervisedLayerTransformer:
        """Return the transformer without fitting the supplied models."""
        return self

    def fit_transform(self, X: pl.DataFrame, y: pl.Series | None = None) -> None:
        """Fit without scoring the ground-truth frame."""
        self.fit(X, y)

    def calc_features(self, X: pl.DataFrame) -> pl.DataFrame:
        """Append each model's feature columns, including for disabled models."""
        logger.info("calculcating sm features.")
        for model_col, model_dict in self.supervised_models.items():
            model = model_dict["model"]
            if "feat" in model.named_steps:
                original_feature = model.named_steps["feat"]
                feature = _polars_feature_step(original_feature)
                feat = feature.transform(X)
                if not isinstance(feat, pl.DataFrame):
                    msg = "the supervised model's feature step must return a Polars DataFrame"
                    raise TypeError(msg)
                if feat.height != X.height:
                    msg = "the supervised model's feature step must preserve the number of rows"
                    raise ValueError(msg)
                _update_fitted_pandas_feature(original_feature, feature)
                X = X.with_columns([feat[col].rename(f"{model_col}_feat_{col}") for col in feat.columns])
        return X

    def calc_score(self, X: pl.DataFrame) -> pl.DataFrame:
        """Score only rows with a ground-truth candidate."""
        for model_col, model_dict in self.supervised_models.items():
            if not model_dict["enable"]:
                continue
            mask = _not_missing(X, "gt_uid")
            candidates = X.filter(mask)
            if candidates.is_empty():
                X = X.with_columns(pl.lit(0.0).alias(model_col))
                continue

            model = _polars_model(model_dict["model"])
            predictions = np.asarray(model.predict_proba(candidates))
            if predictions.ndim != 2 or predictions.shape[0] != candidates.height or predictions.shape[1] < 2:
                msg = "predict_proba must return one row per candidate and at least two probability columns"
                raise ValueError(msg)
            original_model = model_dict["model"]
            if model is not original_model and "feat" in model.named_steps:
                _update_fitted_pandas_feature(original_model.named_steps["feat"], model.named_steps["feat"])
            row_id = "__emm_supervised_row"
            while row_id in X.columns:
                row_id = f"_{row_id}"
            score_id = f"_{row_id}_score"
            while score_id in X.columns:
                score_id = f"_{score_id}"
            indexed = X.with_row_index(row_id)
            scores = indexed.filter(mask).select(row_id).with_columns(pl.Series(score_id, predictions[:, 1]))
            indexed = indexed.join(scores, on=row_id, how="left", maintain_order="left")
            previous = pl.col(model_col) if model_col in X.columns else pl.lit(None)
            X = indexed.with_columns(pl.when(mask).then(pl.col(score_id)).otherwise(previous).alias(model_col)).drop(
                row_id, score_id
            )
        return X

    def select_best_score(
        self,
        X: pl.DataFrame,
        group_cols: list[str],
        best_score_col: str | None = "nm_score",
        sort_cols: list[str] | None = None,
        sort_asc: list[bool] | None = None,
        best_match_col: str = "best_match",
        best_rank_col: str = "best_rank",
        gt_uid_col: str | None = "gt_uid",
    ) -> pl.DataFrame:
        """Rank candidates within groups and mark the best positive score."""
        if best_score_col not in self.supervised_models or not self.supervised_models[best_score_col]["enable"]:
            return X
        if sort_cols is None:
            sort_cols, sort_asc = [best_score_col], [False]
        if sort_asc is None:
            msg = "sort_asc is required when sort_cols is provided"
            raise ValueError(msg)
        if gt_uid_col is None:
            msg = "gt_uid_col is required to break score ties"
            raise ValueError(msg)

        rank_sort = [*group_cols, best_score_col, gt_uid_col]
        X = X.sort(
            [_sort_expression(X, col) for col in rank_sort], descending=True, nulls_last=True, maintain_order=True
        )
        valid_group = pl.all_horizontal([_not_missing(X, col) for col in group_cols])
        X = X.with_columns(
            pl.when(valid_group)
            .then(pl.int_range(1, pl.len() + 1).over(group_cols))
            .otherwise(None)
            .cast(pl.Int64)
            .alias(best_rank_col)
        )
        X = X.with_columns(
            (
                (pl.col(best_rank_col) == 1).fill_null(False)
                & _not_missing(X, best_score_col)
                & (pl.col(best_score_col) > 0).fill_null(False)
            ).alias(best_match_col)
        )
        sort_by = [*group_cols, *sort_cols]
        return X.sort(
            [_sort_expression(X, col) for col in sort_by],
            descending=[False] * len(group_cols) + [not value for value in sort_asc],
            nulls_last=True,
            maintain_order=True,
        )

    def transform(self, X: pl.DataFrame | None) -> pl.DataFrame | None:
        """Score, rank, and optionally append features for candidate pairs."""
        if X is None:
            return None
        if not isinstance(X, pl.DataFrame):
            msg = f"expected a Polars DataFrame, got {type(X).__name__}"
            raise TypeError(msg)
        with Timer("PolarsSupervisedLayerTransformer.transform") as timer:
            timer.log_params({"X.shape": X.shape, "return_features": self.return_features})
            X = self.calc_score(X)
            X = self.select_best_score(X, best_score_col=self.best_score_col, group_cols=["uid"])
            if self.return_features:
                X = self.calc_features(X)
            timer.log_param("cands", X.height)
        return X
