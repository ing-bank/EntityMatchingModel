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

from typing import TYPE_CHECKING, Any, Mapping

from sklearn.base import TransformerMixin

from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.supervised_model.base_supervised_model import BaseSupervisedModel, calc_features_from_sm

if TYPE_CHECKING:
    import pandas as pd


class PandasSupervisedLayerTransformer(TransformerMixin, BaseSupervisedModel):
    """Pandas implementation of supervised model(s) transformer"""

    def __init__(
        self,
        supervised_models: Mapping[str, dict],
        best_score_col: str | None = "nm_score",
        return_features: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Pandas implementation of supervised model(s) transformer

        PandasSupervisedLayerTransformer is the third (optional) step in the pipeline of PandasEntityMatching,
        after name preprocessing and name-pair candidate selection.
        PandasSupervisedLayerTransformer is used to score each candidate name-pair, and based on the scoring
        to pick the best ground truth name with each name-to-match.

        PandasSupervisedLayerTransformer uses one (or multiple) trained sklearn-based supervised model(s).
        Such a supervised model itself is a pipeline consisting of multiple steps. For example, by default:

        - PandasFeatureExtractor: calculation of custom edit-distance and rank-based features for each name-pair.
        - XBGClassifier: classification model to score each name-pair based on calculated features.

        For an example pipeline see `base_supervised_model.create_new_model_pipeline()`

        Args:
            supervised_models: supervised model dictionary with models used for scoring. Each model has a key and
                                  a dict containing the `model` and `enable` boolean flag.
            best_score_col: in case of several models, select name of best one. default is "nm_score".
            return_features: return generated input feature for supervised model. default is False.
            args: ignored.
            kwargs: ignored.

        Examples:
            A trained sklearn model needs to be provided in order to do scoring with transform(), see example below.
            The training of a supervised model is done in a separate step.
            See here `PandasEntityMatching.fit_classifier()` for details, or `base_supervised_model.train_model()`.

            >>> model = load_pickle("name_matching.pkl")
            >>> c = PandasSupervisedLayerTransformer(supervised_models={'nm_score': {'model': model, 'enable': True}})
            >>> scored_df = c.transform(candidates_df)

            When `return_features=True` the features calculated by CalcFeatures are also returned when calling transform().

            PandasSupervisedLayerTransformer can hold multiple sklearn-based supervised models (pipeline), in
            the `supervised_models` dictionary, which are each applied to score a name-pair candidate.
            The key of the best (or only) model is indicated with argument `best_score_col`.

            The `return_features=True` also works for an untrained supervised model. This model needs to be disabled.

            >>> from emm.supervised_model.base_supervised_model import create_new_model_pipeline
            >>>
            >>> # untrained pipeline
            >>> model = create_new_model_pipeline()
            >>> c = PandasSupervisedLayerTransformer(supervised_models={'X': {'model': model, 'enable': False}},
            >>>                                      return_features=True)
            >>> c.fit(ground_truth_df)
            >>> c.transform(candidates_df)
        """
        self.supervised_models = supervised_models
        self.return_features = return_features
        self.best_score_col = best_score_col
        BaseSupervisedModel.__init__(self)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PandasSupervisedLayerTransformer:
        """Fitting of CalcFeatures model of untrained supervised model.

        When an untrained supervised model has been provided, calling fit() updates the vocabularies of the
        CalcFeatures module, if that is present in the pipeline under key 'feat'.

        To update the vocabularies, provide a list of processed ground truth names.

        When this has been done, and `return_features=True`, then calling transform() returns the features
        calculated by CalcFeatures.

        Args:
            X: processed ground-truth names.
            y: ignored

        Returns:
            self
        """
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """Placeholder for `fit_transform`

        This avoids unnecessary transform `gt` during `SklearnPipeline.fit_transform(gt)`.

        (The sklearn Pipeline is doing fit_transform for all stages excluding the last one, and with supervised model
        the CandidateSelection stage is an intermediate step.)

        Args:
            X: input dataframe for fitting.
            y: ignored.
        """
        self.fit(X, y)

    def calc_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate the name-pair features.

        Append calculated features to the input dataframe
        """
        logger.info("calculcating sm features.")
        for model_col, model_dict in self.supervised_models.items():
            model = model_dict["model"]
            if "feat" in model.named_steps:
                feat = calc_features_from_sm(model, X, features_name="feat")
                feat = feat.rename(columns=lambda x: f"{model_col}_feat_{x}")
                for c in feat.columns:
                    X[c] = feat[c]
        return X

    def calc_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate the score using supervised model.

        Supervised model is run for each group on uid separately.
        """
        for model_col, model_dict in self.supervised_models.items():
            if not model_dict["enable"]:
                continue
            model = model_dict["model"]
            i_to_score = X["gt_uid"].notna()
            if i_to_score.sum() == 0:
                # No candidates to score, then just create the column
                X[model_col] = 0.0
            else:
                X.loc[i_to_score, model_col] = model.predict_proba(X[i_to_score])[:, 1]

        return X

    def select_best_score(
        self,
        X: pd.DataFrame,
        group_cols: list[str],
        best_score_col: str | None = "nm_score",
        sort_cols: list[str] | None = None,
        sort_asc: list[bool] | None = None,
        best_match_col: str = "best_match",
        best_rank_col: str = "best_rank",
        gt_uid_col: str | None = "gt_uid",
    ) -> pd.DataFrame:
        """Select final best score from supervised model (before penalty calculation).

        Returned dataframe will be sorted by group_cols + sort_cols to make it easier
        to calculate penalty.

        Args:
            X: pandas DataFrame with scores from supervised model
            group_cols: column name or list of column names used in aggregation
            best_score_col: sort these scores in descending order. default is "nm_score".
            sort_cols: (optional) list of columns used in ordering the results
            sort_asc: (optional) list of booleans to determine ascending order of sort_cols
            best_match_col: column indicating best match of all name-matching scores. "best_match".
            best_rank_col: column with rank of sorted scores. default is "best_rank".
            gt_uid_col: column indicating name of gt uid. default id "gt_uid_col".
        """
        # triviality checks
        if best_score_col not in self.supervised_models:
            return X
        model_dict = self.supervised_models[best_score_col]
        if not model_dict["enable"]:
            return X

        # best score available from here on
        if sort_cols is None:
            sort_cols = [best_score_col]
            sort_asc = [False]
        full_sort_by = group_cols + sort_cols
        assert sort_asc is not None
        full_sort_asc = [True] * len(group_cols) + sort_asc

        # rank the candidates based on best_score column. note that rank starts at 1
        # gt_uid is used for tie-breaking of identical nm_scores. descending, to make behaviour identical to pandas.
        X = X.sort_values(by=[*group_cols, best_score_col, gt_uid_col], ascending=False, na_position="last")
        # groupby preserves the order of the rows in each group. See:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html (sort)
        gb = X.groupby(group_cols)
        X[best_rank_col] = gb[best_score_col].transform(lambda x: range(1, len(x) + 1))

        # indicate the best match out of all candidates, also requires not-null and > 0.
        X[best_match_col] = (X[best_rank_col] == 1) & (X[best_score_col].notnull()) & (X[best_score_col] > 0)

        return X.sort_values(by=full_sort_by, ascending=full_sort_asc)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Supervised layer transformation for name matching of name-pair candidates.

        PandasSupervisedLayerTransformer is used to score each candidate name-pair, and based on the scoring
        to pick the best ground truth name with each name-to-match.

        When `return_features=True` calling transform() also returns the features calculated by CalcFeatures.

        Args:
            X: input name-pair candidates for scoring.

        Returns:
            candidates dataframe including the name-matching scoring column `nm_score`.
        """
        if X is None:
            return None

        with Timer("PandasSupervisedLayerTransformer.transform") as timer:
            timer.log_params({"X.shape": X.shape, "return_features": self.return_features})
            X = self.calc_score(X)
            X = self.select_best_score(X, best_score_col=self.best_score_col, group_cols=["uid"])

            if self.return_features:
                # note: does not require model to be enabled, only return_features=True.
                X = self.calc_features(X)

            timer.log_param("cands", len(X))
        return X
