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

from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from emm.aggregation.base_entity_aggregation import BaseEntityAggregation
from emm.aggregation.pandas_entity_aggregation import PandasEntityAggregation
from emm.data.prepare_name_pairs import prepare_name_pairs_pd
from emm.helper.io import IOFunc
from emm.helper.sklearn_pipeline import SklearnPipelineWrapper
from emm.helper.util import string_columns_to_pyarrow
from emm.indexing.base_indexer import BaseIndexer
from emm.indexing.pandas_candidate_selection import PandasCandidateSelectionTransformer
from emm.indexing.pandas_cos_sim_matcher import PandasCosSimIndexer
from emm.indexing.pandas_naive_indexer import PandasNaiveIndexer
from emm.indexing.pandas_sni import PandasSortedNeighbourhoodIndexer
from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.parameters import DEFAULT_CARRY_ON_COLS, MODEL_PARAMS
from emm.pipeline.base_entity_matching import BaseEntityMatching
from emm.preprocessing.base_name_preprocessor import AbstractPreprocessor
from emm.preprocessing.pandas_preprocessor import PandasPreprocessor
from emm.supervised_model.base_supervised_model import BaseSupervisedModel, train_model
from emm.supervised_model.pandas_supervised_model import PandasSupervisedLayerTransformer

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline


class PandasEntityMatching(BaseEntityMatching):
    """Implementation of EntityMatching using Pandas."""

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        supervised_models: Mapping[str, Any] | None = None,
        name_col: str | None = None,
        entity_id_col: str | None = None,
        name_only: bool | None = None,
        preprocessor: str | None = None,
        indexers: list | None = None,
        supervised_on: bool | None = None,
        without_rank_features: bool | None = None,
        with_legal_entity_forms_match: bool | None = None,
        return_sm_features: bool | None = None,
        supervised_model_object: Pipeline | None = None,
        aggregation_layer: bool | None = None,
        aggregation_method: Literal["mean_score", "max_frequency_nm_score"] | None = None,
        carry_on_cols: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Implementation of EntityMatching using Pandas dataframes as a data format.

        EntityMatching object is a pipeline consisting of:
        - Preprocessor: cleaning and standardization of input names and their legal entity forms.
        - Candidate selection: generation of name-pair candidates, known as `indexing`, using list of indexers.
        - Supervised model (optional): classification of each name-pair, to pick the best name-pair candidate.
        - Aggregation (optional): combine a group of company names that belong together to match to ground truth.

        Below are the most common arguments. For complete list see `emm.parameters.MODEL_PARAMS`.
        key-word arguments (besides parameters and supervised_models) are optional and update the `parameters` dictionary.

        Args:
            parameters: a dictionary with custom EMM parameters, missing values filled in by default values from `emm.config`
            supervised_models: optional dictionary of pretrained models.
            name_col: name column in dataframe. default is "name".
            entity_id_col: id column in dataframe. default is "id".
            name_only: Use only name-based features for name-matching, no extra features like country. default is False.
            preprocessor: Preprocessor or processing configuration for name cleaning. default is "preprocess_merge_abbr".
            indexers: list of indexers or indexer settings. default is [word-based cossim, 2-char cossim, sni].
            supervised_on: if true provide trained (or else instantiate untrained) supervised model. default is False.
            without_rank_features: if True ignore rank based features in model. default is False.
            with_legal_entity_forms_match: if True, add match of legal entity forms feature
            return_sm_features: if True returns supervised model features in transform() call.
               This also works when supervised_on=True but no trained supervised model is present. default is False.
            supervised_model_object: provide a trained supervised model. default is None.
            aggregation_layer: if True, turn on aggregation later. Default if False.
            aggregation_method: aggregation method: 'mean_score' or 'max_frequency_nm_score'.
            n_jobs: desired number of parallel jobs in pandas candidate selection. default is all cores.
            carry_on_cols: list of column names that should be copied to the dataframe with candidates (optional)
            kwargs: extra key-word arguments are passed on to parameters dictionary.

        Examples:
            >>> em = PandasEntityMatching(name_only=True)
            >>> em.fit(ground_truth_df)
            >>> matches = em.transforms(names_df)
            >>>
            >>> em.fit_classifier(matching_names_df)
        """
        # copy known model-parameter arguments into parameters dict
        function_locals = locals()
        model_parameters = {
            key: function_locals.get(key) for key in MODEL_PARAMS if function_locals.get(key) is not None
        }
        if parameters is None:
            parameters = {}
        parameters.update({**model_parameters, **kwargs})
        super().__init__(parameters=parameters, supervised_models=supervised_models)

        self.model: TransformerMixin | None = None
        self.initialize()

    def initialize(self):
        """If you updated parameters of EntityMatching, you might want to initialize again."""
        self.pipeline = self._create_pipeline()

    def _create_preprocessor(self) -> TransformerMixin:
        params = self.parameters
        preprocessor = params["preprocessor"]
        if isinstance(preprocessor, AbstractPreprocessor):
            return preprocessor
        return PandasPreprocessor(preprocess_pipeline=preprocessor, spark_session=params.get("spark_session"))

    def _create_indexers(self) -> list[TransformerMixin]:
        params = self.parameters
        INDEXER_CLASS = {
            "cosine_similarity": PandasCosSimIndexer,
            "sni": PandasSortedNeighbourhoodIndexer,
            "naive": PandasNaiveIndexer,
        }
        DEFAULT_INDEXER_PARAMS_PANDAS = {
            "cosine_similarity": {
                "input_col": "preprocessed",
                "spark_session": params.get("spark_session"),
                "n_jobs": params.get("n_jobs", -1),
            },
            "sni": {"input_col": "preprocessed"},
            "naive": {},
        }
        if "indexers" in params:
            indexers_definition = params["indexers"]
        else:
            indexers_definition = []
            for c in ["sni", "cosine_similarity"]:
                if params[c]:
                    if isinstance(params[c], list):
                        for elem in params[c]:
                            indexers_definition.append({"type": c, **elem})
                    else:
                        indexers_definition.append({"type": c})

        indexers_definition = self._indexers_set_default_values(indexers_definition)
        indexers = []
        for curr_d in indexers_definition:
            if isinstance(curr_d, dict):
                d = curr_d.copy()
                t = d["type"]
                del d["type"]
                kwargs = {**DEFAULT_INDEXER_PARAMS_PANDAS[t], **d}
                indexers.append(INDEXER_CLASS[t](**kwargs))
            elif isinstance(curr_d, BaseIndexer):
                # indexer already instantiated
                indexers.append(curr_d)
        return indexers

    def _create_candidate_selection_step(self, indexers: list[BaseIndexer] | None = None) -> TransformerMixin | None:
        if indexers is None:
            indexers = self._create_indexers()
        if len(indexers) == 0:
            return None

        return PandasCandidateSelectionTransformer(
            indexers=indexers,
            uid_col="entity_id",
            carry_on_cols=list(set(DEFAULT_CARRY_ON_COLS + self.parameters.get("carry_on_cols", []))),
            with_no_matches=self.parameters.get("with_no_matches", True),
        )

    def _create_supervised_step(self) -> BaseSupervisedModel | None:
        """Creates supervised layer."""
        if self.parameters["supervised_on"] is not False:
            # this init call enables all known supervised models
            self._initialize_supervised_models()
            return PandasSupervisedLayerTransformer(
                self.supervised_models, return_features=self.parameters["return_sm_features"]
            )
        return None

    def _create_aggregation_step(self) -> BaseEntityAggregation | None:
        aggregation_layer = self.parameters.get("aggregation_layer", False)
        if isinstance(aggregation_layer, BaseEntityAggregation):
            return aggregation_layer
        if aggregation_layer:
            return PandasEntityAggregation(
                score_col="nm_score" if self.parameters["supervised_on"] else "score_0",
                freq_col=self.parameters["freq_col"],
                aggregation_method=self.parameters["aggregation_method"],
                blacklist=self.parameters.get("aggregation_blacklist", []),
            )
        return None

    def _create_pipeline(self) -> Pipeline:
        """Creates sklearn pipeline with the model.

        Returns:
            pipeline object with the full model (preprocessing, candidate selection, supervised layer)
        """
        steps = [
            ("preprocess", self._create_preprocessor()),
            ("candidate_selection", self._create_candidate_selection_step()),
            ("supervised", self._create_supervised_step()),
            ("aggregation", self._create_aggregation_step()),
        ]
        # drop skipped steps represented by None values
        steps = [(name, step) for (name, step) in steps if step is not None]

        return SklearnPipelineWrapper(steps)

    def fit(self, ground_truth_df: pd.DataFrame, copy_ground_truth: bool = False) -> PandasEntityMatching:
        """Fits name indexers on ground truth data.

        Fit excludes the supervised model, which needs training list of names that match to the ground truth.
        See instead: cls.fit_classifier().

        Args:
            ground_truth_df: spark dataframe with ground truth names and corresponding ids.
            copy_ground_truth: if true, keep a copy of the ground truth, useful for storage of the model.

        Returns:
            self reference (for compatibility with sklearn models)
        """
        with Timer("PandasEntityMatching.fit") as timer:
            self._check_relevant_columns_present(ground_truth_df, ground_truth=True)
            ground_truth_df = self._normalize_column_names(ground_truth_df)
            ground_truth_df = string_columns_to_pyarrow(df=ground_truth_df)
            if copy_ground_truth:
                self.ground_truth_df = ground_truth_df.copy()
            self.model = self.pipeline.fit(ground_truth_df)
            self.n_ground_truth = len(ground_truth_df)

            timer.log_param("n", self.n_ground_truth)

        return self

    def transform(self, names_df: pd.DataFrame | pd.Series, top_n: int = -1) -> pd.DataFrame:
        """Matches given names against ground truth.

        transform() returns a pandas dataframe with name-pair candidates.

        Args:
            names_df: dataframe or series with names to be matched.
            top_n: return top-n candidates per name to match, top-n > 0. -1 returns all candidates. default is -1.

        Returns:
            dataframe with candidate name-pairs
        """
        if self.model is None:
            msg = "indexing pipeline has not been trained. Did you already fit()?"
            raise TypeError(msg)

        with Timer("PandasEntityMatching.transform") as timer:
            if isinstance(names_df, pd.Series):
                names_df = pd.DataFrame(names_df).copy()

            self._check_relevant_columns_present(names_df)
            names_df = self._normalize_column_names(names_df)

            # only relevant normalized columns for current setup
            columns = ["name"]
            if "entity_id" in names_df.columns:
                columns += ["entity_id"]
            if "country" in names_df.columns:
                columns += ["country"]
            if self.parameters["aggregation_layer"]:
                columns += ["account", "counterparty_account_count_distinct"]
            # keep all carry-on columns that are found
            if self.parameters.get("carry_on_cols", []):
                extra_cols = [c for c in self.parameters["carry_on_cols"] if c not in columns and c in names_df.columns]
                columns += extra_cols

            # convert string columns to pyarrow
            names_df = string_columns_to_pyarrow(df=names_df, columns=columns)

            names_to_match = names_df[columns]
            logger.info(f"Matching {len(names_to_match)} records against ground-truth with size {self.n_ground_truth}.")

            res = self.model.transform(names_to_match)

            if isinstance(top_n, int) and top_n > 0 and "best_rank" in res.columns:
                res = res[(res["best_rank"] <= top_n) & (res["gt_uid"].notnull())]
            timer.log_param("cands", len(res))

        return res

    def create_training_name_pairs(
        self,
        train_positive_names_to_match: pd.DataFrame,
        create_negative_sample_fraction: float = 0,
        n_train_ids: int = -1,
        random_seed: int = 42,
        drop_duplicate_candidates: bool | None = None,
    ) -> pd.DataFrame:
        """Create name-pairs for training from positive names that match to the ground truth.

        Positive names are names that are supposed to match to the ground truth.
        A fraction of the positive names can be converted to negative names, which are not supposed to match to the
        ground truth.

        Args:
            train_positive_names_to_match: pandas dataframe of positive names to match for training. A positive name
                                              has a guaranteed match to a name in the ground truth. Two columns are
                                              needed: a name and id (to determine a corresponding match to the
                                              ground truth).
            create_negative_sample_fraction: fraction of ids converted to negative names. A negative name has
                                                guaranteed no match to any name in the ground truth. default is 0:
                                                no negative names are created.
            n_train_ids: down-sample the positive names to match, keep only n_train_ids number of ids.
                            default value is -1 (keep all).
            random_seed: random seed for down-sampling of ids. default is 42.
            drop_duplicate_candidates: if True drop any duplicate training candidates and keep just one,
                            if available keep the correct match. Recommended for string-similarity models, eg. with
                            without_rank_features=True. default is False.

        Returns:
            pandas dataframe with name-pair candidates to be used for training.
        """
        if self.model is None:
            msg = "indexer pipeline not yet fit and train_gt not provided to do so."
            raise TypeError(msg)

        # reduce training sample? (no need for too many training names)
        # do reduction based on id to avoid signal leakage
        if n_train_ids > 0:
            id_col = self.parameters["entity_id_col"]
            ids = sorted(train_positive_names_to_match[id_col].unique())
            if len(ids) > n_train_ids:
                # make a random sub-selection of ids
                logger.info(f"Reducing training set down to {len(ids)} ids through random selection.")
                rng = np.random.default_rng(random_seed)
                ids = list(rng.choice(ids, n_train_ids, replace=False))
                train_positive_names_to_match = train_positive_names_to_match[
                    train_positive_names_to_match[id_col].isin(ids)
                ].copy()

        # negative sample creation?
        create_negative_sample_fraction = min(create_negative_sample_fraction, 1)
        create_negative_sample = create_negative_sample_fraction > 0
        # prepare training candidate name-pair data
        logger.info(
            "generating training candidates (len(train_positive_names_to_match)=%d)", len(train_positive_names_to_match)
        )
        if create_negative_sample:
            # increase indexing window size, needed for negative sample creation,
            # used & corrected during prepare_dataset_pd()
            self.increase_window_by_one_step()
        candidates = self.transform(train_positive_names_to_match)

        candidates = candidates.drop(columns=["name", "gt_name"]).rename(columns={"score": "score_0"})
        if create_negative_sample:
            # reset indexers back to normal settings
            self.decrease_window_by_one_step()

        # create training sample from name-pair candidates.
        # this creates the negative names, add labels, and returns a pandas dataframe.
        return prepare_name_pairs_pd(
            candidates,
            drop_duplicate_candidates=self.parameters.get("drop_duplicate_candidates", False)
            if drop_duplicate_candidates is None
            else drop_duplicate_candidates,
            create_negative_sample_fraction=create_negative_sample_fraction,
            positive_set_col=self.parameters.get("positive_set_col", "positive_set"),
            random_seed=random_seed,
        )

    def fit_classifier(
        self,
        train_positive_names_to_match: pd.DataFrame | None = None,
        train_name_pairs=None,
        create_negative_sample_fraction: float = 0,
        n_train_ids: int = -1,
        random_seed: int = 42,
        train_gt: pd.DataFrame | None = None,
        store_key="nm_score",
        train_function=train_model,
        score_columns=None,
        drop_duplicate_candidates: bool | None = None,
        extra_features: list[str | tuple[str, Callable]] | None = None,
        **fit_kws,
    ) -> PandasEntityMatching:
        """Function to train the supervised model based on positive input names.

        Positive names are names that are supposed to match to the ground truth.
        A fraction of the positive names can be converted to negative names, which are not supposed to match to the
        ground truth.

        Args:
            train_positive_names_to_match: pandas dataframe of positive names to match for training. A positive name
                                              has a guaranteed match to a name in the ground truth. Two columns are
                                              needed: a name and id (to determine a corresponding match to the
                                              ground truth).
            train_name_pairs: pandas dataframe with training name pair candidates, an alternative to
                                 train_positive_names_to_match. When not provided, train name pairs are
                                 created from positive names to match using self.create_training_name_pairs().
                                 default is None (optional.)
            create_negative_sample_fraction: fraction of ids converted to negative names. A negative name has
                                                guaranteed no match to any name in the ground truth. default is 0:
                                                no negative names are created.
            n_train_ids: down-sample the positive names to match, keep only n_train_ids number of ids.
                            default value is -1 (keep all).
            random_seed: random seed for down-sampling of ids. default is 42.
            train_gt: pandas dataframe of ground truth names and ids for training the indexers. By default we assume
                         the the indexers have already been fit. default is None (optional).
            store_key: storage key for new supervised model. default is 'nm_score'.
            train_function: provide custom function to create and train model pipeline. optional.
            score_columns: list of columns with raw scores from indexers to pass to classifier.
                              default is None, meaning all indexer scores (e.g. cosine similarity values).
            drop_duplicate_candidates: if True drop any duplicate training candidates and keep just one,
                            if available keep the correct match. Recommended for string-similarity models, eg. with
                            without_rank_features=True. default is False.
            extra_features: list of columns (and possibly functions) used for extra features calculation,
                            e.g. country if name_only=False, default is None.
                            With ``name_only=False`` internally ``extra_features=['country']``.
            fit_kws: extra kwargs passed on to model fit function. optional.

        Returns:
            self reference (object including the trained supervised model)
        """
        if not callable(train_function):
            msg = f'training function "{train_function}" is not callable.'
            raise TypeError(msg)

        if self.model is None and train_gt is None:
            msg = "indexer pipeline not yet fit and train_gt not provided to do so."
            raise TypeError(msg)
        if train_positive_names_to_match is None and train_name_pairs is None:
            msg = "Must provide either positive training names or training candidate name-pairs."
            raise TypeError(msg)

        if train_gt is not None:
            # reset and refit the indexers to new gt. supervised model is turned off.
            self.parameters["supervised_on"] = False
            self.pipeline = self._create_pipeline()
            logger.debug("training using following params: %s", self.parameters)
            logger.info("fitting on train gt (len(train_gt)=%d", len(train_gt))
            # this creates the fitted model: self.model
            self.fit(train_gt)

        # bookkeeping 1/2
        # if present remove existing supervised model and aggregation layer before transform
        # only want to call the indexing which makes the candidate name-pairs we want to fit.
        # keep both steps for re-adding later (e.g. in case of no training).
        if "supervised" in self.model.named_steps:
            self.model.steps.pop(2)
        aggregation_step = None
        if "aggregation" in self.model.named_steps:
            aggregation_step = self.model.steps.pop()
        # remove any existing untrained model 'X', no longer needed.
        if isinstance(self.supervised_models, dict):
            self.supervised_models.pop("X", None)

        # create training sample of name-pair candidates.
        if train_positive_names_to_match is not None:
            logger.info("Making candidate name-pairs from positive names to match.")
            train_pd = self.create_training_name_pairs(
                train_positive_names_to_match,
                create_negative_sample_fraction,
                n_train_ids=n_train_ids,
                random_seed=random_seed,
                drop_duplicate_candidates=drop_duplicate_candidates,
            )
        else:
            train_pd = train_name_pairs

        # train supervised model
        model = train_function(
            train_pd,
            without_rank_features=self.parameters.get("without_rank_features", False),
            name_only=self.parameters.get("name_only", False),
            positive_set_col=self.parameters.get("positive_set_col", "positive_set"),
            score_columns=score_columns,
            with_legal_entity_forms_match=self.parameters.get("with_legal_entity_forms_match", False),
            extra_features=extra_features,
            **fit_kws,
        )
        # add new supervised model to self.model pipeline
        self.parameters["supervised_on"] = True
        self.parameters["supervised_model_object"] = model
        self._add_supervised_model(model_key=store_key, overwrite=True)
        sm_step = ("supervised", self._create_supervised_step())

        # bookkeeping 2/2
        # reinsert (new/old) supervised model into unfitted pipeline and fitted model
        # note: inserting in self.model also updates self.pipeline, they are the same.
        if sm_step is not None:
            idx = len(self.model.steps)
            self.model.steps.insert(idx, sm_step)
        # re-add aggregation layer into fitted pipeline
        if aggregation_step is not None:
            if aggregation_step[1].score_col != store_key:
                logger.info(f'updating aggregation score column to new model "{store_key}"')
                aggregation_step[1].score_col = store_key
            self.model.steps.append(aggregation_step)

        return self

    def test_classifier(self, test_names_to_match: pd.DataFrame, test_gt: pd.DataFrame | None = None):
        """Helper function for testing the supervised model.

        Print multiple ML model metrics.

        Args:
            test_names_to_match: test dataframe with names (and ids) to match.
            test_gt: provide alternative GT. optional, default is None.
        """
        if self.model is None or self.parameters["supervised_on"] is False:
            msg = "No supervised model available."
            raise TypeError(msg)
        if test_gt is None and self.ground_truth_df is None:
            msg = "No ground truth names available."
            raise TypeError(msg)
        if test_gt is None:
            test_gt = self.ground_truth_df

        def combine_sm_results(df: pd.DataFrame, sel_cand: pd.DataFrame, test_gt: pd.DataFrame) -> pd.DataFrame:
            res = df.join(sel_cand[["gt_entity_id", "gt_name", "gt_preprocessed", "nm_score", "score_0"]], how="left")
            res["nm_score"] = res["nm_score"].fillna(-1)
            res["score_0"] = res["score_0"].fillna(-1)
            is_in_pos = res["id"].isin(test_gt["id"])
            res["correct"] = ((is_in_pos) & (res["id"] == res["gt_entity_id"])) | ((~is_in_pos) & (res["id"].isnull()))
            return res

        test_candidates = self.transform(test_names_to_match.copy())
        cand_after_sm = test_candidates[test_candidates.best_match].set_index("uid", drop=True)
        results_after_sm = combine_sm_results(test_names_to_match, cand_after_sm, test_gt)
        logger.info(
            "AUC of the supervised model: %.4f",
            roc_auc_score(results_after_sm["correct"], results_after_sm["nm_score"]),
        )

    def add_supervised_model(
        self,
        path: str | None = None,
        model: Pipeline | None = None,
        name_only: bool = True,
        store_key: str = "nm_score",
        overwrite: bool = True,
        return_features: bool | None = None,
    ) -> None:
        """Add trained sklearn supervised model to existing pipeline

        Args:
            path: file path of pickled sklearn pipeline. Or provide model directly.
            model: trained sklearn pipeline to add to spark supervised layer.
            name_only: name-only model? If false, presence of extra features (country) is checked. Default is True.
            store_key: storage key for new sklearn supervised model. default is 'nm_score'.
            overwrite: overwrite existing model if store_key already used, default is True.
            return_features: bool to to return supervised model features. None means default: False.
        """
        if path is None and model is None:
            msg = "Need to provided either path to trained model or model itself."
            raise TypeError(msg)
        if self.model is None:
            msg = "indexer pipeline not yet fit. Cannot add supervised layer."
            raise TypeError(msg)

        # if present remove existing spark supervised model from trained and untrained pipelines
        # reinsert again below with new sklearn model included.
        if self.parameters.get("supervised_on", False):
            self.model.steps.pop(2)
        aggregation_step = self.model.steps.pop() if self.parameters.get("aggregation_layer", False) else None

        # add new supervised model to self.supervised_models
        # self.supervised_models contains all trained and untrained sklearn models
        self.parameters["supervised_on"] = True
        self.parameters["supervised_model_filename"] = path
        self.parameters["supervised_model_object"] = model
        self.parameters["name_only"] = name_only
        self._add_supervised_model(model_key=store_key, overwrite=overwrite)

        # this init call enables all known supervised models (same as pandas version)
        self._initialize_supervised_models()
        # update parameter settings
        if return_features is not None:
            self.parameters["return_sm_features"] = return_features
        sm_step = ("supervised", self._create_supervised_step())

        # reinsert (new/old) supervised model into pipeline
        # note: inserting in self.model also updates self.pipeline, they are the same.
        if sm_step is not None:
            idx = len(self.model.steps)
            self.model.steps.insert(idx, sm_step)
        # re-add aggregation layer into fitted pipeline
        if aggregation_step is not None:
            if aggregation_step[1].score_col != store_key:
                logger.info(f'updating aggregation score column to new model "{store_key}"')
                aggregation_step[1].score_col = store_key
            self.model.steps.append(aggregation_step)

    def add_aggregation_layer(
        self,
        account_col: str | None = None,
        freq_col: str | None = None,
        aggregation_method: str | None = None,
        blacklist: list | None = None,
        aggregation_layer: BaseEntityAggregation | None = None,
    ) -> None:
        """Add or replace aggregation layer to spark pipeline

        Args:
            account_col: `account_col` column indicates which names-to-match belongs together. default is "account".
            freq_col: name frequency column, default is "counterparty_account_count_distinct".
            aggregation_method: aggregation method: 'name_clustering' or 'mean_score'. Default is 'name_clustering'.
            blacklist: blacklist of names to skip in clustering.
            aggregation_layer: existing aggregation layer to add. Default is None, if so one is created.
        """
        if self.model is None:
            msg = "indexer pipeline not yet fit."
            raise TypeError(msg)

        # remove existing aggregation layer if present. add new one below.
        if self.parameters.get("aggregation_layer", False):
            self.model.steps.pop(-1)

        # create a new aggregation layer
        if aggregation_layer is None:
            self.parameters["aggregation_layer"] = True
            if account_col is not None:
                self.parameters["account_col"] = account_col
            if freq_col is not None:
                # freq column matches with counterparty_account_count_distinct
                self.parameters["freq_col"] = freq_col
            if aggregation_method is not None:
                self.parameters["aggregation_method"] = aggregation_method
            if blacklist is not None:
                self.parameters["aggregation_blacklist"] = blacklist
        elif isinstance(aggregation_layer, BaseEntityAggregation):
            self.parameters["aggregation_layer"] = aggregation_layer
        else:
            msg = "aggregation_layer does not have type BaseEntityAggregation"
            raise TypeError(msg)
        aggregation_layer = ("aggregation", self._create_aggregation_step())

        # insert (new) aggregation layer at the end of fitted and unfitted spark pipelines
        self.model.steps.append(aggregation_layer)

        # configure candidate selector to pass on relevant features for aggregation
        candidate_selector = self.model.steps[1][1]
        if "account" not in candidate_selector.carry_on_cols:
            candidate_selector.carry_on_cols.append("account")
        if "counterparty_account_count_distinct" not in candidate_selector.carry_on_cols:
            candidate_selector.carry_on_cols.append("counterparty_account_count_distinct")

    def increase_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        if self.model is not None and "candidate_selection" in self.model.named_steps:
            step = self.model.named_steps["candidate_selection"]
            step.increase_window_by_one_step()

    def decrease_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        if self.model is not None and "candidate_selection" in self.model.named_steps:
            step = self.model.named_steps["candidate_selection"]
            step.decrease_window_by_one_step()

    def set_return_sm_features(self, return_features=True):
        """Toggle setting to return supervised model features

        Args:
            return_features: bool to return supervised model features, default is True.
        """
        self.parameters["return_sm_features"] = return_features
        if self.model is not None and "supervised" in self.model.named_steps:
            sm = self.model.named_steps["supervised"]
            sm.return_features = return_features

    def save(self, emo_path: str, dump_func: Callable = IOFunc().writer):
        """Serialize the EMM object.

        Args:
            emo_path: path to the EMM pickle file.
            dump_func: function used for dumping self. default is joblib.dump() with compression turned on.
        """
        if self.model is None:
            msg = "indexer pipeline not yet fit. Nothing useful to store."
            raise TypeError(msg)

        # Avoid storage of spark_session
        spark_session = self.parameters.pop("spark_session", None)
        # Avoid duplicate storage of ground truth
        ground_truth_df = self.ground_truth_df
        self.ground_truth_df = None
        # turn off GT for SNI indexers. GT is kept in the candidate selector.
        cand_selector = self.model.steps[1][1]
        cand_selector._reset_sni_ground_truth()

        # persist self.
        dump_func(self, emo_path)

        # restore spark_session
        if spark_session is not None:
            self.parameters["spark_session"] = spark_session
        # set ground truth settings back again
        self.ground_truth_df = ground_truth_df
        cand_selector._set_sni_ground_truth()

    @staticmethod
    def load(
        emo_path: str,
        load_func: Callable = IOFunc().reader,
        override_parameters: Mapping[str, Any] | None = None,
        name_col: str | None = None,
        entity_id_col: str | None = None,
        **kwargs,
    ) -> object:
        """Load the EMM object.

        Below are the most common arguments. For complete list see `emm.parameters.MODEL_PARAMS`.
        These arguments are optional and update the `parameters` dictionary.

        Args:
            emo_path: path to the EMM pickle file.
            load_func: function used for loading object. default is joblib.load()
            override_parameters: parameters that overwrite the settings of the EMM object. optional.
            name_col: name column in dataframe. default is "name".
            entity_id_col: id column in dataframe. default is "id".
            kwargs: extra key-word arguments are passed on to parameters dictionary.

        Returns:
            instantiated EMM object

        Examples:
            >>> # deserialize pickled EMM object and rename name column
            >>> em = PandasEntityMatching.load(emo_path, name_col='Name', entity_id_col='Id')

        """
        # copy known model-parameter arguments into parameters dict
        function_locals = locals()
        model_parameters = {
            key: function_locals.get(key) for key in MODEL_PARAMS if function_locals.get(key) is not None
        }
        if override_parameters is None:
            override_parameters = {}
        override_parameters.update({**model_parameters, **kwargs})

        # load the pandas em object
        emobj = load_func(emo_path)

        # turn on GT for any SNI indexers.
        # (GT is kept in the candidate selector.)
        cand_selector = emobj.model.steps[1][1]
        cand_selector._convert_ground_truth_to_pyarrow()
        cand_selector._set_sni_ground_truth()

        # update emm parameters, such as names of relevant columns
        emobj.parameters.update(override_parameters)

        return emobj
