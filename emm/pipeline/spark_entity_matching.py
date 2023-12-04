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
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import numpy as np
import pandas as pd
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from emm.aggregation.base_entity_aggregation import BaseEntityAggregation
from emm.aggregation.spark_entity_aggregation import SparkEntityAggregation
from emm.data.prepare_name_pairs import prepare_name_pairs
from emm.helper.io import IOFunc
from emm.helper.spark_custom_reader_writer import SparkReadable, SparkWriteable
from emm.helper.spark_ml_pipeline import EMPipeline
from emm.helper.spark_utils import auto_repartitioning, check_uid, set_partitions, set_spark_job_group
from emm.indexing.base_indexer import BaseIndexer
from emm.indexing.spark_candidate_selection import SparkCandidateSelectionEstimator
from emm.indexing.spark_cos_sim_matcher import SparkCosSimIndexer
from emm.indexing.spark_sni import SparkSortedNeighbourhoodIndexer
from emm.loggers.logger import logger
from emm.parameters import DEFAULT_CARRY_ON_COLS, MODEL_PARAMS
from emm.pipeline.base_entity_matching import BaseEntityMatching
from emm.preprocessing.base_name_preprocessor import AbstractPreprocessor
from emm.preprocessing.spark_preprocessor import SparkPreprocessor
from emm.supervised_model.base_supervised_model import train_model
from emm.supervised_model.spark_supervised_model import SparkSupervisedLayerEstimator

if TYPE_CHECKING:
    from pyspark.ml import Pipeline, PipelineModel


class SparkEntityMatching(
    SparkReadable, SparkWriteable, BaseEntityMatching, DefaultParamsReadable, DefaultParamsWritable
):
    """Spark implementation of EntityMatching"""

    SERIALIZE_ATTRIBUTES = ("create_pipeline", "parameters", "supervised_models", "model")

    def __init__(
        self,
        parameters: dict | None = None,
        create_pipeline: bool | None = True,
        supervised_models: dict | None = None,
        name_col: str | None = None,
        entity_id_col: str | None = None,
        name_only: bool | None = None,
        preprocessor: str | None = None,
        indexers: list | None = None,
        supervised_on: bool | None = None,
        without_rank_features: bool | None = None,
        with_legal_entity_forms_match: bool | None = None,
        return_sm_features: bool | None = None,
        supervised_model_object: Any | None = None,
        aggregation_layer: bool | None = None,
        aggregation_method: str | None = None,
        carry_on_cols: list[str] | None = None,
        model: PipelineModel | None = None,
        **kwargs,
    ) -> None:
        """Spark implementation of EntityMatching

        EntityMatching object is a pipeline consisting of:
        - Preprocessor: cleaning and standardization of input names and their legal entity forms.
        - Candidate selection: generation of name-pair candidates, known as `indexing`, using list of indexers.
        - Supervised model (optional): classification of each name-pair, to pick the best name-pair candidate.
        - Aggregation (optional): combine a group of company names that belong together to match to ground truth.

        Below are the most common kw arguments. For complete list see `emm.config`.
        key-word arguments (besides parameters and supervised_models) are optional and update the `parameters` dictionary.

        Args:
            parameters: a dictionary with algorithm parameters, missing values would be filled in by default values from `emm.config`
            create_pipeline: create the EMM pipeline. default is True.
            supervised_models: optional dictionary of pretrained models.
            name_col: name column in dataframe. default is "name".
            entity_id_col: id column in dataframe. default is "id".
            name_only: Use only name-based features for name-matching, no extra features like country. default is False.
            preprocessor: preprocessor or processing configuration for name cleaning. default is "preprocess_merge_abbr".
            indexers: list of indexers or indexer settings. default is [word-based cossim, 2-char cossim, sni].
            supervised_on: if true provide trained (or else instantiate untrained) supervised model. default is False.
            without_rank_features: if True ignore rank based features in model. default is False.
            with_legal_entity_forms_match: if True, add match of legal entity forms feature
            return_sm_features: if True returns supervised model features in transform() call.
               This also works when supervised_on=True but no trained supervised model is present. default is False.
            supervised_model_object: provide a trained supervised model. default is None.
            aggregation_layer: if True, turn on aggregation later. Default if False.
            aggregation_method: aggregation method: 'name_clustering' or 'mean_score'. Default is 'name_clustering'.
            carry_on_cols: list of column names that should be copied to the dataframe with candidates (optional)
            model: the PipelineModel
            kwargs: extra key-word arguments are passed on to parameters' dictionary.

        Examples:
            >>> em = SparkEntityMatching(name_only=True)
            >>> em.fit(ground_truth_sdf)
            >>> matches_sdf = em.transforms(names_sdf)
            >>>
            >>> em.fit_classifier(matching_names_sdf)
        """
        # copy known model-parameter arguments into parameters dict
        function_locals = locals()
        model_parameters = {
            key: function_locals.get(key) for key in MODEL_PARAMS if function_locals.get(key) is not None
        }
        if parameters is None:
            parameters = {}
        parameters.update({**model_parameters, **kwargs})
        BaseEntityMatching.__init__(self, parameters=parameters, supervised_models=supervised_models)

        # set (missing) parameters of indexers
        self.parameters["indexers"] = self._indexers_set_default_values(self.parameters["indexers"])
        self.initialize(create_pipeline)

        # Default: model is set during fit(), but may be passed as individual kwarg.
        self.model = model

    def initialize(self, create_pipeline: bool = True):
        """If you updated parameters of EntityMatching, you might want to initialize again."""
        for i, idx_params in enumerate(self.parameters["indexers"]):
            if not isinstance(idx_params, dict):
                continue
            idx_params["indexer_id"] = i

        # Let's define sm, even if we don't create the pipeline, so we can serialize/deserialize without Spark
        if self.parameters["supervised_on"] is not False:
            self._initialize_supervised_models()

        self.pipeline = None
        if create_pipeline:
            # To create the pipeline we need Spark (because of RegexTokenizer) which we don't have when we set threshold_curves
            self._create_pipeline()

    @staticmethod
    def _create_single_indexer(params, type=None):
        if type == "sni":
            return (
                SparkSortedNeighbourhoodIndexer(
                    window_length=params["window_length"],
                    mapping_func=params["mapping_func"],
                    indexer_id=params["indexer_id"],
                    store_ground_truth=False,
                )
                ._set(outputCol="candidates")
                ._set(inputCol="preprocessed")
            )
        return SparkCosSimIndexer(parameters=params)

    def _create_multiple_indexers(self, params):
        if params["uid_col"] is None:
            msg = "Multiple indexers requires uid_col parameter"
            raise ValueError(msg)

        indexers = []
        for idx_params in params["indexers"]:
            if isinstance(idx_params, dict):
                if idx_params["type"] not in {"cosine_similarity", "sni"}:
                    msg = f"idx_params.type={idx_params['type']} not supported yet"
                    raise ValueError(msg)
                idx = self._create_single_indexer(
                    {
                        **params,
                        **idx_params,  # values from idx_params override all default parameters
                    },
                    type=idx_params["type"],
                )
                indexers.append(idx)
            elif isinstance(idx_params, BaseIndexer):
                # indexer already instantiated
                indexers.append(idx_params)

        return SparkCandidateSelectionEstimator(
            indexers=indexers,
            force_execution=params.get("force_execution", False),
            unpersist_broadcast=params.get("unpersist_broadcast", False),
            with_no_matches=params.get("with_no_matches", True),
            carry_on_cols=list(set(DEFAULT_CARRY_ON_COLS + params.get("carry_on_cols", []))),
        )

    def _create_pipeline(self) -> Pipeline:
        """Build the Spark-ML pipeline object"""
        stages = []

        # step 1: Preprocessor
        preprocessor = self.parameters["preprocessor"]
        if isinstance(preprocessor, AbstractPreprocessor):
            self.pipeline_preprocessor = preprocessor
        else:
            self.pipeline_preprocessor = SparkPreprocessor(preprocessor)
        stages += [self.pipeline_preprocessor]
        # step 2: Candidate name-pair selection (= indexing)
        self.pipeline_candidate_selection = self._create_multiple_indexers(self.parameters)
        stages += [self.pipeline_candidate_selection]
        # step 3: classifier model (= name matching)
        if self.parameters["supervised_on"]:
            # Disable multiprocessing in Spark, because it is using pandarallel and it is copying the memory for each process.
            self._disable_multiprocessing_all_models()
            self.pipeline_supervised_layer = SparkSupervisedLayerEstimator(
                self.supervised_models,
                return_features=self.parameters["return_sm_features"],
                force_execution=self.parameters.get("force_execution", False),
            )
            stages += [self.pipeline_supervised_layer]
        else:
            self.pipeline_supervised_layer = None

        # step 4: aggregation of name scores (= account matching)
        # Remark: We can have aggregation layer without the supervised layer, since we could develop an aggregation based on indexers score only.
        aggregation_layer = self.parameters.get("aggregation_layer", False)
        if isinstance(aggregation_layer, BaseEntityAggregation):
            self.pipeline_entity_aggregation = aggregation_layer
            stages += [self.pipeline_entity_aggregation]
        elif aggregation_layer:
            self.pipeline_entity_aggregation = SparkEntityAggregation(
                score_col="nm_score" if self.parameters["supervised_on"] else "score_0",
                aggregation_method=self.parameters["aggregation_method"],
                blacklist=self.parameters.get("aggregation_blacklist", []),
            )
            stages += [self.pipeline_entity_aggregation]
        else:
            self.pipeline_entity_aggregation = None
        self.pipeline = EMPipeline(stages=stages)
        return self.pipeline

    def fit(self, ground_truth_df, copy_ground_truth: bool = False) -> SparkEntityMatching:
        """Fits name indexers on ground truth data.

        Fit excludes the supervised model, which needs training list of names-to-match.
        See instead: cls.fit_classifier()

        Args:
            ground_truth_df: spark dataframe with ground truth names and corresponding ids.
            copy_ground_truth: if true, keep a link to the ground truth, useful for storage of the model.

        Returns:
            self reference
        """
        logger.info("SparkEntityMatching.fit()")
        set_spark_job_group(
            "Fit", f"Fit and broadcast model (ground truth matrix) to workers. Parameters: {self.parameters}"
        )

        self._check_relevant_columns_present(ground_truth_df, ground_truth=True)

        if isinstance(ground_truth_df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            ground_truth_df = spark.createDataFrame(ground_truth_df)

        # We repartition in order to have at least 200, to have a nice parallel computation
        # (assuming memory is not an issue here) and nice parallelism for joins in transform() later on.
        # We usually have less than 200 partitions in case the ground_truth is not that long.
        ground_truth_df, self.n_ground_truth = auto_repartitioning(ground_truth_df, self.parameters["partition_size"])
        ground_truth_df = check_uid(ground_truth_df, self.parameters["uid_col"])
        ground_truth_df = self._normalize_column_names(ground_truth_df)
        self.model = self.pipeline.fit(ground_truth_df)

        if copy_ground_truth:
            self.ground_truth_df = ground_truth_df

        return self

    def transform(self, names_df: DataFrame, top_n: int = -1) -> DataFrame:
        """Matches given names against ground truth.

        transform() returns a spark dataframe with name-pair candidates.

        Args:
            names_df: dataframe with names to be matched.
            top_n: return top-n candidates per name to match, top-n > 0. -1 returns all candidates. default is -1.

        Returns:
            dataframe with candidate name-pairs
        """
        logger.info("SparkEntityMatching.transform()")
        set_spark_job_group("Transform", f"Match names. Parameters: {self.parameters}")

        self._check_relevant_columns_present(names_df)
        names_df = check_uid(names_df, self.parameters["uid_col"])
        names_df = self._normalize_column_names(names_df)

        # for streaming we don't need to repartition (plus we can't do any actions)
        if self.parameters["streaming"]:
            n_names = names_df.rdd.countApprox(timeout=20)
        else:
            names_df, n_names = auto_repartitioning(names_df, self.parameters["partition_size"])
            num_partitions = names_df.rdd.getNumPartitions()
            # update num_partitions of candidate_selection_model
            self.model.stages[1].num_partitions = num_partitions
            if num_partitions > 200:
                # If bigger than default value update this to have the number partitions kept after join() and groupby()
                set_partitions(num_partitions)

        logger.info(f"Matching {n_names} records against ground-truth with size {self.n_ground_truth}.")
        matched_df = self.model.transform(names_df)

        if not self.parameters["keep_all_cols"]:
            # Drop all intermediary columns like (token, ngram_tokens, tf, etc)
            # but keep the columns in names_df, preprocessed (useful for training), score_*, rank_*, nm_score, nm_score_feat_*, agg_score, gt_*
            cols_list = ["gt_", "score_", "rank_", "best_"]
            if self.parameters["supervised_on"]:
                cols_list += list(self.supervised_models.keys())
            cols_regex = "|".join(cols_list)
            regex = rf"^({cols_regex}).*"
            pattern = re.compile(regex)

            cols_to_keep = names_df.columns
            cols_to_drop = [c for c in matched_df.columns if c not in cols_to_keep]
            cols_to_drop = [
                c for c in cols_to_drop if not re.match(pattern, c) and not c.endswith("_score") and c != "preprocessed"
            ]
            matched_df = matched_df.drop(*cols_to_drop)
            logger.debug(f"Dropping columns: {cols_to_drop}")

        if isinstance(top_n, int) and top_n > 0 and "best_rank" in matched_df.columns:
            return matched_df.filter((F.col("best_rank") <= top_n) & (F.col("gt_uid").isNotNull()))

        return matched_df

    def create_training_name_pairs(
        self,
        train_positive_names_to_match,
        create_negative_sample_fraction=0,
        n_train_ids=-1,
        random_seed=42,
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
            msg = "indexer pipeline not yet fit."
            raise TypeError(msg)

        # reduce training sample? (no need for too many training names)
        # do reduction based on id to avoid signal leakage
        if n_train_ids > 0:
            id_col = self.parameters["entity_id_col"]
            ids = sorted(train_positive_names_to_match.select(id_col).distinct().toPandas()[id_col].values)
            if len(ids) > n_train_ids:
                # make a random sub-selection of ids
                logger.info("Reducing training set down to %d ids through random selection.", len(ids))
                rng = np.random.default_rng(random_seed)
                ids = list(rng.choice(ids, n_train_ids, replace=False))
                train_positive_names_to_match = train_positive_names_to_match.filter(
                    train_positive_names_to_match[id_col].isin(ids)
                )

        # negative sample creation?
        create_negative_sample_fraction = min(create_negative_sample_fraction, 1)
        create_negative_sample = create_negative_sample_fraction > 0
        # prepare training candidate name-pair data
        if create_negative_sample:
            # increase indexing window size, needed for negative sample creation,
            # used & corrected during prepare_dataset()
            self.increase_window_by_one_step()
        candidates = self.transform(train_positive_names_to_match)
        if create_negative_sample:
            # reset indexers back to normal settings
            self.decrease_window_by_one_step()

        # create training sample from name-pair candidates.
        # this creates the negative names, add labels, and returns a pandas dataframe.
        return prepare_name_pairs(
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
        train_positive_names_to_match=None,
        train_name_pairs=None,
        create_negative_sample_fraction=0,
        n_train_ids=-1,
        random_seed=42,
        train_gt=None,
        store_key="nm_score",
        train_function=train_model,
        score_columns=None,
        n_jobs=1,
        drop_duplicate_candidates=None,
        extra_features: list[str | tuple[str, Callable]] | None = None,
        **fit_kws,
    ) -> SparkEntityMatching:
        """Function to train the supervised model based on positive input names.

        Positive names are names that are supposed to match to the ground truth.
        A fraction of the positive names can be converted to negative names, which are not supposed to match to the
        ground truth.

        Args:
            train_positive_names_to_match: spark dataframe of positive names to match for training. A positive name
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
            train_gt: spark dataframe of ground truth names and ids for training the indexers. By default we assume
                         the the indexers have already been fit. default is None (optional).
            store_key: storage key for new supervised model. default is 'nm_score'.
            train_function: provide custom function to create and train model pipeline. optional.
            score_columns: list of columns with raw scores from indexers to pass to classifier.
                              default is None, meaning all indexer scores (e.g. cosine similarity values).
            n_jobs: number of parallel jobs passed on to model. Default for spark is 1.
            drop_duplicate_candidates: if True drop any duplicate training candidates and keep just one,
                            if available keep the correct match. Recommended for string-similarity models, eg. with
                            without_rank_features=True. default is False.
            extra_features: list of columns (and possibly functions) used for extra features calculation,
                            e.g. country if name_only=False, default is None.
                            With ``name_only=False`` internally ``extra_features=['country']``.
            fit_kws: extra kwargs passed on to model fit function. optional.

        Returns:
            self (object including the trained supervised model)
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
            # reset and refit the indexers to new gt.
            # supervised model is turned off as it will be trained below
            self.parameters["supervised_on"] = False
            self._create_pipeline()
            logger.debug(f"training indexers, using following params: {self.parameters}")
            # this creates the fitted model: self.model
            self.fit(train_gt)

        # bookkeeping 1/2
        # if present remove existing supervised model and aggregation layer before transform
        # keep both stages for re-adding later (also in case of do_training=False).
        if self.parameters.get("supervised_on", False):
            self.model.stages.pop(2)
        aggregation_model = self.model.stages.pop() if self.parameters.get("aggregation_layer", False) else None
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
            n_jobs=n_jobs,
            extra_features=extra_features,
            **fit_kws,
        )

        # add new supervised model to self.model pipeline
        self.parameters["supervised_on"] = True
        self.parameters["supervised_model_object"] = model
        self._add_supervised_model(model_key=store_key, overwrite=True)
        # this init call enables all known supervised models (same as pandas version)
        self._initialize_supervised_models()
        self._disable_multiprocessing_all_models()
        self.pipeline_supervised_layer = SparkSupervisedLayerEstimator(
            self.supervised_models, return_features=self.parameters["return_sm_features"]
        )
        # dummy call, this simply creates the spark _model_
        sm_model = self.pipeline_supervised_layer.fit(dataset=None)

        # bookkeeping 2/2
        # recreate untrained pipeline with updated settings for consistency.
        self._create_pipeline()
        # reinsert (new) fitted supervised model into unfitted pipeline and the fitted model
        if sm_model is not None:
            idx = len(self.model.stages)
            self.model.stages.insert(idx, sm_model)
        # re-add aggregation layer into fitted pipeline
        if aggregation_model is not None:
            if aggregation_model.score_col != store_key:
                logger.info(f'updating aggregation score column to new model "{store_key}"')
                aggregation_model.score_col = store_key
            self.model.stages.append(aggregation_model)
        return self

    def add_supervised_model(
        self,
        path: str | None = None,
        model: Pipeline | None = None,
        name_only: bool = True,
        store_key: str = "nm_score",
        overwrite: bool = True,
        return_features: bool | None = None,
    ) -> None:
        """Add trained sklearn supervised model to spark supervised layer

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

        # if present, remove existing spark supervised model from trained and untrained pipelines
        # reinsert again below with new sklearn model included.
        if self.parameters.get("supervised_on", False):
            self.model.stages.pop(2)
        aggregation_model = self.model.stages.pop() if self.parameters.get("aggregation_layer", False) else None

        # add new supervised model to self.supervised_models
        # self.supervised_models contains all trained and untrained sklearn models
        self.parameters["supervised_on"] = True
        self.parameters["supervised_model_filename"] = path
        self.parameters["supervised_model_object"] = model
        self.parameters["name_only"] = name_only
        self._add_supervised_model(model_key=store_key, overwrite=overwrite)

        # this init call enables all known supervised models (same as pandas version)
        self._initialize_supervised_models()
        self._disable_multiprocessing_all_models()
        # update parameter settings
        if return_features is not None:
            self.parameters["return_sm_features"] = return_features
        supervised_layer = SparkSupervisedLayerEstimator(
            self.supervised_models, return_features=self.parameters["return_sm_features"]
        )
        # dummy call, this simply creates the "trained" spark supervised model that includes new sklearn model
        sm_model = supervised_layer.fit(dataset=None)

        # recreate untrained pipeline with updated settings for consistency.
        self._create_pipeline()
        # reinsert (new) spark supervised layer into existing fitted spark pipeline
        self.model.stages.insert(2, sm_model)
        # re-add aggregation layer into fitted pipeline with updated score column
        if aggregation_model is not None:
            if aggregation_model.score_col != store_key:
                logger.info(f'updating aggregation score column to new model "{store_key}"')
                aggregation_model.score_col = store_key
            self.model.stages.append(aggregation_model)

    def add_aggregation_layer(
        self,
        score_col: str | None = None,
        account_col: str | None = None,
        freq_col: str | None = None,
        aggregation_method: Literal["max_frequency_nm_score", "mean_score"] | None = None,
        blacklist: list | None = None,
        aggregation_layer: BaseEntityAggregation | None = None,
    ) -> None:
        """Add or replace aggregation layer to spark pipeline

        Args:
            score_col: name-matching score "nm_score" (default) or e.g. first cosine similarity score "score_0".
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
            self.model.stages.pop(-1)

        # create new aggregation layer.
        if aggregation_layer is None:
            self.parameters["aggregation_layer"] = True
            if score_col is None:
                score_col = "nm_score" if self.parameters.get("supervised_on", False) else "score_0"
            if account_col is not None:
                self.parameters["account_col"] = account_col
            if freq_col is not None:
                self.parameters["freq_col"] = freq_col
            if aggregation_method is not None:
                self.parameters["aggregation_method"] = aggregation_method
            if blacklist is not None:
                self.parameters["aggregation_blacklist"] = blacklist
            aggregation_layer = SparkEntityAggregation(
                score_col=score_col,
                aggregation_method=self.parameters["aggregation_method"],
                blacklist=self.parameters.get("aggregation_blacklist", []),
            )
        elif isinstance(aggregation_layer, BaseEntityAggregation):
            self.parameters["aggregation_layer"] = True
        else:
            msg = "aggregation_layer does not have type BaseEntityAggregation "
            raise TypeError(msg)

        # recreate untrained pipeline with updated settings for consistency.
        self._create_pipeline()
        # insert (new) aggregation layer at end of fitted and unfitted spark pipelines
        self.model.stages.append(aggregation_layer)

    def _unpersist(self):
        """If you want to run multiple experiments with multiple indexer,
        then you will have multiple broadcasted object that might use to much memory.
        We tried to use unpersist() but it didn't solve the memory issue.
        Conclusion: Don't use unpersist, just restart a new Spark Session.
        """
        for stage in self.model.stages:
            if hasattr(stage, "_unpersist"):
                stage._unpersist()

    def increase_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        if self.model is not None and len(self.model.stages) >= 2:
            stage = self.model.stages[1]
            stage.increase_window_by_one_step()

    def decrease_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        if self.model is not None and len(self.model.stages) >= 2:
            stage = self.model.stages[1]
            stage.decrease_window_by_one_step()

    def set_return_sm_features(self, return_features=True):
        """Toggle setting to return supervised model features

        Args:
            return_features: bool to to return supervised model features, default is True.
        """
        self.parameters["return_sm_features"] = return_features
        if self.model is not None and self.parameters.get("supervised_on", False):
            sm = self.model.stages[2]
            sm.return_features = return_features
        if self.pipeline_supervised_layer is not None:
            self.pipeline_supervised_layer.return_features = return_features

    @property
    def create_pipeline(self):
        """Trigger creation of unfitted pipeline at initialization (default)"""
        return True

    @classmethod
    def load(
        cls,
        emo_path: str,
        load_func: Callable | None = None,
        override_parameters: Mapping[str, Any] | None = None,
        create_pipeline: bool | None = True,
        name_col: str | None = None,
        entity_id_col: str | None = None,
        **kwargs,
    ) -> object:
        """Deserialize the persisted EMM object.

        Reads an instance from the input path, a shortcut of `read().load(path)`.

        Below are the most common key-word arguments. For complete list see `emm.config`.
        Extra key-word arguments are optional and update the `override_parameters` dict.

        Args:
            emo_path: path to the EMM pickle file.
            load_func: function used for loading of non-spark objects. default is joblib.load()
            override_parameters: parameters that overwrite the settings of the EMM object. optional.
            create_pipeline: create the EMM pipeline. default is True.
            name_col: name column in dataframe. default is "name".
            entity_id_col: id column in dataframe. default is "id".
            kwargs: extra key-word arguments are passed on to parameters dictionary.

        Returns:
            instantiated EMM object.

        Examples:
            >>> # deserialize pickled EMM object and rename name column
            >>> em = SparkEntityMatching.load(emo_path, name_col='Name', entity_id_col='Id')

        """
        # copy known model-parameter arguments into parameters dict
        function_locals = locals()
        model_parameters = {
            key: function_locals.get(key) for key in MODEL_PARAMS if function_locals.get(key) is not None
        }
        if override_parameters is None:
            override_parameters = {}
        override_parameters.update({**model_parameters, **kwargs})

        if callable(load_func):
            IOFunc().set_reader(load_func)

        # load the spark em object
        emobj = cls.read().load(emo_path)

        # update emm parameters, such as names of relevant columns
        emobj.parameters.update(override_parameters)

        return emobj
