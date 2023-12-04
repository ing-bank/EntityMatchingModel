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

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.window import Window

from emm.helper.spark_custom_reader_writer import SparkReadable, SparkWriteable
from emm.helper.spark_utils import set_partitions, set_spark_job_group
from emm.loggers.logger import logger
from emm.supervised_model.base_supervised_model import (
    BaseSupervisedModel,
    calc_features_from_sm,
    features_schema_from_sm,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class SparkSupervisedLayerEstimator(Estimator, DefaultParamsReadable, DefaultParamsWritable, BaseSupervisedModel):
    """Unfitted spark implementation of supervised model(s) estimator"""

    def __init__(
        self,
        supervised_models=None,
        return_features=False,
        preprocessed_col: str = "preprocessed",
        force_execution: bool = False,
    ) -> None:
        """Unfitted spark implementation of supervised model(s) estimator

        When fit, returns a SparkSupervisedLayerModel.

        SparkSupervisedLayerEstimator is the third (optional) step in the pipeline of SparkEntityMatching,
        after name preprocessing and name-pair candidate selection.
        SparkSupervisedLayerEstimator is used to score each candidate name-pair, and based on the scoring
        to pick the best ground truth name with each name-to-match.

        SparkSupervisedLayerEstimator uses one (or multiple) trained sklearn-based supervised model(s).
        Such a supervised model itself is a pipeline consisting of multiple steps. For example, by default:

        - PandasFeatureExtractor: calculation of custom edit-distance and rank-based features for each name-pair.
        - Scaler: scaling of all features used as input for classier.
        - XBGClassifier: classification model to score each name-pair based on calculated features.

        For an example pipeline see `base_supervised_model.create_new_model_pipeline()`

        Args:
            supervised_models: dictionary with models used for scoring. Each model has a key and
                                  a dict containing the `model` and `enable` boolean flag.
            return_features: return generated input feature for supervised model. default is False.
            preprocessed_col: name of preprocessed names column, default is "preprocessed".
            force_execution: if true, force spark execution after transform call.

        Examples:
            A trained sklearn model needs to be provided in order to do scoring with transform(), see example below.
            The training of a supervised model is done in a separate step.
            See here `SparkEntityMatching.fit_classifier()` for details, or `base_supervised_model.train_model()`.

            >>> model = load_pickle("name_matching.pkl")
            >>> c = SparkSupervisedLayerTransformer(supervised_models={'nm_score': {'model': model, 'enable': True}})
            >>> scored_sdf = c.transform(candidates_sdf)

            When `return_features=True` the features calculated by CalcFeatures are also returned when calling transform().

            SparkSupervisedLayerEstimator can hold multiple sklearn-based supervised models (pipeline), in
            the `supervised_models` dictionary, which are each applied to score a name-pair candidate.

            The `return_features=True` also works for an untrained supervised model. This model needs to be disabled.

            >>> from emm.supervised_model.base_supervised_model import create_new_model_pipeline
            >>>
            >>> # untrained pipeline
            >>> model = create_new_model_pipeline()
            >>> c = SparkSupervisedLayerEstimator(supervised_models={'X': {'model': model, 'enable': False}},
            >>>                              return_features=True)
            >>> c.fit(ground_truth_sdf)
            >>> c.transform(candidates_sdf)

        """
        super().__init__()
        self.supervised_models = supervised_models or {}
        self.return_features = return_features
        self.preprocessed_col = preprocessed_col
        self.force_execution = force_execution

    def _fit(self, dataset) -> SparkSupervisedLayerModel:
        """Fitting of CalcFeatures model of untrained (disabled) supervised model.

        When an untrained (disabled) supervised model X has been provided, calling fit() updates the vocabularies of the
        CalcFeatures module, if present in a sklearn pipeline under key 'feat'.

        To update the vocabularies, provide a list of processed ground truth names.

        When this has been done, and `return_features=True`, then calling transform() returns the features
        calculated by CalcFeatures.

        Args:
            dataset: processed ground-truth names.

        Returns:
            SparkSupervisedLayerModel
        """
        logger.info("SparkSupervisedLayerEstimator._fit()")
        return SparkSupervisedLayerModel(self.supervised_models, self.return_features, self.force_execution)


class SparkSupervisedLayerModel(Model, SparkReadable, SparkWriteable, DefaultParamsReadable, DefaultParamsWritable):
    """Fitted spark implementation of supervised model(s) estimator"""

    SERIALIZE_ATTRIBUTES = ("supervised_models", "return_features", "force_execution")

    def __init__(self, supervised_models, return_features: bool = False, force_execution=False) -> None:
        """Fitted spark implementation of supervised model(s) estimator

        See SparkSupervisedLayerEstimator for details on usage.

        Args:
            supervised_models: dictionary with models used for scoring. Each model has a key and
                                  a dict containing the `model` and `enable` boolean flag.
            return_features: return generated input feature for supervised model. default is False.
            force_execution: if true, force spark execution after transform call.
        """
        super().__init__()
        self.supervised_models = supervised_models
        self.return_features = return_features
        self.force_execution = force_execution

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        """Supervised layer transformation for name matching of name-pair candidates.

        SparkSupervisedLayerModel is used to score each candidate name-pair, and based on the scoring
        to pick the best ground truth name with each name-to-match.

        When `return_features=True` calling transform() also returns the features calculated by CalcFeatures.

        Args:
            dataframe: input name-pair candidates for scoring.

        Returns:
            candidates dataframe including the name-matching scoring column `nm_score`.
        """
        logger.info("SparkSupervisedLayerModel._transform()")
        set_spark_job_group("SparkSupervisedLayerModel._transform()", "")
        dataframe = dataframe.withColumn("partition_id", F.spark_partition_id())

        # add trained sm model scores (works when model enabled)
        dataframe = self.calc_score(dataframe)

        # add best_match column
        dataframe = self.select_best_score(dataframe, group_col="uid", best_score_col="nm_score")

        # add sm model input features, if so requested
        # (this also works when the model is not enabled.)
        if self.return_features:
            return self.calc_features(dataframe)

        if self.force_execution:
            logger.info("SparkSupervisedLayerModel._transform(): force execution.")
            _ = dataframe.count()

        return dataframe

    def calc_features(self, dataframe: DataFrame) -> DataFrame:
        """Calculate the name-pair features.

        Append calculated features to the input dataframe
        """
        schema = copy.deepcopy(dataframe.schema)
        for model_col, model_dict in self.supervised_models.items():
            for name, dtype in features_schema_from_sm(model_dict["model"], return_spark_types=True):
                schema.add(T.StructField(f"{model_col}_feat_{name}", dtype, True))

        @F.pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
        def run_cf_model(key, data) -> pd.DataFrame:
            for model_col, model_dict in self.supervised_models.items():
                sm = model_dict["model"]
                if len(data) > 0 and "feat" in sm.named_steps:
                    feat = calc_features_from_sm(sm, data, features_name="feat")
                    for name in feat.columns:
                        data[f"{model_col}_feat_{name}"] = pd.Series(feat[name].values, index=data.index)
                else:
                    for f in schema.fields:
                        if f.name.startswith(f"{model_col}_feat_"):
                            data[f.name] = pd.Series([], index=[])
            return data

        # apply function
        num_partitions = dataframe.rdd.getNumPartitions()
        set_partitions(num_partitions)
        return dataframe.groupby(dataframe.partition_id).applyInPandas(
            run_cf_model.func, schema=run_cf_model.returnType
        )

    def calc_score(self, dataframe: DataFrame) -> DataFrame:
        """Calculate the score using supervised model.

        Supervised model is run for each group on uid separately.
        """
        schema = copy.deepcopy(dataframe.schema)
        for model_col, model_dict in self.supervised_models.items():
            if not model_dict["enable"]:
                continue
            schema.add(T.StructField(model_col, T.FloatType(), True))

        """
        Using simple withColumn Pandas UDF cannot work because Data partitions in Spark are then converted into Arrow record batches,
        which makes it difficult to enforce uid consistency for rank features.

        So we use: pyspark.sql.GroupedData.applyInPandas
        "This function requires a full shuffle. All the data of a group will be loaded into memory,
        so the user should be aware of the potential OOM risk if data is skewed and certain groups are too large to fit in memory."
        To control this we disable spark.sql.adaptive.enabled, and repartition manually ourself, see logical_repartitioning().

        With spark.sql.adaptive.enabled it was merging partitions, because too small, then we had 834 partitions, each containing 1.2M candidates, 5MB in parquet, 172MB in Pandas.
        Those big partition were running for 30 min to 1 hour, which was not good for parallelize and preemption loss.
        """

        @F.pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
        def run_score_model(key, data) -> pd.DataFrame:
            for model_col, model_dict in self.supervised_models.items():
                if not model_dict["enable"]:
                    continue
                sm = model_dict["model"]
                raw_preds = sm.predict_proba(data)[:, 1] if len(data) > 0 else np.array([], dtype="float64")
                preds = pd.Series(raw_preds, index=data.index, name="nm_score")
                data[model_col] = preds
                data[model_col] = data.apply(lambda x: None if pd.isnull(x["gt_entity_id"]) else x[model_col], axis=1)
            return data

        # applyInPandas is the new API function, apply is going to be deprecated
        # groupby give number of partition based on spark.sql.shuffle.partitions,
        # so let's set it correctly and hope for no shuffling.
        num_partitions = dataframe.rdd.getNumPartitions()
        set_partitions(num_partitions)
        return dataframe.groupby(dataframe.partition_id).applyInPandas(
            run_score_model.func, schema=run_score_model.returnType
        )

    @staticmethod
    def select_best_score(
        df: DataFrame,
        group_col: str | None = "uid",
        best_score_col: str | None = "nm_score",
        best_rank_col: str | None = "best_rank",
        best_match_col: str | None = "best_match",
        gt_uid_col: str | None = "gt_uid",
    ) -> DataFrame:
        """Select final best score from supervised model (before penalty calculation).

        Returned dataframe will be sorted by group_cols + sort_cols to make it easier
        to calculate penalty.

        Args:
            df: pandas DataFrame with scores from supervised model
            group_col: column name used in aggregation. default is "uid".
            best_score_col: sort these scores in descending order. default is "nm_score".
            best_rank_col: column with rank of sorted scores. default is "best_rank".
            best_match_col: column indicating best match of all name-matching scores. default is "best_match".
            gt_uid_col: column indicating name of gt uid. default id "gt_uid_col".

        Returns:
            dataframe with best scoring name pairs
        """
        if any(col not in df.columns for col in [best_score_col, group_col]):
            logger.debug(f"Column {best_score_col} and/or {group_col} not in dataframe, cannot add best_match.")
            return df

        logger.info("Marking best name-pair candidate matches.")
        # gt_uid is used for tie-breaking of identical nm_scores. descending, to make behaviour identical to pandas.
        window = Window.partitionBy(group_col).orderBy([F.col(best_score_col).desc(), F.col(gt_uid_col).desc()])
        df = df.withColumn(best_rank_col, F.row_number().over(window))
        # indicate the best match out of all candidates, also requires not-null and > 0.
        return df.withColumn(
            best_match_col,
            (F.col(best_rank_col) == 1) & F.col(best_score_col).isNotNull() & (F.col(best_score_col) > 0),
        )
