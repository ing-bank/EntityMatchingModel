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

"""Spark Implementation of Sorted Neighbourhood Indexing (SNI)"""

from __future__ import annotations

from functools import reduce
from typing import Callable

import pyspark
import pyspark.sql.functions as F
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType, IntegerType, StringType

from emm.helper.spark_custom_reader_writer import SparkReadable, SparkWriteable
from emm.helper.spark_utils import set_spark_job_group
from emm.indexing.base_indexer import SNBaseIndexer
from emm.indexing.spark_indexing_utils import flatten_df, take_topn_per_group
from emm.loggers.logger import logger


class SparkSortedNeighbourhoodIndexer(
    Estimator, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, SNBaseIndexer
):
    """Unfitted spark estimator for sorted neighbourhood indexing"""

    def __init__(
        self,
        window_length: int,
        uid_col: str = "uid",
        index_col: str = "entity_id",
        name_col: str = "name",
        mapping_func: Callable | None = None,
        indexer_id: int | None = None,
        input_col: str = "preprocessed",
        output_col: str = "candidates",
        store_ground_truth: bool = True,
    ) -> None:
        """Unfitted spark estimator for sorted neighbourhood indexing.

        For generating name-pair candidates using sorted neighbourhood indexing.
        When fitted with ground truth names it returns SNIMatcherModel.

        The most important setting is "window_length".

        Args:
            window_length: size of SNI window (odd integer).
            uid_col: uid column, default is "uid".
            index_col: index column, default is "entity_id".
            name_col: name column, default is "name".
            mapping_func: python function that should be applied to names before SNI indexing (i.e. name reversal)
            indexer_id: optional index, used for bookkeeping in case of multiple spark indexers. default is None.
            input_col: spark input column, default is "preprocessed".
            output_col: spark output column, default is "candidates".
            store_ground_truth: store ground truth when calling write. default is True.

        Examples:
            >>> c = SparkSortedNeighbourhoodIndexer(window_length=5)
            >>> c.fit(ground_truth_sdf)
            >>> candidates_sdf = c.transform(names_sdf)

        """
        super().__init__()
        SNBaseIndexer.__init__(self, window_length=window_length)
        self._set(inputCol=input_col)
        self._set(outputCol=output_col)
        self.uid_col = uid_col
        self.index_col = index_col
        self.name_col = name_col
        self.mapping_func = mapping_func
        self.indexer_id = indexer_id
        self.store_ground_truth = store_ground_truth

        # if mapping_func is applied modifications are made to GT, always need to store GT at write()
        if mapping_func is not None and not store_ground_truth:
            logger.info("mapping_func is applied to ground truth; store_ground_truth set to True.")
            self.store_ground_truth = True

    def _fit(self, ground_truth_df: pyspark.sql.DataFrame) -> Model:
        """Default Estimator action on fitting with ground truth names.

        If custom mapping function is defined, then it is applied to names and
        the results are stored in `sni_name_mapping` column.

        Args:
            ground_truth_df: spark data frame with ground truth
        """
        assert self.uid_col in ground_truth_df.columns
        if self.mapping_func is not None:
            m_func_udf = F.udf(self.mapping_func, StringType())
            logger.info("calculating sni name mapping ground_truth_df")
            ground_truth_df = ground_truth_df.withColumn("sni_name_mapping", m_func_udf(self.getInputCol()))
        else:
            m_func_udf = None

        # Remove skewness: The ground-truth can have many duplicate name_preprocessed, generating then many candidates for one name to match.
        # This is problematic for memory usage (too many candidates can create out of memory errors on the supervised model stage).
        # When there is more than 10 duplicate gt_name_preprocessed, 2 possibilities:
        # - we drop all of them, because we have no way to decide which one are better
        # - we take top-10 random (is it really randomly?) ground-truth, which what is currently happening in cosine similarity
        # It is difficult to choose.
        ground_truth_df = take_topn_per_group(ground_truth_df, n=10, group="name")

        assert self.index_col in ground_truth_df.columns
        assert self.name_col in ground_truth_df.columns
        return SNIMatcherModel(
            ground_truth_df,
            self.window_length,
            self.uid_col,
            self.index_col,
            self.name_col,
            m_func_udf,
            self.getInputCol(),
            self.getOutputCol(),
            self.indexer_id,
            self.store_ground_truth,
        )


class SNIMatcherModel(
    Model,
    SparkReadable,
    SparkWriteable,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    SNBaseIndexer,
):
    """Already initialized spark model for SNI."""

    SERIALIZE_ATTRIBUTES = (
        "window_length",
        "uid_col",
        "index_col",
        "name_col",
        "mapping_func_udf",
        "indexer_id",
        "store_ground_truth",
        "_input_col",
        "_output_col",
        "_ground_truth_df",
    )

    def __init__(
        self,
        ground_truth_df: pyspark.sql.DataFrame | None = None,
        window_length: int = 3,
        uid_col: str = "uid",
        index_col: str = "entity_id",
        name_col: str = "name",
        mapping_func_udf: Callable | None = None,
        input_col: str = "preprocessed",
        output_col: str = "candidates",
        indexer_id: int | None = None,
        store_ground_truth: bool = True,
    ) -> None:
        """Already initialized spark model for SNI.

        See SparkSortedNeighbourhoodIndexer for details on usage.

        Args:
            ground_truth_df: spark data frame with ground truth names
            window_length: the size of indexing window (odd integer)
            uid_col: uid column, default is "uid".
            index_col: index column, default is "entity_id".
            name_col: name column, default is "name".
            mapping_func_udf: python function that should be applied to names before SNI indexing (i.e. name reversal)
            input_col: spark input column, default is "preprocessed".
            output_col: spark output column, default is "candidates".
            indexer_id: optional index, used for bookkeeping in case of multiple spark indexers. default is None.
            store_ground_truth: store ground truth when calling write. default is True.
        """
        super().__init__()
        SNBaseIndexer.__init__(self, window_length=window_length)
        self.ground_truth_df = ground_truth_df
        self.uid_col = uid_col
        self.index_col = index_col
        self.name_col = name_col
        self.mapping_func_udf = mapping_func_udf
        self._set(inputCol=input_col)._set(outputCol=output_col)
        self.indexer_id = indexer_id
        self.store_ground_truth = store_ground_truth

        # if mapping_func has been applied modifications are made to GT, if so always need to store GT at write()
        if mapping_func_udf is not None and not store_ground_truth:
            logger.info("mapping_func has been applied to ground truth; store_ground_truth set to True.")
            self.store_ground_truth = True

    def _transform(self, names_df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Default Model action on transforming names to match

        Args:
            names_df: spark data frame with names to match
        """
        logger.info(f"SparkCosSimMatcherModel._transform() : indexer_id = {self.indexer_id}")
        set_spark_job_group(
            "SNIMatcherModel._transform()",
            f"indexer_id={self.indexer_id}, window_length={self.window_length}, mapping_func_udf={self.mapping_func_udf}",
        )
        assert self.uid_col in self.ground_truth_df.columns
        assert self.uid_col in names_df.columns

        if self.mapping_func_udf is not None:
            names_df = names_df.withColumn("sni_name_mapping", self.mapping_func_udf(self.getInputCol()))
            sni_column = "sni_name_mapping"
        else:
            sni_column = self.getInputCol()

        # get all unique names with schema [name]
        # it looks like dropDuplicates does not require sorted data
        # and we need to sort after dropDuplicates because it randomly shuffles the data
        all_unique_names = (
            self.ground_truth_df.select(sni_column).union(names_df.select(sni_column)).dropDuplicates().sort(sni_column)
        )

        index_rdd = all_unique_names.rdd.zipWithIndex()
        index = index_rdd.toDF(["original", "_sni_rank"])
        # index_rdd from zipWithIndex has then 2 "columns" where the first one is a struct containing original columns
        index = flatten_df(index, ["original"], keep_root_name=False)
        index.cache()

        # join back the SNI rank to the ground_truth and names_to_match
        data_gt = self.ground_truth_df.join(index, on=sni_column)
        data_names = names_df.join(index, on=sni_column)

        results = []
        w = self.window_length // 2

        for i in range(-w, w + 1):
            logger.debug(f"SNI stage {i}")
            results.append(
                data_names.withColumn("_curr_rank", F.col("_sni_rank") + i)
                .select(F.col(self.uid_col), "_curr_rank")
                .withColumn("indexer_score", F.lit(1 - abs(i) / (w + 1)).cast(FloatType()))
                .withColumn("indexer_rank", F.lit(i).cast(IntegerType()))
                .withColumnRenamed("_curr_rank", "_sni_rank")
                .join(data_gt.select(F.col(self.uid_col).alias("gt_uid"), "_sni_rank"), on="_sni_rank")
                .drop("_sni_rank")
            )

        results = reduce(DataFrame.unionAll, results)
        index.unpersist(blocking=True)

        return results

    def calc_score(self, sdf: pyspark.sql.DataFrame, name1_col: str, name2_col: str) -> pyspark.sql.DataFrame:
        return sdf.withColumn("indexer_score", F.lit(0.0))

    @property
    def _input_col(self):
        return self.getInputCol()

    @property
    def _output_col(self):
        return self.getOutputCol()

    @property
    def _ground_truth_df(self):
        if self.store_ground_truth:
            return self.ground_truth_df

        return None
