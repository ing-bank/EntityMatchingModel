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
from typing import TYPE_CHECKING, Literal

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import col, lit
from pyspark.sql.pandas.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField

from emm.aggregation.base_entity_aggregation import BaseEntityAggregation, matching_max_candidate
from emm.helper.spark_custom_reader_writer import SparkReadable, SparkWriteable
from emm.helper.spark_utils import set_spark_job_group
from emm.loggers.logger import logger

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame


class SparkEntityAggregation(
    Transformer, SparkReadable, SparkWriteable, DefaultParamsReadable, DefaultParamsWritable, BaseEntityAggregation
):
    """Spark name-matching aggregation code"""

    SERIALIZE_ATTRIBUTES = (
        "score_col",
        "index_col",
        "uid_col",
        "freq_col",
        "output_col",
        "processed_col",
        "aggregation_method",
        "blacklist",
    )

    def __init__(
        self,
        score_col: str = "nm_score",
        index_col: str = "entity_id",
        uid_col: str = "uid",
        account_col: str = "account",
        name_col: str = "name",
        freq_col: str = "counterparty_account_count_distinct",
        output_col: str = "agg_score",
        preprocessed_col: str = "preprocessed",
        gt_name_col: str = "gt_name",
        gt_preprocessed_col: str = "gt_preprocessed",
        aggregation_method: Literal["max_frequency_nm_score", "mean_score"] = "max_frequency_nm_score",
        blacklist: list | None = None,
    ) -> None:
        """Spark name-matching aggregation code

        Last and optional step in SparkEntityMatching.

        Optionally, the EMM package can also be used to match a group of company names that belong together,
        to a company name in the ground truth. (For example, all names used to address an external bank account.)

        This step makes use of name-matching scores from the supervised layer. We refer to this as the aggregation step.
        (This step is not needed for standalone name matching.)

        The `account_col` column indicates which names-to-match belongs together.
        The combination of scores is based on `score_col`, e.g. the name-matching score `nm_score`.

        Two aggregation methods are available:
        - "mean_score": takes the mean score from all names-to-match to find the best ground-truth name.
        - "max_frequency_nm_score": weights the nm_score with the frequency and takes the maximum to find the best
            ground-truth name.

        Args:
            score_col: name-matching score "nm_score" or first cosine similarity score "score_0".
            index_col: id column, default is "entity_id".
            uid_col: uid column, default is "uid".
            account_col: account column, default is "account".
            name_col: name column, default is "name".
            freq_col: name frequency column, default is "counterparty_account_count_distinct".
            output_col: Name of column to store the final score
            preprocessed_col: Name of column of preprocessed input, default is "preprocessed".
            gt_name_col: ground truth name column, default is "gt_name".
            gt_preprocessed_col: column name of preprocessed ground truth names. default is "gt_preprocessed".
            aggregation_method: default is "max_frequency_nm_score", alternative is "mean_score".
            blacklist: blacklist of names to skip in clustering.
        """
        Transformer.__init__(self)
        BaseEntityAggregation.__init__(
            self,
            score_col=score_col,
            index_col=index_col,
            uid_col=uid_col,
            name_col=name_col,
            freq_col=freq_col,
            account_col=account_col,
            aggregation_method=aggregation_method,
            output_col=output_col,
            preprocessed_col=preprocessed_col,
            gt_name_col=gt_name_col,
            gt_preprocessed_col=gt_preprocessed_col,
            blacklist=blacklist or [],
        )

    def _transform(self, dataframe):
        """Combine scores of a group of name-pair candidates that belong together.

        Natch a group of company names that belong together, to a company name in the ground truth.

        Args:
            dataframe: dataframe of scored candidates

        Returns:
            dataframe of scored candidates, only one row per account
        """
        logger.info("SparkEntityAggregationTransformer._transform score_col = %s", self.score_col)
        set_spark_job_group("SparkEntityAggregationTransformer._transform()", f"score_col: {self.score_col}")

        group = self.get_group(dataframe)
        gt_group = self.get_gt_group()

        schema = copy.deepcopy(dataframe.select(group).schema)
        schema.add(StructField(self.gt_uid_col, IntegerType(), True))
        schema.add(StructField(self.gt_entity_id_col, IntegerType(), True))
        schema.add(StructField(self.output_col, FloatType(), True))
        schema.add(StructField(self.score_col, FloatType(), True))
        schema.add(StructField(self.name_col, StringType(), True))
        schema.add(StructField(self.preprocessed_col, StringType(), True))
        schema.add(StructField(self.freq_col, IntegerType(), True))
        schema.add(StructField(self.gt_name_col, StringType(), True))
        schema.add(StructField(self.gt_preprocessed_col, StringType(), True))

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def matching_max_candidate_wrapper(_, df) -> pd.DataFrame:
            df = matching_max_candidate(
                df,
                group=gt_group,
                score_col=self.score_col,
                name_col=self.name_col,
                account_col=self.account_col,
                freq_col=self.freq_col,
                output_col=self.output_col,
                aggregation_method=self.aggregation_method,
            )

            return df[[c.name for c in schema]]

        # remove all irrelevant non-matches before applying account matching
        dataframe = dataframe.filter(col(self.gt_uid_col).isNotNull())

        # filter out all processed names that are in blacklist or empty.
        dataframe = self.remove_blacklisted_names(df=dataframe, preprocessed_col=self.preprocessed_col)

        dataframe = dataframe.groupby(group).applyInPandas(
            matching_max_candidate_wrapper.func, schema=matching_max_candidate_wrapper.returnType
        )

        assert self.output_col in dataframe.columns

        # currently we leave only 1 row per account, so by definition it is the best match
        dataframe = dataframe.withColumn("best_match", lit(True))
        return dataframe.withColumn("best_rank", lit(1))

    def remove_blacklisted_names(self, df: DataFrame, preprocessed_col: str = "preprocessed") -> DataFrame:
        # filter out all processed names that are in blacklist or empty.
        # idea: these are too generic/not-good to use for account matching anyway.
        if preprocessed_col in df.columns:
            # preprocessed column should always be present
            return df.filter(~col(preprocessed_col).isin([*self.blacklist, ""]))

        return df
