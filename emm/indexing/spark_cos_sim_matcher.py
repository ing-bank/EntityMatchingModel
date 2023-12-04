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

"""Cosine Similarity Matcher"""

from __future__ import annotations

import contextlib
from itertools import islice
from sys import getsizeof
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from pyspark.ml import Estimator, Model, Pipeline
from pyspark.ml.feature import CountVectorizer, NGram
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.types import ArrayType, FloatType, LongType, StringType, StructField, StructType
from sparse_dot_topn import awesome_cossim_topn

from emm.helper.spark_custom_reader_writer import SparkReadable, SparkWriteable
from emm.helper.spark_utils import set_spark_job_group
from emm.indexing.base_indexer import BaseIndexer, CosSimBaseIndexer
from emm.indexing.spark_character_tokenizer import SparkCharacterTokenizer
from emm.indexing.spark_indexing_utils import (
    as_matrix,
    collect_matrix,
    curry,
    dot_product,
    explode_candidates,
    groupby,
    stack_features,
)
from emm.indexing.spark_normalized_tfidf import SparkNormalizedTfidfVectorizer
from emm.indexing.spark_word_tokenizer import SparkWordTokenizer
from emm.loggers.logger import logger

if TYPE_CHECKING:
    import pyspark

dot_product_udf = F.udf(dot_product, FloatType())


class SparkCosSimIndexer(
    Estimator, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, BaseIndexer
):
    """Unfitted Cosine similarity indexer to generate candidate name-pairs of possible matches"""

    def __init__(
        self,
        parameters: dict | None = None,
        tokenizer: Literal["words", "characters"] = "words",
        ngram: int = 1,
        binary_countvectorizer: bool = False,
        num_candidates: int = 2,
        cos_sim_lower_bound: float = 0.5,
        max_features: int = 2**25,
        blocking_func: Callable[[str], str] | None = None,
        streaming: bool = False,
        indexer_id: int | None = None,
        keep_all_cols: bool = False,
        n_threads: int = 1,
    ) -> None:
        """Unfitted Cosine similarity indexer to generate candidate name-pairs of possible matches

        When fitted with ground truth names it returns SparkCosSimIndexerModel.

        Pipeline of tokenization, ngram creation, vectorization, tfidf. There is a separate cosine similarity step.
        The vectorizer used is a customized Spark Tfidf Vectorizer.
        Cosine similarity is calculated in fast manner using ING's dedicated sparse-dot-topn library.

        The most important settings are: tokenizer, ngram, num_candidates and cos_sim_lower_bound.

        Args:
            parameters: dictionary with settings of the cossim indexer. (all arguments below in a dictionary.)
                           Can use arguments below instead. default is None.
            tokenizer: tokenization used, either "words" or "characters". default is "words".
            ngram: number of n-grams used in name tokenization. default is 1. (for characters we recommend 2.)
            binary_countvectorizer: use binary_countvectorizer in spark's TfidfVectorizer. default is False.
            num_candidates: maximum number of candidates per name-to-match. default is 2.
            cos_sim_lower_bound: lower bound on cosine similarity values of name-pairs. default is 0.5.
            max_features: maximum number of features used by TfidfVectorizer. default is 2**25.
            blocking_func: blocking function for matching of names (e.g. block on first character). default is None.
            streaming: use spark streaming, default is False. (So use batching.)
            indexer_id: optional index, used for bookkeeping in case of multiple spark indexers. default is None.
            keep_all_cols: keep all columns with info coming out of vectorizer. default is False.
            n_threads: number of threads for spark worker parallelization of matrix multiplication. default is 1.

        Examples:
            >>> c = SparkCosSimIndexer(
            >>>    tokenizer="words",
            >>>    ngram=1,
            >>>    num_candidates=10,
            >>>    binary_countvectorizer=True,
            >>>    cos_sim_lower_bound=0.2,
            >>> )
            >>>
            >>> c.fit(ground_truth_sdf)
            >>> candidates_sdf = c.transform(names_sdf)

        """
        super().__init__()

        if parameters is None:
            parameters = {
                "tokenizer": tokenizer,
                "ngram": ngram,
                "binary_countvectorizer": binary_countvectorizer,
                "num_candidates": num_candidates,
                "cos_sim_lower_bound": cos_sim_lower_bound,
                "max_features": max_features,
                "blocking_func": blocking_func,
                "streaming": streaming,
                "indexer_id": indexer_id,
                "keep_all_cols": keep_all_cols,
                "n_threads": n_threads,
            }

        self.parameters = parameters

        self.vectorizer = self._create_pipeline()
        # WARN: besides calling .fit() the Pipeline might also call .transform() if there
        # would be another Estimator later in the pipeline. We don't want this.
        # https://spark.apache.org/docs/latest/ml-pipeline.html#pipeline-components
        self.cossim = (
            SparkCosSimMatcher(
                num_candidates=parameters.get("num_candidates", 10),
                cos_sim_lower_bound=parameters["cos_sim_lower_bound"],
                streaming=parameters["streaming"],
                blocking_func=parameters["blocking_func"],
                indexer_id=parameters["indexer_id"],
                n_threads=parameters.get("n_threads", 1),
            )
            ._set(outputCol="candidates")
            ._set(inputCol="features")
        )

    def _create_pipeline(self) -> Pipeline:
        if self.parameters["tokenizer"] == "words":
            tokenizer = SparkWordTokenizer(inputCol="preprocessed", outputCol="tokens")
        else:
            tokenizer = SparkCharacterTokenizer(inputCol="preprocessed", outputCol="tokens")

        return Pipeline(
            stages=[
                tokenizer,
                NGram(inputCol="tokens", outputCol="ngram_tokens", n=self.parameters["ngram"]),
                CountVectorizer(
                    inputCol="ngram_tokens",
                    outputCol="tf",
                    vocabSize=self.parameters["max_features"],
                    binary=self.parameters["binary_countvectorizer"],
                ),
                SparkNormalizedTfidfVectorizer(
                    count_col="tf",
                    token_col="ngram_tokens",
                    output_col="features",
                    binary_countvectorizer=self.parameters["binary_countvectorizer"],
                ),
            ]
        )

    def _fit(self, ground_truth_df):
        """Fit the cosine similarity indexer to ground truth names

        For example this creates TFIDF weights and matrix of the ground truth names.

        Args:
            ground_truth_df: ground truth dataframe with preprocessed names.

        Returns:
            fitted SparkCosSimIndexerModel
        """
        logger.info("SparkCosSimIndexer._fit(): indexer_id = %s", self.parameters["indexer_id"])
        logger.info("SparkCosSimIndexer._fit(): stage vectorizer.fit(gt)")
        self.fitted_vectorizer = self.vectorizer.fit(ground_truth_df)
        if self.cossim is not None:
            logger.info("SparkCosSimIndexer._fit(): stage vectorizer.transform(gt)")
            ground_truth_df_vec = self.fitted_vectorizer.transform(ground_truth_df)
            logger.info("SparkCosSimIndexer._fit(): stage cossim.fit(gt_vec)")
            self.fitted_cossim = self.cossim.fit(ground_truth_df_vec)
        else:
            self.fitted_cossim = None
        return SparkCosSimIndexerModel(
            parameters=self.parameters, vectorizer=self.fitted_vectorizer, cossim=self.fitted_cossim
        )


class SparkCosSimIndexerModel(
    Model,
    SparkReadable,
    SparkWriteable,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    BaseIndexer,
):
    """Fitted Cosine similarity indexer to generate candidate name-pairs of possible matches"""

    SERIALIZE_ATTRIBUTES = ("parameters", "vectorizer", "cossim")

    def __init__(self, parameters: dict | None = None, vectorizer=None, cossim=None) -> None:
        """Fitted Cosine similarity indexer to generate candidate name-pairs of possible matches

        See SparkCosSimIndexer for unfitted spark class and details on usage.

        Args:
            parameters: dictionary with settings of the cosine similarity indexer.
            vectorizer: fitted pipeline from SparkCosSimIndexer.
            cossim: fitted cosine similarity calculator from SparkCosSimIndexer.
        """
        super().__init__()
        self.parameters = parameters or {}
        self.vectorizer = vectorizer
        self.cossim = cossim

    def _transform(self, names_df):
        """Match processed names in names_df to the previously fitted ground truth.

        Args:
            names_df: Spark dataframe with preprocessed names that should be matched

        Returns:
            Spark dataframe with the candidate matches returned by indexers. Each row contains a single candidate name-pair.
            Columns `gt_uid`, `uid` contains index value from ground truth and X.
            Optionally id column (specified by `self.uid_col`) and carry on columns (specified by `self.carry_on_cols`)
            are copied from gt/X dataframes with the prefixes: `gt_` or `.
            Any additional columns calculated by indexers are also preserved (i.e. score).
        """
        logger.info("SparkCosSimIndexerModel._transform()")
        names_vec = self.vectorizer.transform(names_df)
        if self.cossim is not None:
            res = self.cossim.transform(names_vec)
            if self.parameters.get("keep_all_cols", False):
                return res.join(names_vec.select("uid", "tokens", "ngram_tokens", "tf", "idf", "features"), on="uid")
            return res
        return names_vec

    def calc_score(self, sdf: pyspark.sql.DataFrame, name1_col: str, name2_col: str) -> pyspark.sql.DataFrame:
        res = sdf
        for input_col, output_col in [(name1_col, "feat1"), (name2_col, "feat2")]:
            res = self.vectorizer.transform(res.withColumn("preprocessed", F.col(input_col)))
            res = res.withColumnRenamed("features", output_col)
            res = res.drop("preprocessed", "tokens", "ngram_tokens", "tf", "idf")
        res = res.withColumn("indexer_score", dot_product_udf("feat1", "feat2"))
        return res.drop("feat1", "feat2")

    def increase_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        if self.cossim is not None:
            self.cossim.increase_window_by_one_step()

    def decrease_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        if self.cossim is not None:
            self.cossim.decrease_window_by_one_step()


def add_blocking_col(
    sdf: DataFrame, name_col: str, blocking_col: str | None, blocking_func: Callable | None
) -> DataFrame:
    if blocking_func is not None:
        b_udf = F.udf(blocking_func, StringType())
        sdf = sdf.withColumn(blocking_col, b_udf(name_col))
    return sdf


def match_one(features, ground_truth_features, ground_truth_indices, num_candidates=10, lower_bound=0.5):
    matched_rows = awesome_cossim_topn(as_matrix(features, False), ground_truth_features, num_candidates, lower_bound)
    return get_candidate_list(ground_truth_indices, zip(matched_rows.indices, matched_rows.data))


def get_n_top_matches(gt_features_csr_bc, gt_indices_bc, nm_features_csr, ntop, cos_sim_lower_bound, n_threads=1):
    """Use fast cython implementation
    Get n best candidates for a list of names given their sparse feature vectors
    """
    if gt_features_csr_bc is None:
        return [None] * nm_features_csr.shape[0]
    try:
        # We tried quickly to set use_threads=True, n_jobs=8, but not execution time difference
        results = awesome_cossim_topn(
            nm_features_csr, gt_features_csr_bc, ntop, cos_sim_lower_bound, n_jobs=n_threads, use_threads=n_threads > 1
        )
        candidate = [
            get_candidate_list(gt_indices_bc, zip(row.indices, row.data)) if len(row.data) > 0 else None
            for row in results
        ]
    except BaseException as be:
        raise ValueError("Error from C++ code:" + str(be)) from be

    return candidate


def get_candidate_list(gt_indices, candidate_row_and_score):
    """Return (similarity-score, id) for a tuple (row_index, sim_score)"""
    if candidate_row_and_score:
        return [
            (
                float(sim_score),
                int(
                    gt_indices[row_index]
                ),  # convert matrix position index into ground-truth uid, since it is a np.array convert to int for LongType
            )
            for row_index, sim_score in candidate_row_and_score
        ]
    return None


def split_every(nr_row_per_slice, iterable, blocking_col=None):
    if blocking_col is None:
        i = iter(iterable)
        piece = list(islice(i, nr_row_per_slice))
        while piece:
            yield None, piece
            piece = list(islice(i, nr_row_per_slice))
    else:
        data = list(iterable)
        data = groupby(data, [row[blocking_col] for row in data])
        for key, group in data.items():
            for i in range(0, len(group), nr_row_per_slice):
                yield key, group[i : i + nr_row_per_slice]


def get_n_top_matches_for_all(
    iterator,
    gt_features_csr_bc,
    gt_indices_bc,
    ntop,
    cos_sim_lower_bound,
    uid_col,
    feature_col="features",
    dense=False,
    blocking_col=None,
    n_threads=1,
):
    """Match at partition of names to the ground truth
    (The ground truth has been already transposed)
    """
    # Iterate over all the rows of the partition
    try:
        indices, features, groups = zip(
            *((r[uid_col], r[feature_col], None if blocking_col is None else r[blocking_col]) for r in iterator)
        )
    except ValueError as ve:
        logger.warning(f"Empty partition iterator, exception: {ve}")
        return

    # It is important to do the following two conversions before blocking, because we need it for groupby
    indices = np.array(
        indices, dtype=int
    )  # List can't be "sliced/masked", this is why we need to convert it to numpy array
    nm_features_csr = stack_features(
        [as_matrix(x, dense) for x in features], dense
    )  # This type is important to have a fast groupby

    # If we have blocking, split the data for each available key in this partition
    if blocking_col:
        nm_features_csr_grouped = groupby(nm_features_csr, groups)
        indices_grouped = groupby(indices, groups)
    else:
        nm_features_csr_grouped = {None: nm_features_csr}
        indices_grouped = {None: indices}

    # Loop over all blocking keys. If no blocking this loop is executed once.
    for blocking_key in nm_features_csr_grouped:
        nm_features_csr = nm_features_csr_grouped[blocking_key]
        indices = indices_grouped[blocking_key]

        if nm_features_csr.shape[0] > 0:
            try:
                candidate = get_n_top_matches(
                    gt_features_csr_bc.value if blocking_col is None else gt_features_csr_bc.value.get(blocking_key),
                    gt_indices_bc.value if blocking_col is None else gt_indices_bc.value.get(blocking_key),
                    nm_features_csr,
                    ntop,
                    cos_sim_lower_bound,
                    n_threads,
                )
            except BaseException as be:
                error = "Exception: " + str(be) + "\non row: " + str(indices)
                error += "\nlen(nm_features_csr): " + str(nm_features_csr.shape[0])
                with contextlib.suppress(BaseException):
                    error += "\nfeatures: " + str(nm_features_csr)

                raise ValueError(error) from be

            # using .tolist() is important to get a list of int, matching the excepted uid LongType
            yield zip(*([indices.tolist(), candidate]))


class SparkCosSimMatcher(
    Estimator, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, CosSimBaseIndexer
):
    """Unfitted Cosine similarity calculator of name-pairs candidates"""

    def __init__(
        self,
        num_candidates,
        cos_sim_lower_bound,
        index_col: str = "entity_id",
        uid_col: str = "uid",
        name_col: str = "preprocessed",
        streaming: bool = False,
        blocking_func=None,
        indexer_id=None,
        n_threads=1,
    ) -> None:
        """Unfitted Cosine similarity calculator of name-pairs candidates

        When fitted it returns SparkCosSimMatcherModel.

        SparkCosSimMatcher is used by SparkCosSimIndexer, the last step coming after pipeline.

        Args:
            num_candidates: maximum number of candidates per name-to-match. default is 2.
            cos_sim_lower_bound: lower bound on cosine similarity values of name-pairs. default is 0.5.
            index_col: id column
            uid_col: uid column
            name_col: (preprocessed) names column
            streaming: use spark streaming, default is False. (So use batching.)
            blocking_func: blocking function for matching of names (e.g. block on first character). default is None.
            indexer_id: optional index, used for bookkeeping in case of multiple spark indexers. default is None.
            n_threads: number of threads for spark worker parallelization of matrix multiplication. default is 1.
        """
        super().__init__()
        CosSimBaseIndexer.__init__(self, num_candidates=num_candidates)
        self.index_col = index_col
        self.uid_col = uid_col
        self.name_col = name_col
        self.cos_sim_lower_bound = cos_sim_lower_bound
        self.streaming = streaming
        self.cos_sim_matcher_model = None
        self.blocking_func = blocking_func
        self.blocking_col = "block" if blocking_func is not None else None
        self.indexer_id = indexer_id
        self.n_threads = n_threads

    def _fit(self, ground_truth_df):
        """Fit the SparkCosSimMatcher to ground truth names

        In particular, this applied the blocking function.

        Args:
            ground_truth_df: ground truth dataframe with preprocessed names.

        Returns:
            fitted SparkCosSimMatcherModel
        """
        logger.info("SparkCosSimMatcher._fit(): indexer_id = %d. ", self.indexer_id)
        set_spark_job_group(
            "CosSimMatcher._fit()",
            f"num_candidates:{self.num_candidates}, cos_sim_lower_bound:{self.cos_sim_lower_bound}, blocking_func:{self.blocking_func}",
        )
        ground_truth_df = add_blocking_col(ground_truth_df, self.name_col, self.blocking_col, self.blocking_func)

        cos_sim_matcher_model = SparkCosSimMatcherModel(
            ground_truth_df,
            self.num_candidates,
            self.cos_sim_lower_bound,
            self.index_col,
            self.uid_col,
            self.name_col,
            self.getInputCol(),
            self.getOutputCol(),
            self.streaming,
            blocking_func=self.blocking_func,
            blocking_col=self.blocking_col,
            indexer_id=self.indexer_id,
            n_threads=self.n_threads,
        )
        self.cos_sim_matcher_model = cos_sim_matcher_model
        self.ground_truth_df = ground_truth_df

        return cos_sim_matcher_model


class SparkCosSimMatcherModel(
    Model,
    SparkReadable,
    SparkWriteable,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    CosSimBaseIndexer,
):
    """Fitted Cosine similarity calculator of name-pairs candidates"""

    SERIALIZE_ATTRIBUTES = (
        "num_candidates",
        "cos_sim_lower_bound",
        "index_col",
        "uid_col",
        "name_col",
        "_input_col",
        "_output_col",
        "streaming",
        "blocking_func",
        "blocking_col",
        "indexer_id",
        "gt_indices",
        "gt_features",
        "n_threads",
    )

    def __init__(
        self,
        ground_truth_df=None,
        num_candidates: int = 2,
        cos_sim_lower_bound: float = 0.5,
        index_col: str = "entity_id",
        uid_col: str = "uid",
        name_col: str = "preprocessed",
        input_col: str = "features",
        output_col: str = "candidates",
        streaming: bool = False,
        blocking_func=None,
        blocking_col=None,
        indexer_id=None,
        gt_indices=None,
        gt_features=None,
        n_threads=1,
    ) -> None:
        """Unfitted Cosine similarity calculator of name-pairs candidates

        See SparkCosSimMatcher for details on usage.

        Args:
            ground_truth_df: ground truth dataframe with preprocessed names after vectorization.
            num_candidates: maximum number of candidates per name-to-match. default is 2.
            cos_sim_lower_bound: lower bound on cosine similarity values of name-pairs. default is 0.5.
            index_col: id column, default is "entity_id".
            uid_col: uid column, default is "uid".
            name_col: (preprocessed) names column, default is "preprocessed".
            streaming: use spark streaming, default is False. (So use batching.)
            blocking_func: blocking function for matching of names (e.g. block on first character). default is None.
            indexer_id: optional index, used for bookkeeping in case of multiple spark indexers. default is None.
            input_col: spark input column.
            output_col: spark output column.
            streaming: use spark streaming, default is False. (So use batching.)
            blocking_func: blocking function for matching of names (e.g. block on first character). default is None.
            blocking_col: column indicating blocked name-pairs. default is None.
            indexer_id: optional index, used for bookkeeping in case of multiple spark indexers. default is None.
            gt_indices: alternative to ground_truth_df, combined with gt_features. default is None.
            gt_features: alternative to ground_truth_df, combined with gt_indices. default is None.
            n_threads: number of threads for spark worker parallelization of matrix multiplication. default is 1.
        """
        super().__init__()
        CosSimBaseIndexer.__init__(self, num_candidates=num_candidates)
        self.streaming = streaming
        self.spark = SparkSession.builder.getOrCreate()
        self.index_col, self.uid_col, self.name_col = index_col, uid_col, name_col
        self.cos_sim_lower_bound = cos_sim_lower_bound
        self._set(inputCol=input_col)._set(outputCol=output_col)
        self.blocking_func = blocking_func
        self.blocking_col = blocking_col
        self.indexer_id = indexer_id
        self.n_threads = n_threads

        if ground_truth_df is None and (gt_indices is None or gt_features is None):
            msg = "ground_truth_df not filled, and neither are gt_indices and gt_features."
            raise ValueError(msg)

        self.gt_indices = gt_indices
        self.gt_features = gt_features

        if ground_truth_df is not None:
            # this (re)sets gt_indices and gt_features
            self.gt_indices, self.gt_features = self._process_ground_truth(ground_truth_df)

        # broadcast gt_indices, gt_features to worker nodes
        self._broadcast_ground_truth()

    def _transform(self, names_df):
        """Match processed names in names_df to the previously fitted ground truth.

        Args:
            names_df: Spark dataframe with preprocessed names that should be matched

        Returns:
            Spark dataframe with the candidate matches returned by indexers. Each row contains a single candidate name-pair.
            Columns `gt_uid`, `uid` contains index value from ground truth and X.
            Optionally id column (specified by `self.uid_col`) and carry on columns (specified by `self.carry_on_cols`)
            are copied from gt/X dataframes with the prefixes: `gt_` or `.
            Any additional columns calculated by indexers are also preserved (i.e. score).
        """
        logger.info("SparkCosSimMatcherModel._transform(): indexer_id = %d", self.indexer_id)
        set_spark_job_group(
            "CosSimMatcherModel._transform()",
            f"indexer_id:{self.indexer_id}, num_candidates:{self.num_candidates}, cos_sim_lower_bound:{self.cos_sim_lower_bound}, blocking_func:{self.blocking_func}",
        )
        candidate_list_schema = ArrayType(
            StructType(
                [
                    StructField(name="indexer_score", dataType=T.FloatType(), nullable=True),
                    StructField(name="gt_uid", dataType=T.LongType(), nullable=True),
                ]
            )
        )

        # We don't sort (sorting gives a minimum number of different blocks per partitions, but it is shuffling data and introduce skewness)
        names_df = add_blocking_col(names_df, self.name_col, self.blocking_col, self.blocking_func)

        # 'Save' the vectorized name for missing score computation in the multiple indexers case.
        self.names_df = names_df

        if self.streaming:
            match_name = curry(
                match_one,
                self.gt_features_csr_bc.value,
                self.gt_indices_bc.value,
                self.num_candidates,
                self.cos_sim_lower_bound,
            )
            match_name_udf = F.udf(match_name, candidate_list_schema)
            return names_df.withColumn("candidates", match_name_udf(names_df.features))

        # We use mapPartitions(). FYI we can't use Spark Pandas UDF because the type of the feature vector column is not supported and we get error:
        #   java.lang.UnsupportedOperationException: Unsupported data type: struct<type:tinyint,size:int,indices:array<int>,values:array<double>>
        #   at org.apache.spark.sql.util.ArrowUtils$.toArrowType(ArrowUtils.scala:57)

        match_partition = curry(
            get_n_top_matches_for_all,
            self.gt_features_csr_bc,
            self.gt_indices_bc,
            self.num_candidates,
            self.cos_sim_lower_bound,
            self.uid_col,
            self.getInputCol(),
            False,
            self.blocking_col,
            self.n_threads,
        )
        # Match names
        matched_rdd = (
            names_df
            # We select the minimum number of columns to optimize memory:
            .select([self.uid_col, self.getInputCol()] + ([] if self.blocking_col is None else [self.blocking_col]))
            .rdd.mapPartitions(match_partition)
            .flatMap(lambda x: x)
        )

        # Make output a DataFrame again
        # FYI: we could use instead self.spark.createDataFrame(matched_rdd) but then we need to yield Row() with the column names
        output_schema = StructType(
            [StructField(self.uid_col, LongType()), StructField("candidates", candidate_list_schema)]
        )
        matched_df = matched_rdd.toDF(output_schema)

        # We have here two independent UIDs, one in each sides (ground-truth and names_to_match)
        # All the column available on names_to_match are going to be passed along

        # Explode candidates to have one row per candidate and to be able to join ground-truth columns
        matched_df = explode_candidates(matched_df, with_rank=True)

        return matched_df.select(
            self.uid_col,
            F.col("candidate.indexer_score").alias("indexer_score"),
            F.col("candidate_rank").alias("indexer_rank"),
            F.col("candidate.gt_uid").alias("gt_uid"),
        )

    def _process_ground_truth(self, ground_truth_df):
        """Collect index and features matrices from the ground truth"""
        logger.info("CosSimMatcherModel indexer_id: %d", self.indexer_id)

        gt_indices, gt_features = collect_matrix(
            ground_truth_df, self.uid_col, self.getInputCol(), blocking_col=self.blocking_col
        )

        if self.blocking_col is None:
            logger.debug(f"{self.indexer_id} gt_features.shape_0: %d", gt_features.shape[0])
            logger.debug(f"{self.indexer_id} gt_features.nnz: %s", gt_features.nnz)
            logger.debug(f"{self.indexer_id} gt_indices.len: %d", len(gt_indices))
            gt_features_dtype = gt_features.dtype
            gt_indices_dtype = gt_indices.dtype
        else:
            gt_features_dtype = f"dict of {next(iter(gt_features.values())).dtype}"
            gt_indices_dtype = f"dict of {next(iter(gt_indices.values())).dtype}"

        logger.debug(
            f"{self.indexer_id} getsizeof(gt_features)={getsizeof(gt_features):,.0f} bytes, dtype: {gt_features_dtype}"
        )
        logger.debug(
            f"{self.indexer_id} getsizeof(gt_indices)={getsizeof(gt_indices):,.0f} bytes, dtype: {gt_indices_dtype}"
        )
        return gt_indices, gt_features

    def _broadcast_ground_truth(self):
        """Distribute the ground truth to each worker"""
        self.gt_features_csr_bc = self.spark.sparkContext.broadcast(self.gt_features)  # Already transposed here
        self.gt_indices_bc = self.spark.sparkContext.broadcast(self.gt_indices)

    def _unpersist(self):
        """If you want to run multiple experiments with multiple indexer,
        then you will have multiple broadcast object that might use too much memory.
        We tried to use unpersist() but it didn't solve the memory issue.
        Conclusion: Don't use unpersist, just restart a new Spark Session.
        """
        logger.info("CosSimMatcherModel._unpersist()")
        self.gt_features_csr_bc.unpersist(blocking=True)
        self.gt_ids_and_names_bc.unpersist(blocking=True)

    @property
    def _input_col(self):
        return self.getInputCol()

    @property
    def _output_col(self):
        return self.getOutputCol()
