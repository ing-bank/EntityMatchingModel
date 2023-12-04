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

import numpy as np
import pyspark.sql.functions as sf
from pyspark.ml import Estimator, Model
from pyspark.ml.feature import IDF
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from emm.helper.spark_custom_reader_writer import SparkReadable, SparkWriteable


class SparkNormalizedTfidfVectorizer(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    """Unfitted implementation of Spark TFIDF vectorizer"""

    def __init__(self, count_col, token_col, output_col, binary_countvectorizer) -> None:
        """Unfitted implementation of Spark TFIDF vectorizer

        Based on Spark IDF ML model.
        Tailored to give same results as (pandas) CustomizedTfidfVectorizer.

        SparkNormalizedTfidf is a step in the pipeline used in SparkCosSimIndexer.

        Args:
            count_col: count column to use (e.g. "tf")
            token_col: token column to use (e.g. "ngram_tokens")
            output_col: output column (eg. "features")
            binary_countvectorizer: use binary countvectorizer flag.
        """
        super().__init__()
        self.count_col = count_col
        self.token_col = token_col
        self.output_col = output_col
        self.spark_idf = IDF(inputCol=count_col, outputCol="idf")
        self.spark_idf_model = None
        self.max_idf = None
        self.binary_countvectorizer = binary_countvectorizer

    def _fit(self, dataset):
        """Fit the vectorizer output dataset to calculate TFIDF weights and matrix

        Args:
            dataset: vectorizer output dataset

        Returns:
            fitted SparkNormalizedTfidfModel
        """
        self.spark_idf_model = self.spark_idf.fit(dataset)
        self.max_idf = max(self.spark_idf_model.idf)
        return SparkNormalizedTfidfModel(
            self.spark_idf_model,
            self.max_idf,
            self.count_col,
            self.token_col,
            self.output_col,
            self.binary_countvectorizer,
        )


class SparkNormalizedTfidfModel(Model, SparkReadable, SparkWriteable, DefaultParamsReadable, DefaultParamsWritable):
    """Fitted implementation of Spark TFIDF vectorizer"""

    SERIALIZE_ATTRIBUTES = (
        "max_idf",
        "count_col",
        "token_col",
        "output_col",
        "binary_countvectorizer",
        "spark_idf_model",
    )

    def __init__(
        self,
        spark_idf_model=None,
        max_idf=1.0,
        count_col: str = "tf",
        token_col: str = "ngram_tokens",
        output_col: str = "features",
        binary_countvectorizer=False,
    ) -> None:
        """Fitted implementation of Spark TFIDF vectorizer

        Based on Spark IDF model. For more details see SparkNormalizedTfidf.

        Args:
            spark_idf_model: spark idf model.
            max_idf: default is 1.
            count_col: count column to use (e.g. "tf")
            token_col: token column to use (e.g. "ngram_tokens")
            output_col: output column (eg. "features")
            binary_countvectorizer: use binary countvectorizer flag. default is False.
        """
        super().__init__()
        self.spark_idf_model = spark_idf_model
        self.max_idf = max_idf
        self.count_col = count_col
        self.token_col = token_col
        self.output_col = output_col
        self.binary_countvectorizer = binary_countvectorizer
        self._initialize()

    def _initialize(self):
        self.idf_normalizer_udf = sf.udf(
            idf_normalizer_getter(
                binary_countvectorizer=self.binary_countvectorizer, max_idf_square=pow(self.max_idf, 2)
            ),
            VectorUDT(),
        )

    def _transform(self, dataset):
        """Transform vectorized input names to tfidf vectors

        Args:
            dataset: dataset with vectorized names.

        Returns:
            same dataset now including tfidf features column.
        """
        dataset = self.spark_idf_model.transform(dataset)
        return dataset.withColumn(
            self.output_col, self.idf_normalizer_udf(sf.col(self.count_col), sf.col(self.token_col), sf.col("idf"))
        )


def idf_normalizer_getter(binary_countvectorizer, max_idf_square):
    """Input:
        count_vec: output of CountVectorizer
        token_vec: output of RegexTokenizer or Ngram
        idf_vec: created tfidf vector
        max_idf: square of maximum idf value (idf value of the rarest word)

    Return:
        normalized tfidf vector
    """

    def idf_normalizer(count_vec, token_vec, idf_vec):
        # if there is no vocabulary word, return the empty vector
        if len(idf_vec.values) == 0:
            return idf_vec

        len_token_vec = len(set(token_vec)) if binary_countvectorizer else len(token_vec)

        # number of out-of-vocabulary words in the name
        len_words_out_voc = len_token_vec - sum(count_vec.values)

        # norm2
        square_total = np.sum(np.power(idf_vec.values, 2))
        if square_total > 0:
            normalizer = 1.0 / np.sqrt(np.sum(np.power(idf_vec.values, 2)) + len_words_out_voc * max_idf_square)
            normalized_values = normalizer * idf_vec.values
        else:
            normalized_values = idf_vec.values
        return SparseVector(idf_vec.size, dict(zip(idf_vec.indices, normalized_values)))

    return idf_normalizer
