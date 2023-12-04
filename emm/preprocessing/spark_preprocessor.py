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

from typing import Any

import pyspark.sql.functions as sf
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import StringType

from emm.helper.spark_custom_reader_writer import SparkReadable, SparkWriteable
from emm.loggers.logger import logger
from emm.preprocessing.base_name_preprocessor import AbstractPreprocessor
from emm.preprocessing.functions import replace_none


class SparkPreprocessor(
    Transformer,
    HasInputCol,
    HasOutputCol,
    AbstractPreprocessor,
    SparkReadable,
    SparkWriteable,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Spark implementation of Name Preprocessor"""

    SERIALIZE_ATTRIBUTES = ("preprocess_pipeline", "_input_col", "_output_col")
    SPARK_SESSION_KW = "spark_session"

    def __init__(
        self,
        preprocess_pipeline: Any = "preprocess_merge_abbr",
        input_col: str = "name",
        output_col: str = "preprocessed",
        spark_session: Any | None = None,
    ) -> None:
        """Spark implementation of Name Preprocessor

        SparkPreprocessor is the first step of the SparkEntityMatching pipeline.
        It performs cleaning and standardization of input names and their legal entity forms. Perform string cleaning,
        to-lower, remove punctuation and white spaces, convert legal entity forms to standard abbreviations.

        Four predefined options for "preprocess_pipeline" are available:

        - "preprocess_name": normal cleaning, remove punctuation, handle unicode, lower and trim
        - "preprocess_with_punctuation": normal cleaning. punctuation will be kept, insert spaces around it.
        - "preprocess_merge_abbr": normal cleaning. merge all abbreviations. (default.)
        - "preprocess_merge_legal_abbr": normal cleaning. merge only legal form abbreviation.

        See `emm.preprocessing.base_name_preprocessor.DEFINED_PIPELINE_DICT` for details of all cleaning functions.

        Args:
            preprocess_pipeline: default is "preprocess_merge_abbr". Perform string cleaning, to-lower, remove
                                    punctuation and white spaces, convert legal entity forms to standard abbreviations.
            input_col: column name of input names. optional. default is "name".
            output_col: column name of output names. optional. default is "preprocessed".
            spark_session: spark session for processing. default processing is local. optional.


        Examples:
            >>> p = SparkPreprocessor(preprocess_pipeline="preprocess_merge_abbr", input_col="name")
            >>> clean_names_sdf = p.transform(names_sdf)

        """
        super().__init__()
        self._set(inputCol=input_col)
        self._set(outputCol=output_col)
        AbstractPreprocessor.__init__(self, preprocess_pipeline, input_col, output_col, spark_session)

    def _transform(self, dataset):
        """Apply preprocessing functions to input names in dataframe

        Perform string cleaning, to-lower, remove punctuation and white spaces, convert legal entity forms to
        standard abbreviations.

        Args:
            dataset: dataframe containing input names.

        Returns:
            dataframe with preprocessed names
        """
        logger.info("SparkPreprocessor._transform()")

        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        replace_none_udf = sf.udf(replace_none, StringType())
        dataset = dataset.withColumn(output_col, replace_none_udf(input_col))
        func_dict = self.create_func_dict()
        for preprocess_def in self.preprocess_list:
            func = (
                func_dict[preprocess_def] if isinstance(preprocess_def, str) else sf.udf(preprocess_def, StringType())
            )
            dataset = dataset.withColumn(output_col, func(output_col))
        return dataset

    @property
    def _input_col(self) -> str:
        """Alias for getInputCol method"""
        return self.getInputCol()

    @property
    def _output_col(self) -> str:
        """Alias for getOutputCol method"""
        return self.getOutputCol()

    @property
    def preprocess_pipeline(self):
        """Alias for preprocess_list"""
        return self.preprocess_list
