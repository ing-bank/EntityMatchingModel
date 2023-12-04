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

from functools import partial
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.preprocessing.base_name_preprocessor import AbstractPreprocessor, create_func_dict


class PandasPreprocessor(TransformerMixin, AbstractPreprocessor):
    """Pandas implementation of Name Preprocessor"""

    def __init__(
        self,
        preprocess_pipeline: Any = "preprocess_merge_abbr",
        input_col: str = "name",
        output_col: str = "preprocessed",
        spark_session: Any | None = None,
    ) -> None:
        """Pandas implementation of Name Preprocessor

        PandasPreprocessor is the first step of the PandasEntityMatching pipeline.
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
            >>> p = PandasPreprocessor(preprocess_pipeline="preprocess_merge_abbr", input_col="name")
            >>> clean_names_df = p.transform(names_df)

        """
        super().__init__()
        AbstractPreprocessor.__init__(self, preprocess_pipeline, input_col, output_col, spark_session)

    def create_func_dict(self) -> Mapping[str, Callable]:
        return create_func_dict(use_spark=False)

    def fit(self, *args: Any, **kwargs: Any) -> TransformerMixin:
        """Dummy function, this class does not require fitting

        Args:
            args: ignored.
            kwargs: ignored.

        Returns:
            self
        """
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **extra_params: Any) -> pd.DataFrame:
        """Perform preprocessing transform() of input names

        Perform string cleaning, to-lower, remove punctuation and white spaces, convert legal entity forms to
        standard abbreviations.

        Note this class does not require fitting, so not done.

        Args:
            X: dataframe containing input names.
            y: ignored.
            extra_params: extra parameters are passed on to transform() function.

        Returns:
            dataframe with preprocessed names
        """
        return self.transform(X, **extra_params)

    def _spark_apply_steps(
        self, series: pd.Series, preprocess_list: list[Any], func_dict: Mapping[str, Any], chunk_size: int = 10**4
    ) -> pd.Series:
        # Remark: 'chunk_size' is not the same as 'partition_size'
        # because here we just do name preprocessing and that can be done with much larger partitions
        # than 'partition_size' that is designed to handle the fact that cosine similarity creates 10 times more data after the candidate generation

        with Timer("PandasPreprocessor._spark_apply_steps") as timer:
            X_chunks = np.array_split(series, (len(series) + chunk_size - 1) // chunk_size)
            sc = self.spark_session.sparkContext
            rdd = sc.parallelize(X_chunks, len(X_chunks))

            def calc(chunk, funcs):
                for func in funcs:
                    chunk = func(chunk)
                return chunk.index.values, chunk.values

            functions = [func_dict[x] if isinstance(x, str) else lambda series: series.map(x) for x in preprocess_list]
            cs_rdd = rdd.map(partial(calc, functions=functions))
            cs_list = cs_rdd.collect()
            res = pd.concat((pd.Series(x[1], index=x[0]) for x in cs_list), axis=0, sort=False)

            timer.log_param("n", len(series))
        return res

    @staticmethod
    def _local_apply_steps(series: pd.Series, preprocess_list: list[Any], func_dict: Mapping[str, Any]) -> pd.Series:
        with Timer("PandasPreprocessor._local_apply_steps") as timer:
            for preprocess_def in preprocess_list:
                timer.label(preprocess_def)
                func = (
                    func_dict[preprocess_def]
                    if isinstance(preprocess_def, str)
                    else lambda series: series.map(preprocess_def)
                )
                series = func(series)

            timer.log_param("n", len(series))

        return series

    def transform(self, dataset: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply preprocessing functions to input names in dataframe

        Perform string cleaning, to-lower, remove punctuation and white spaces, convert legal entity forms to
        standard abbreviations.

        Args:
            dataset: dataframe containing input names.
            y: ignored.

        Returns:
            dataframe with preprocessed names
        """
        with Timer("PandasPreprocessor.transform") as timer:
            timer.log_params({"X.shape": dataset.shape})

            if not (isinstance(dataset, (pd.DataFrame, pd.Series))):
                logger.info("converting to pandas dataframe")
                dataset = dataset.toPandas()
            elif isinstance(dataset, pd.Series):
                dataset = pd.DataFrame(dataset).copy()
            else:
                dataset = dataset.copy()
            series = dataset[self.input_col]
            # in verbose mode we store value of names before/after preprocessing
            # save original names or not used in non-verbose mode, but still required to avoid warning
            series = series.fillna("")

            func_dict = self.create_func_dict()
            if self.spark_session is not None and len(dataset) > 2 * 10**5:
                series = self._spark_apply_steps(series, self.preprocess_list, func_dict, chunk_size=10**4)
            else:
                series = self._local_apply_steps(series, self.preprocess_list, func_dict)

            # pyarrow string datatype is much more memory efficient. important for large lists of names (1M+).
            dataset[self.output_col] = series.astype("string[pyarrow]")
            timer.log_param("n", len(dataset))
        return dataset
