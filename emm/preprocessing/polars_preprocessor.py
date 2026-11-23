# Copyright (c) 2023 ING Analytics Wholesale Banking
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Any, Callable, Mapping

import polars as pl
from sklearn.base import TransformerMixin

from emm.loggers import Timer
from emm.preprocessing.base_name_preprocessor import AbstractPreprocessor, create_func_dict


class PolarsPreprocessor(TransformerMixin, AbstractPreprocessor):
    """Polars implementation of Name Preprocessor."""

    def __init__(
        self,
        preprocess_pipeline: Any = "preprocess_merge_abbr",
        input_col: str = "name",
        output_col: str = "preprocessed",
        spark_session: Any | None = None,
    ) -> None:
        """Create a Polars name preprocessor.

        Args:
            preprocess_pipeline: Predefined pipeline name or a list of preprocessing functions.
            input_col: Column containing the input names.
            output_col: Column to contain the preprocessed names.
            spark_session: Accepted for compatibility with the other preprocessors; unused here.
        """
        super().__init__()
        AbstractPreprocessor.__init__(self, preprocess_pipeline, input_col, output_col, spark_session)

    def create_func_dict(self) -> Mapping[str, Callable]:
        return create_func_dict(use_spark=False, use_polars=True)

    def fit(self, *args: Any, **kwargs: Any) -> TransformerMixin:
        """Return self; preprocessing does not require fitting."""
        return self

    def fit_transform(self, X: pl.DataFrame | pl.Series, y: Any = None, **extra_params: Any) -> pl.DataFrame:
        """Apply preprocessing to the input names."""
        return self.transform(X, **extra_params)

    def transform(self, dataset: pl.DataFrame | pl.Series, y: Any = None) -> pl.DataFrame:
        """Apply preprocessing functions to the input name column."""
        with Timer("PolarsPreprocessor.transform") as timer:
            if isinstance(dataset, pl.Series):
                dataset = dataset.to_frame()
            elif isinstance(dataset, pl.DataFrame):
                dataset = dataset.clone()
            else:
                msg = f"expected a Polars DataFrame or Series, got {type(dataset).__name__}"
                raise TypeError(msg)

            timer.log_param("X.shape", dataset.shape)
            series = dataset[self.input_col].fill_null("")
            func_dict = self.create_func_dict()

            for preprocess_def in self.preprocess_list:
                timer.label(preprocess_def)
                if isinstance(preprocess_def, str):
                    func = func_dict[preprocess_def]
                else:
                    func = lambda values: values.map_elements(
                        preprocess_def, return_dtype=pl.String, skip_nulls=False
                    )
                series = func(series)

            dataset = dataset.with_columns(series.alias(self.output_col))
            timer.log_param("n", dataset.height)
        return dataset