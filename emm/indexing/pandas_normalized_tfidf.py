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

"""Customized TFIDF vectorization."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from emm.loggers import Timer


class PandasNormalizedTfidfVectorizer(TfidfVectorizer):
    """Implementation of customized TFIDF vectorizer"""

    dtype = np.float32

    def __init__(self, **kwargs: Any) -> None:
        """Implementation of customized TFIDF vectorizer

        Custom implementation of sklearn's TfidfVectorizer. (Please see there for details.)
        Written to give same results as SparkNormalizedTfidf vectorizer.

        * idf_diag is using formula `np.log(n_samples / df)` instead of default `np.log(n_samples / df) + 1`
        * custom normalization function that takes into account out-of-vocabulary words

        CustomizedTfidfVectorizer is used as step in pipeline in PandasCosSimIndexer.

        Args:
            kwargs: kew-word arguments are same as TfidfVectorizer.
        """
        kwargs.update({"norm": None, "smooth_idf": True, "lowercase": True})
        if kwargs.get("analyzer") in {"word", None}:
            kwargs["token_pattern"] = r"\w+"
        super().__init__(**kwargs)

    def fit(self, X: pd.Series | pd.DataFrame) -> TfidfVectorizer:
        """Fit the TFIDF vectorizer.

        Args:
            X: dataframe with preprocessed names

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            assert len(X.columns) == 1
            X = X.iloc[:, 0]

        with Timer("CustomizedTfidfVectorizer.fit") as timer:
            timer.label("super fit")
            super().fit(X)

            timer.label("normalize")
            idf_diag = self._tfidf._idf_diag
            idf_diag = idf_diag - scipy.sparse.diags(np.ones(idf_diag.shape[0]), shape=idf_diag.shape, dtype=self.dtype)
            self._tfidf._idf_diag = idf_diag
            assert self._tfidf._idf_diag.dtype == self.dtype
            # this value is used in normalization step for simulating out-of-vocabulary tokens
            self.max_idf_square = idf_diag.max() ** 2

            timer.log_params({"n": len(X), "n_features": idf_diag.shape[0]})

        return self

    def transform(self, X: pd.Series | pd.DataFrame) -> scipy.sparse.csr_matrix:
        """Apply the fitted TFIDF vectorizer

        Args:
            X: dataframe with preprocessed names

        Returns:
            normalized tfidf vectors of names
        """
        if isinstance(X, pd.DataFrame):
            assert len(X.columns) == 1
            X = X.iloc[:, 0]

        with Timer("CustomizedTfidfVectorizer.transform") as timer:
            timer.label("number_of_all_tokens")
            analyzer = self.build_analyzer()

            def calc_number_of_tokens(x, binary: bool):
                if binary:
                    return len(set(analyzer(x)))
                return len(analyzer(x))

            number_of_all_tokens = X.map(partial(calc_number_of_tokens, binary=self.binary)).values

            # calculate out-of-vocabulary tokens
            timer.label("counts")
            counts = CountVectorizer.transform(self, X)
            number_of_matched_tokens = counts.sum(axis=1).A1
            oov = number_of_all_tokens - number_of_matched_tokens

            assert oov.min() >= 0

            timer.label("transform")
            res_before_norm = self._tfidf.transform(counts, copy=False)

            timer.label("normalization")
            norm_sum_part = res_before_norm.power(2).sum(axis=1).A1
            norm_oov_part = oov * self.max_idf_square
            eps = 1e-9  # to get rid of division by zero errors
            norm = (1.0 / np.clip(np.sqrt(norm_sum_part + norm_oov_part), eps, None)).astype(self.dtype)
            res = res_before_norm.T.multiply(norm).T.tocsr()
            assert res.dtype == self.dtype

            timer.log_param("n", len(X))

        return res

    def fit_transform(self, raw_documents: pd.Series | pd.DataFrame, y: Any | None = None) -> scipy.sparse.csr_matrix:
        """Implementation of fit followed by transform

        Args:
            raw_documents: dataframe with preprocessed input names.
            y: ignored.

        Returns:
            normalized tfidf vectors of names.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def transform_parallel(self, X: pd.Series | pd.DataFrame, n_jobs: int = -1) -> scipy.sparse.csr_matrix:
        """Parallel apply the fitted TFIDF vectorizer

        Inspired by: https://github.com/scikit-learn/scikit-learn/issues/7635#issuecomment-254407618

        Args:
            X: dataframe with preprocessed names
            n_jobs: desired number of parallel jobs. default is all available cores.

        Returns:
            normalized tfidf vectors of names
        """
        if effective_n_jobs(n_jobs) == 1:
            return self.transform(X=X)

        transform_splits = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self.transform)(X_split)
            for X_split in np.array_split(X, effective_n_jobs(n_jobs))
            if len(X_split) > 0
        )
        return sp.vstack(transform_splits)
