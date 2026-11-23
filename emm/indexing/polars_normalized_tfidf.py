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

"""Customized TF-IDF vectorization for Polars inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import scipy
import scipy.sparse as sp
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from emm.loggers import Timer


def _as_text_list(X: pl.Series | pl.DataFrame) -> list[str]:
    if isinstance(X, pl.DataFrame):
        if X.width != 1:
            msg = "Polars DataFrame input must contain exactly one column"
            raise ValueError(msg)
        X = X.to_series()
    if not isinstance(X, pl.Series):
        msg = f"expected a Polars Series or DataFrame, got {type(X).__name__}"
        raise TypeError(msg)
    return ["" if value is None else str(value) for value in X.to_list()]


class PolarsNormalizedTfidfVectorizer(TfidfVectorizer):
    """TF-IDF vectorizer with EMM's OOV-aware normalization."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("dtype", np.float32)
        kwargs.update({"norm": None, "smooth_idf": True, "lowercase": True})
        if kwargs.get("analyzer") in {"word", None}:
            kwargs["token_pattern"] = r"\w+"
        super().__init__(**kwargs)

    def fit(self, X: pl.Series | pl.DataFrame, y: Any = None) -> PolarsNormalizedTfidfVectorizer:
        documents = _as_text_list(X)
        with Timer("PolarsNormalizedTfidfVectorizer.fit") as timer:
            timer.label("super fit")
            super().fit(documents)
            self.idf_ = self.idf_.astype(self.dtype)

            timer.label("normalize")
            n_features = self.idf_.shape[0]
            idf_diag = scipy.sparse.diags(
                self.idf_, offsets=0, shape=(n_features, n_features), format="csr", dtype=self.dtype
            )
            idf_diag = idf_diag - scipy.sparse.diags(np.ones(n_features), shape=idf_diag.shape, dtype=self.dtype)
            self.max_idf_square = idf_diag.max() ** 2

            if hasattr(self._tfidf, "_idf_diag"):
                self._tfidf._idf_diag = idf_diag
            else:
                self.idf_ = self.idf_ - np.ones(n_features, dtype=self.dtype)

            timer.log_params({"n": len(documents), "n_features": n_features})
        return self

    def transform(self, X: pl.Series | pl.DataFrame) -> scipy.sparse.csr_matrix:
        documents = _as_text_list(X)
        if not documents:
            return scipy.sparse.csr_matrix((0, len(self.vocabulary_)), dtype=self.dtype)
        with Timer("PolarsNormalizedTfidfVectorizer.transform") as timer:
            timer.label("number_of_all_tokens")
            analyzer = self.build_analyzer()

            def calc_number_of_tokens(document: str, binary: bool) -> int:
                tokens = analyzer(document)
                return len(set(tokens)) if binary else len(tokens)

            number_of_all_tokens = np.fromiter(
                (calc_number_of_tokens(document, self.binary) for document in documents), dtype=np.int64
            )

            timer.label("counts")
            counts = CountVectorizer.transform(self, documents)
            number_of_matched_tokens = counts.sum(axis=1).A1
            oov = number_of_all_tokens - number_of_matched_tokens
            if oov.size:
                assert oov.min() >= 0

            timer.label("transform")
            res_before_norm = self._tfidf.transform(counts, copy=False)

            timer.label("normalization")
            norm_sum_part = res_before_norm.power(2).sum(axis=1).A1
            norm_oov_part = oov * self.max_idf_square
            norm = (1.0 / np.clip(np.sqrt(norm_sum_part + norm_oov_part), 1e-9, None)).astype(self.dtype)
            res = res_before_norm.T.multiply(norm).T.tocsr()
            assert res.dtype == self.dtype
            timer.log_param("n", len(documents))
        return res

    def fit_transform(self, raw_documents: pl.Series | pl.DataFrame, y: Any | None = None) -> scipy.sparse.csr_matrix:
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def transform_parallel(self, X: pl.Series | pl.DataFrame, n_jobs: int = -1) -> scipy.sparse.csr_matrix:
        documents = _as_text_list(X)
        n_jobs = effective_n_jobs(n_jobs)
        if n_jobs == 1 or len(documents) == 0:
            return self.transform(pl.Series("documents", documents))

        chunk_size = int(np.ceil(len(documents) / n_jobs))
        chunks = [documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)]
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self.transform)(pl.Series("documents", chunk)) for chunk in chunks
        )
        return sp.vstack(results, format="csr")
