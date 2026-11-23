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

import multiprocessing
from typing import Any, Callable, Literal

import numpy as np
import polars as pl
import scipy
import scipy.sparse
from sklearn.base import TransformerMixin
from sparse_dot_topn import sp_matmul_topn

from emm.helper.blocking_functions import _parse_blocking_func
from emm.indexing.base_indexer import CosSimBaseIndexer
from emm.indexing.polars_normalized_tfidf import PolarsNormalizedTfidfVectorizer
from emm.loggers import Timer
from emm.loggers.logger import logger


def _column_values(X: pl.DataFrame, column: str) -> list[str]:
    return ["" if value is None else str(value) for value in X[column].to_list()]


class PolarsCosSimIndexer(TransformerMixin, CosSimBaseIndexer):
    """Cosine similarity indexer that returns Polars candidate pairs."""

    def __init__(
        self,
        input_col: str = "preprocessed",
        tokenizer: Literal["words", "characters"] = "words",
        ngram: int = 1,
        binary_countvectorizer: bool = False,
        num_candidates: int = 5,
        cos_sim_lower_bound: float = 0.5,
        partition_size: int = 5000,
        max_features: int | None = None,
        n_jobs: int = 1,
        spark_session: Any | None = None,
        blocking_func: Callable[[str], str] | str | None = None,
        dtype: type[float] = np.float32,
        indexer_id: int | None = None,
    ) -> None:
        CosSimBaseIndexer.__init__(self, num_candidates=num_candidates)
        if tokenizer not in {"words", "characters"}:
            msg = f"unsupported tokenizer: {tokenizer}"
            raise ValueError(msg)
        if partition_size <= 0:
            msg = "partition_size should be a positive integer"
            raise ValueError(msg)
        self.input_col = input_col
        self.ngram = ngram
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.cos_sim_lower_bound = cos_sim_lower_bound
        self.partition_size = partition_size
        self.blocking_func = _parse_blocking_func(blocking_func)
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.spark_session = spark_session
        self.tfidf = PolarsNormalizedTfidfVectorizer(
            analyzer={"words": "word", "characters": "char"}[tokenizer],
            binary=binary_countvectorizer,
            ngram_range=(ngram, ngram),
            max_features=max_features,
            dtype=dtype,
        )
        self.gt_uid_values: np.ndarray | dict[Any, np.ndarray] | None = None
        self.gt_enc_t: scipy.sparse.csr_matrix | dict[Any, scipy.sparse.csr_matrix] | None = None

    def fit(self, X: pl.DataFrame, y: Any = None) -> PolarsCosSimIndexer:
        with Timer("PolarsCosSimIndexer.fit") as timer:
            timer.log_params({"X.shape": X.shape})
            names = pl.Series(self.input_col, _column_values(X, self.input_col))
            timer.label("fit")
            self.tfidf.fit(names)

            timer.label("vectorize")
            gt_enc = self.tfidf.transform_parallel(names, n_jobs=self.n_jobs)
            self.gt_uid_values = np.arange(X.height, dtype=np.int64)

            if self.blocking_func is not None:
                gt_blocking = [self.blocking_func(name) for name in names.to_list()]
                keys = dict.fromkeys(gt_blocking)
                self.gt_enc_t = {}
                self.gt_uid_values = {}
                for key in keys:
                    indexes = np.fromiter((i for i, block in enumerate(gt_blocking) if block == key), dtype=np.int64)
                    self.gt_enc_t[key] = gt_enc[indexes].T.tocsr()
                    self.gt_uid_values[key] = indexes
            else:
                self.gt_enc_t = gt_enc.T.tocsr()
            timer.log_param("n", X.height)
        return self

    def transform(self, X: pl.DataFrame, multiple_indexers: bool | None = None) -> pl.DataFrame:
        if self.gt_enc_t is None or self.gt_uid_values is None:
            msg = "model is not fitted yet"
            raise ValueError(msg)

        with Timer("PolarsCosSimIndexer.transform") as timer:
            timer.log_params({"n_jobs": self.n_jobs, "blocking": self.blocking_func is not None})
            timer.label("tfidf")
            names = pl.Series(self.input_col, _column_values(X, self.input_col))
            X_enc = self.tfidf.transform_parallel(names, n_jobs=self.n_jobs)

            if self.blocking_func is not None:
                X_blocking = [self.blocking_func(name) for name in names.to_list()]
                keys = dict.fromkeys(X_blocking)
                query_parts: dict[Any, np.ndarray] = {
                    key: np.fromiter((i for i, block in enumerate(X_blocking) if block == key), dtype=np.int64)
                    for key in keys
                }
                gt_enc_t = self.gt_enc_t
                gt_uid_values = self.gt_uid_values
            else:
                query_parts = {None: np.arange(X.height, dtype=np.int64)}
                gt_enc_t = {None: self.gt_enc_t}
                gt_uid_values = {None: self.gt_uid_values}

            timer.label("cossim")
            uid_results: list[np.ndarray] = []
            gt_uid_results: list[np.ndarray] = []
            score_results: list[np.ndarray] = []
            for key, query_ids in query_parts.items():
                if key not in gt_enc_t:
                    continue
                cossim = self._local_cossim(gt_enc_t[key], X_enc[query_ids])
                cossim = cossim.tocoo()
                assert cossim.dtype == self.dtype
                uid_results.append(query_ids[cossim.row])
                gt_uid_results.append(gt_uid_values[key][cossim.col])
                score_results.append(cossim.data)

            timer.label("candidates")
            if uid_results:
                uids = np.concatenate(uid_results)
                gt_uids = np.concatenate(gt_uid_results)
                scores = np.concatenate(score_results).astype(self.dtype, copy=False)
                order = np.lexsort((-scores, -uids))
                uids, gt_uids, scores = uids[order], gt_uids[order], scores[order]
                ranks = np.empty(len(uids), dtype=np.int64)
                rank_by_uid: dict[int, int] = {}
                for i, uid in enumerate(uids):
                    rank_by_uid[uid] = rank_by_uid.get(int(uid), 0) + 1
                    ranks[i] = rank_by_uid[int(uid)]
            else:
                uids = gt_uids = ranks = np.array([], dtype=np.int64)
                scores = np.array([], dtype=self.dtype)

            result = pl.DataFrame(
                {
                    "uid": pl.Series(uids, dtype=pl.Int64),
                    "gt_uid": pl.Series(gt_uids, dtype=pl.Int64),
                    "score": pl.Series(scores, dtype=pl.Float32 if self.dtype == np.float32 else pl.Float64),
                    "rank": pl.Series(ranks, dtype=pl.Int64),
                }
            )
            timer.log_param("n", X.height)
        return result

    def _local_cossim(
        self, gt_enc_t: scipy.sparse.csr_matrix, X_enc: scipy.sparse.csr_matrix
    ) -> scipy.sparse.csr_matrix:
        assert gt_enc_t.dtype == self.dtype
        assert X_enc.dtype == self.dtype
        logger.debug("calculating cossim gt_enc_t=%r X_enc=%r", gt_enc_t, X_enc)
        chunks = [X_enc[i : i + self.partition_size] for i in range(0, X_enc.shape[0], self.partition_size)]
        if not chunks:
            return scipy.sparse.csr_matrix((0, gt_enc_t.shape[0]), dtype=self.dtype)
        results = [
            sp_matmul_topn(
                chunk,
                gt_enc_t,
                top_n=self.num_candidates,
                threshold=self.cos_sim_lower_bound,
                n_threads=self.n_jobs,
                sort=True,
            )
            for chunk in chunks
        ]
        return scipy.sparse.vstack(results, dtype=self.dtype, format="csr")

    def column_prefix(self) -> str:
        prefix = "w" if self.tokenizer == "words" else "n"
        return f"cossim_{prefix}{self.ngram}"

    def calc_score(self, name1: pl.Series, name2: pl.Series) -> pl.DataFrame:
        if len(name1) != len(name2):
            msg = "name1 and name2 must have the same length"
            raise ValueError(msg)
        names1 = pl.Series(self.input_col, _column_values(name1.to_frame(), name1.name))
        names2 = pl.Series(self.input_col, _column_values(name2.to_frame(), name2.name))
        name1_enc = self.tfidf.transform(names1)
        name2_enc = self.tfidf.transform(names2)
        scores = np.asarray(name1_enc.multiply(name2_enc).sum(axis=1)).ravel()
        return pl.DataFrame({"score": pl.Series(scores, dtype=pl.Float32 if self.dtype == np.float32 else pl.Float64)})
