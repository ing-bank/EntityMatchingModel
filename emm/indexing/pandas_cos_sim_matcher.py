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

import multiprocessing
from functools import partial
from typing import Any, Callable, Generator, Literal

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from sklearn.base import TransformerMixin
from sparse_dot_topn import awesome_cossim_topn

from emm.helper.util import groupby
from emm.indexing.base_indexer import CosSimBaseIndexer
from emm.indexing.pandas_normalized_tfidf import PandasNormalizedTfidfVectorizer
from emm.loggers import Timer
from emm.loggers.logger import logger


class PandasCosSimIndexer(TransformerMixin, CosSimBaseIndexer):
    """Cosine similarity indexer to generate candidate name-pairs of possible matches"""

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
        blocking_func: Callable[[str], str] | None = None,
        dtype: type[float] = np.float32,
        indexer_id: int | None = None,
    ) -> None:
        """Cosine similarity indexer to generate candidate name-pairs of possible matches

        Pipeline of tokenization, ngram creation, vectorization, tfidf, cosine similarity.
        The vectorizer used is a customized version of sklearn's TfidfVectorizer.
        Cosine similarity is calculated in a fast manner using ING's dedicated sparse-dot-topn library.

        The most important settings are: tokenizer, ngram, num_candidates and cos_sim_lower_bound.

        Args:
            input_col: name column in dataframe. default is "preprocessed".
            tokenizer: tokenization used, either "words" or "characters". default is "words".
            ngram: number of n-grams used in name tokenization. default is 1. (for characters we recommend 2.)
            binary_countvectorizer: use binary_countvectorizer in sklearn's TfidfVectorizer. default is False.
            num_candidates: maximum number of candidates per name-to-match. default is 5.
            cos_sim_lower_bound: lower bound on cosine similarity values of name-pairs. default is 0.5.
            partition_size: partition size for chunking of tfidf-matrix of names-to-match for parallelization. default is 5000.
            max_features: maximum number of features used by TfidfVectorizer.
            n_jobs: number of threads for local parallelization of matrix multiplication. default is 1.
            spark_session: for matrix calculation using spark. optional, default is None.
            blocking_func: blocking function for matching of names (e.g. block on first character). default is None.
            dtype: datatype feature used by TfidfVectorizer. default is np.float32.
            indexer_id: ignored. (needed for spark indexers.)

        Examples:
            >>> c = PandasCosSimIndexer(
            >>>     tokenizer="words",
            >>>     ngram=1,
            >>>     num_candidates=10,
            >>>     binary_countvectorizer=True,
            >>>     cos_sim_lower_bound=0.2,
            >>> )
            >>>
            >>> c.fit(ground_truth_df)
            >>> candidates_df = c.transform(names_df)

        """
        CosSimBaseIndexer.__init__(self, num_candidates=num_candidates)
        self.input_col = input_col
        self.ngram = ngram
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.cos_sim_lower_bound = cos_sim_lower_bound
        self.partition_size = partition_size
        self.blocking_func = blocking_func
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.spark_session = spark_session
        # attributes below are set during fit
        self.tfidf = PandasNormalizedTfidfVectorizer(
            analyzer={"words": "word", "characters": "char"}[tokenizer],
            binary=binary_countvectorizer,
            ngram_range=(ngram, ngram),
            max_features=max_features,
            dtype=dtype,
        )
        self.gt_uid_values = None
        self.gt_enc_t = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TransformerMixin:
        """Fit the cosine similarity indexers to ground truth names

        This creates TFIDF weights and matrix based on the ground truth names.

        Args:
            X: ground truth dataframe with preprocessed names.
            y: ignored.

        Returns:
            self
        """
        with Timer("PandasCosSimIndexer.fit") as timer:
            timer.log_params({"X.shape": X.shape})

            timer.label("fit")
            self.tfidf.fit(X[self.input_col])

            timer.label("vectorize")
            # vectorize ground truth (note: could be parallellized)
            self.gt_uid_values = X.index.values
            gt_enc = self.tfidf.transform_parallel(X[self.input_col], n_jobs=self.n_jobs)

            # transpose/block ground truth (note: could be parallellized)
            if self.blocking_func is not None:
                gt_blocking = X[self.input_col].map(self.blocking_func)
                # next: slow step
                self.gt_enc_t = groupby(gt_enc, gt_blocking, postprocess_func=lambda x: x.T)
                self.gt_uid_values = groupby(
                    self.gt_uid_values, gt_blocking, postprocess_func=lambda x: np.array(x, dtype="int64")
                )
            else:
                self.gt_enc_t = gt_enc.T

            timer.log_param("n", len(X))
        return self

    def transform(self, X: pd.DataFrame, multiple_indexers: bool | None = None) -> pd.DataFrame:
        """`transform` matches `X` dataset to the previously fitted ground truth.

        Args:
            X: Pandas dataframe with preprocessed names that should be matched
            multiple_indexers: ignored

        Returns:
            Pandas dataframe with the candidate matches returned by the indexer.
            Each row contains single pair of candidates.
            Columns `gt_uid`, `uid` contains index value from ground truth and X.
            Optionally id column (specified by `self.uid_col`) and carry on columns (specified by `self.carry_on_cols`)
            are copied from gt/X dataframes with the prefixes: `gt_` or `.
            Any additional columns calculated by indexers are also preserved (i.e. score).
        """
        do_blocking = self.blocking_func is not None

        with Timer("PandasCosSimIndexer.transform") as timer:
            timer.log_params({"n_jobs": self.n_jobs, "blocking": do_blocking})

            timer.label("tfidf")

            X_enc = self.tfidf.transform_parallel(X[self.input_col], n_jobs=self.n_jobs)
            if self.blocking_func is not None:
                X_blocking = X[self.input_col].map(self.blocking_func)
                X_enc = groupby(X_enc, X_blocking)
                uid = groupby(X.index.values, X_blocking, postprocess_func=pd.Index)

            def get_work_chunks() -> (
                Generator[tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray, np.ndarray], None, None]
            ):
                if self.blocking_func is not None:
                    for k, X_enc_part in X_enc.items():
                        if k in self.gt_enc_t:
                            yield self.gt_enc_t[k], X_enc_part, self.gt_uid_values[k], uid[k]
                else:
                    yield self.gt_enc_t, X_enc, self.gt_uid_values, X.index.values

            candidates: list[pd.DataFrame] | pd.DataFrame = []

            timer.label("cossim")
            for curr_gt_enc_t, curr_X_enc, curr_gt_uid_values, curr_uid_values in get_work_chunks():
                if self.spark_session is not None and len(X) > 2 * 10**5:
                    cossim = self._spark_cossim(curr_gt_enc_t, curr_X_enc)
                else:
                    cossim = self._local_cossim(curr_gt_enc_t, curr_X_enc)
                cossim = cossim.tocoo()
                assert cossim.dtype == self.dtype

                candidates.append(
                    pd.DataFrame(
                        {
                            "uid": np.take(curr_uid_values, cossim.row),
                            "gt_uid": np.take(curr_gt_uid_values, cossim.col),
                            "score": cossim.data,
                        }
                    )
                )

            timer.label("candidates")
            if len(candidates) == 0:
                candidates = pd.DataFrame(columns=["uid", "gt_uid", "score"])
            else:
                candidates = pd.concat(candidates, axis=0)

            # rank the candidates. note that rank starts at 1
            candidates = candidates.sort_values(by=["uid", "score"], ascending=False)
            # groupby preserves the order of the rows in each group. See:
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html (sort)
            gb = candidates.groupby("uid")
            candidates["rank"] = gb["score"].transform(lambda x: range(1, len(x) + 1))
            timer.log_param("n", len(X))
        return candidates

    def _local_cossim(
        self, gt_enc_t: scipy.sparse.csr_matrix, X_enc: scipy.sparse.csr_matrix
    ) -> scipy.sparse.csr_matrix:
        assert gt_enc_t.dtype == self.dtype
        assert X_enc.dtype == self.dtype
        logger.debug(f"calculating cossim gt_enc_t={gt_enc_t!r} X_enc={X_enc!r}")
        cs_list = []
        # Here we use chunks/partition_size just to get a progress bar
        # np.array_split cannot be used here due to sparse array X_enc
        X_chunks = [X_enc[i : i + self.partition_size] for i in range(0, X_enc.shape[0], self.partition_size)]
        for X_chunk in X_chunks:
            cossim = awesome_cossim_topn(
                X_chunk,
                gt_enc_t,
                self.num_candidates,
                self.cos_sim_lower_bound,
                n_jobs=self.n_jobs,
                use_threads=self.n_jobs > 1,
            )
            cs_list.append(cossim)
        return scipy.sparse.vstack(cs_list, dtype=self.dtype)

    def _spark_cossim(
        self, gt_enc_t: scipy.sparse.csr_matrix, X_enc: scipy.sparse.csr_matrix
    ) -> scipy.sparse.csr_matrix:
        """Spark implementation of cossim by Max Baak (adapted by TW)"""
        assert gt_enc_t.dtype == self.dtype
        assert X_enc.dtype == self.dtype
        logger.debug("calculating cossim using spark gt_enc_t=%s X_enc=%s", repr(gt_enc_t), repr(X_enc))
        sc = self.spark_session.sparkContext
        spark_gt = sc.broadcast(gt_enc_t)
        # np.array_split cannot be used here due to sparse array X_enc
        X_chunks = [X_enc[i : i + self.partition_size] for i in range(0, X_enc.shape[0], self.partition_size)]
        rdd = sc.parallelize(X_chunks, len(X_chunks))

        def calc(
            row: scipy.sparse.csr_matrix, num_candidates: scipy.sparse.csr_matrix, cos_sim_lower_bound: float
        ) -> scipy.sparse.csr_matrix:
            left = row
            right = spark_gt.value
            return awesome_cossim_topn(left, right, num_candidates, cos_sim_lower_bound)

        cs_rdd = rdd.map(
            partial(calc, num_candidates=self.num_candidates, cos_sim_lower_bound=self.cos_sim_lower_bound)
        )
        cs_list = cs_rdd.collect()
        return scipy.sparse.vstack(cs_list, dtype=self.dtype)

    def column_prefix(self) -> str:
        p1 = "w" if self.tokenizer == "words" else "n"
        return f"cossim_{p1}{self.ngram}"

    def calc_score(self, name1: pd.Series, name2: pd.Series) -> pd.DataFrame:
        assert all(name1.index == name2.index)
        name1_enc: scipy.sparse.csr_matrix = self.tfidf.transform(name1)
        name2_enc: scipy.sparse.csr_matrix = self.tfidf.transform(name2)
        cossim = name1_enc.multiply(name2_enc).sum(axis=1)
        cossim = np.array(cossim).flatten()
        return pd.DataFrame({"score": cossim}, index=name1.index)
