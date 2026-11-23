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

from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.base import TransformerMixin

from emm.indexing.base_indexer import SNBaseIndexer
from emm.loggers import Timer


class PolarsSortedNeighbourhoodIndexer(TransformerMixin, SNBaseIndexer):
    """Polars transformer for sorted-neighbourhood indexing."""

    def __init__(
        self,
        input_col: str = "preprocessed",
        window_length: int = 3,
        mapping_func: Callable[[str], str] | None = None,
        indexer_id: int | None = None,
    ) -> None:
        SNBaseIndexer.__init__(self, window_length=window_length)
        self.input_col = input_col
        self.gt: pl.DataFrame | None = None
        self.mapping: dict[str, int] | None = None
        self.mapping_func = mapping_func

    def _map_names(self, names: list[Any]) -> list[str]:
        values = ["" if name is None else str(name) for name in names]
        if self.mapping_func is not None:
            values = [self.mapping_func(name) for name in values]
        return values

    def fit(self, X: pl.DataFrame, y: Any = None) -> PolarsSortedNeighbourhoodIndexer:
        self.gt = X.select(self.input_col).clone()
        gt_names = self._map_names(self.gt[self.input_col].to_list())
        self.gt = self.gt.with_columns(pl.Series(self.input_col, gt_names))
        self.mapping = None
        return self

    def transform(self, X: pl.DataFrame, multiple_indexers: bool = False) -> pl.DataFrame:
        if self.gt is None:
            msg = "model is not fitted yet"
            raise ValueError(msg)

        with Timer("PolarsSortedNeighbourhoodIndexer.transform") as timer:
            timer.log_param("n", X.height)
            timer.label("index")
            gt_names = self.gt[self.input_col].to_list()
            query_names = self._map_names(X[self.input_col].to_list())

            all_names = sorted(set(gt_names) | set(query_names))
            self.mapping = {name: rank for rank, name in enumerate(all_names)}
            gt_ranks = [self.mapping[name] for name in gt_names]
            query_ranks = [self.mapping[name] for name in query_names]

            timer.label("other")
            half_window = self.window_length // 2
            rows: list[tuple[int, int, int, np.float32]] = []
            for uid, query_rank in enumerate(query_ranks):
                for gt_uid, gt_rank in enumerate(gt_ranks):
                    rank = gt_rank - query_rank
                    if abs(rank) <= half_window:
                        score = self._score_formula(abs(rank), self.window_length)
                        rows.append((uid, gt_uid, rank, score))

            rows.sort(key=lambda row: (row[0], row[2], row[1]))
            result = pl.DataFrame(
                {
                    "uid": pl.Series([row[0] for row in rows], dtype=pl.Int64),
                    "gt_uid": pl.Series([row[1] for row in rows], dtype=pl.Int64),
                    "rank": pl.Series([row[2] for row in rows], dtype=pl.Int64),
                    "score": pl.Series([row[3] for row in rows], dtype=pl.Float32),
                }
            )
            timer.log_param("n", X.height)
        return result

    def calc_score(self, name1: pl.Series, name2: pl.Series) -> pl.DataFrame:
        if len(name1) != len(name2):
            msg = "name1 and name2 must have the same length"
            raise ValueError(msg)
        if self.mapping is None:
            msg = "no SNI mapping, calc_score is called before transform"
            raise AssertionError(msg)

        names1 = self._map_names(name1.to_list())
        names2 = self._map_names(name2.to_list())
        missing_names = set(names1 + names2) - self.mapping.keys()
        if missing_names:
            msg = f"names not found in SNI mapping: {sorted(missing_names)!r}"
            raise ValueError(msg)
        distances = [self.mapping[right] - self.mapping[left] for left, right in zip(names1, names2)]
        scores = self._score_formula(np.abs(distances), self.window_length).clip(0, 1)
        return pl.DataFrame(
            {"rank": pl.Series(distances, dtype=pl.Int64), "score": pl.Series(scores, dtype=pl.Float32)}
        )

    @staticmethod
    def _score_formula(sni_distance: Any, window_length: int) -> Any:
        half_window = window_length // 2
        if isinstance(sni_distance, (list, tuple, np.ndarray)):
            return ((half_window + 1 - np.asarray(sni_distance)) / (half_window + 1)).astype(np.float32)
        return np.float32((half_window + 1 - sni_distance) / (half_window + 1))

    def column_prefix(self) -> str:
        return "sni"

    @property
    def store_ground_truth(self) -> bool:
        return self.mapping_func is not None
