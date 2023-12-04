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

from typing import Any, Callable

import numpy as np
import pandas as pd
import recordlinkage
from sklearn.base import TransformerMixin

from emm.indexing.base_indexer import SNBaseIndexer
from emm.loggers import Timer


class PandasSortedNeighbourhoodIndexer(TransformerMixin, SNBaseIndexer):
    """Pandas transformer for sorted neighbourhood indexing"""

    def __init__(
        self,
        input_col: str = "preprocessed",
        window_length: int = 3,
        mapping_func: Callable[[str], str] | None = None,
        indexer_id: int | None = None,
    ) -> None:
        """Pandas transformer for sorted neighbourhood indexing

        For generating name-pair candidates using sorted neighbourhood indexing.
        The most important setting is "window_length".

        Args:
            input_col: (preprocessed) name column, default is "preprocessed".
            window_length: size of SNI window (odd integer).
            mapping_func: python function that should be applied to names before SNI indexing (i.e. name reversal)
            indexer_id: ignored. (needed for spark indexers.)

        Examples:
            >>> c = PandasSortedNeighbourhoodIndexer(window_length=5)
            >>> c.fit(ground_truth_df)
            >>> candidates_sdf = c.transform(names_df)

        """
        SNBaseIndexer.__init__(self, window_length=window_length)
        self.input_col = input_col
        self.gt: pd.DataFrame | None = None
        self.sni: Any | None = None
        self.mapping: pd.Series | None = None
        self.mapping_func: Callable | None = mapping_func

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TransformerMixin:
        """Default Estimator action on fitting with ground truth names.

        If custom mapping function is defined, then it is applied.

        Args:
            X: data frame with ground truth names
            y: ignored

        Returns:
            self
        """
        self.gt = X[[self.input_col]]
        if self.mapping_func is not None:
            self.gt = self.gt.copy()
            self.gt[self.input_col] = self.gt[self.input_col].map(self.mapping_func)
        return self

    def transform(self, X: pd.DataFrame, multiple_indexers: bool = False) -> pd.DataFrame:
        """Default Model action on transforming names to match

        Args:
            X: dataframe with names to match.
            multiple_indexers: ignored.

        Returns:
            dataframe with candidate SNI name-pairs
        """
        with Timer("SortedNeighbourhoodIndexer.transform") as timer:
            timer.log_param("n", len(X))
            timer.label("index")
            names = X[[self.input_col]]
            if self.mapping_func is not None:
                names = names.copy()
                names[self.input_col] = names[self.input_col].map(self.mapping_func)
            self.sni = recordlinkage.index.SortedNeighbourhood(
                left_on=self.input_col, right_on=self.input_col, window=self.window_length
            )
            idx: pd.Index = self.sni.index(self.gt, names)

            timer.label("other")
            candidates = pd.DataFrame({"uid": idx.get_level_values(1).values, "gt_uid": idx.get_level_values(0).values})

            # calculate sni distance, WARNING this is based on recordlinkage internals
            self.mapping = pd.Series(np.arange(len(self.sni.sorting_key_values)), index=self.sni.sorting_key_values)
            assert self.gt is not None
            gt_rank = self.gt.loc[candidates.gt_uid][self.input_col].map(self.mapping).values
            X_rank = names.loc[candidates.uid][self.input_col].map(self.mapping).values
            candidates["rank"] = (gt_rank - X_rank).astype(int)
            assert all(candidates["rank"].abs() <= self.window_length // 2)
            candidates["score"] = self._score_formula(candidates["rank"].abs(), self.window_length)
            assert all(candidates["score"] > 0)
            assert all(candidates["score"] <= 1)

            timer.log_param("n", len(X))
        return candidates

    def calc_score(self, name1: pd.Series, name2: pd.Series) -> pd.DataFrame:
        assert all(name1.index == name2.index)
        assert self.mapping is not None, "no sni mapping, calc_score is called before transform"
        # warning! this works only for names from GT or names_to_match
        if self.mapping_func is not None:
            name1 = name1.map(self.mapping_func)
            name2 = name2.map(self.mapping_func)
        name1_rank = name1.map(self.mapping)
        assert all(name1_rank.notnull())
        name2_rank = name2.map(self.mapping)
        assert all(name2_rank.notnull())
        sni_distance = (name2_rank - name1_rank).astype(int)
        score = self._score_formula(sni_distance.abs(), window_length=self.window_length).clip(0, 1)
        return pd.DataFrame({"rank": sni_distance, "score": score}, index=name1.index)

    def _score_formula(self, sni_distance: pd.Series, window_length: int) -> pd.Series:
        w = window_length // 2
        return (w + 1 - sni_distance).astype("float32") / (w + 1)

    def column_prefix(self) -> str:
        return "sni"

    @property
    def store_ground_truth(self) -> bool:
        return self.mapping_func is not None
