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

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from emm.indexing.base_indexer import SNBaseIndexer
from emm.loggers import Timer


class SortedNeighbourhood:
    """Generate sorted-neighbourhood pairs between two pandas dataframes.

    Adapted from recordlinkage's SortedNeighbourhood; see NOTICE for its license.
    """

    def __init__(self, left_on: str, right_on: str | None = None, window: int = 3) -> None:
        self.left_on = left_on
        self.right_on = left_on if right_on is None else right_on
        self.window = window
        self.sorting_key_values: np.ndarray | None = None

    def index(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.MultiIndex:
        """Return pairs of index labels within the window of distinct sorting keys."""
        if left is None or right is None:
            msg = "provide two dataframes"
            raise ValueError(msg)
        if not left.index.is_unique or not right.index.is_unique:
            msg = "index of DataFrame is not unique"
            raise ValueError(msg)
        if not isinstance(self.window, int) or self.window < 1 or self.window % 2 == 0:
            msg = "window is not a positive and odd integer"
            raise ValueError(msg)

        left_values = left[self.left_on].reset_index(drop=True)
        right_values = right[self.right_on].reset_index(drop=True)
        left_rows = pd.DataFrame({"key": left_values, "left_pos": np.arange(len(left))}).dropna(subset=["key"])
        right_rows = pd.DataFrame({"key": right_values, "right_pos": np.arange(len(right))}).dropna(subset=["key"])

        if self.sorting_key_values is None:
            self.sorting_key_values = np.unique(np.concatenate([left_rows["key"].values, right_rows["key"].values]))
        ranks = pd.Series(np.arange(len(self.sorting_key_values)), index=self.sorting_key_values)
        for rows in (left_rows, right_rows):
            if rows.empty:
                rows["key"] = pd.Series(index=rows.index, dtype="int64")
            else:
                rows["key"] = rows["key"].map(ranks)

        pairs = []
        radius = self.window // 2
        for offset in range(-radius, radius + 1):
            shifted = right_rows.assign(key=right_rows["key"] + offset)
            pairs.append(left_rows.merge(shifted, on="key", how="inner"))
        positions = pd.concat(pairs, ignore_index=True)

        names = [left.index.name, right.index.name]
        if pd.notna(names[0]) and pd.notna(names[1]) and names[0] == names[1]:
            names = [f"{names[0]}_1", f"{names[1]}_2"]
        return pd.MultiIndex(
            levels=[left.index, right.index],
            codes=[positions["left_pos"].to_numpy(), positions["right_pos"].to_numpy()],
            names=names,
            verify_integrity=False,
        )


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
        self.sni: SortedNeighbourhood | None = None
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
            self.sni = SortedNeighbourhood(left_on=self.input_col, right_on=self.input_col, window=self.window_length)
            if self.gt is None:
                msg = "provide at least one dataframe"
                raise ValueError(msg)
            idx = self.sni.index(self.gt, names)

            timer.label("other")
            candidates = pd.DataFrame({"uid": idx.get_level_values(1).values, "gt_uid": idx.get_level_values(0).values})

            # Calculate SNI distance using the indexer's sorting key ranks.
            assert self.sni.sorting_key_values is not None
            self.mapping = pd.Series(np.arange(len(self.sni.sorting_key_values)), index=self.sni.sorting_key_values)
            gt_rank = self.gt.loc[candidates.gt_uid][self.input_col].map(self.mapping).values
            X_rank = names.loc[candidates.uid][self.input_col].map(self.mapping).values
            candidates["rank"] = gt_rank.astype(int) - X_rank.astype(int)
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
