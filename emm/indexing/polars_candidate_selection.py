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

from typing import Any

import polars as pl
from sklearn.base import TransformerMixin

from emm.indexing.polars_sni import PolarsSortedNeighbourhoodIndexer
from emm.loggers import Timer


def select_with_prefix(df: pl.DataFrame, cols: list[str], prefix: str) -> pl.DataFrame:
    selected = [column for column in cols if column in df.columns]
    return df.select(selected).rename({column: f"{prefix}_{column}" for column in selected})


def _row_index_name(df: pl.DataFrame, prefix: str) -> str:
    name = prefix
    while name in df.columns:
        name = f"_{name}"
    return name


def _with_row_index(df: pl.DataFrame, prefix: str) -> tuple[pl.DataFrame, str]:
    name = _row_index_name(df, prefix)
    return df.with_row_index(name).with_columns(pl.col(name).cast(pl.Int64)), name


class PolarsCandidateSelectionTransformer(TransformerMixin):
    """Polars middleware that aggregates indexer candidate pairs."""

    def __init__(
        self,
        indexers: list[Any],
        uid_col: str | None = None,
        carry_on_cols: list[str] | None = None,
        with_no_matches: bool | None = True,
    ) -> None:
        self.indexers = indexers
        self.uid_col = uid_col
        self.carry_on_cols = carry_on_cols
        self.with_no_matches = with_no_matches
        self.gt: pl.DataFrame | None = None

    def fit(self, X: pl.DataFrame, y: Any = None) -> PolarsCandidateSelectionTransformer:
        with Timer("PolarsCandidateSelectionTransformer.fit") as timer:
            timer.log_params({"X.shape": X.shape, "n_indexers": len(self.indexers)})
            self.gt = X.clone()
            for indexer in self.indexers:
                indexer.fit(X, y)
            timer.log_param("n", X.height)
        return self

    def fit_transform(self, X: pl.DataFrame, y: Any = None) -> pl.DataFrame:
        self.fit(X, y)
        return X

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.gt is None:
            msg = "model is not fitted yet"
            raise ValueError(msg)

        with Timer("PolarsCandidateSelectionTransformer.transform") as timer:
            timer.log_param("n_indexers", len(self.indexers))
            multiple_indexers = len(self.indexers) > 1
            candidates: pl.DataFrame | None = None
            indexers_map: dict[str, str] = {}

            for i, indexer in enumerate(self.indexers):
                current = indexer.transform(X, multiple_indexers=multiple_indexers)
                rename = {column: f"{column}_{i}" for column in ("score", "rank") if column in current.columns}
                current = current.rename(rename)
                if multiple_indexers:
                    prefix = indexer.column_prefix()
                    current = current.with_columns(pl.lit(1, dtype=pl.Int8).alias(prefix))
                    indexers_map[prefix] = f"score_{i}"

                if candidates is None:
                    candidates = current
                    continue

                overlaps = (set(candidates.columns) & set(current.columns)) - {"uid", "gt_uid"}
                candidates = candidates.join(
                    current, on=["gt_uid", "uid"], how="full", coalesce=True, suffix="_current"
                )
                for column in overlaps:
                    current_column = f"{column}_current"
                    if current_column not in candidates.columns:
                        continue
                    if candidates.schema[column].base_type() == pl.List:
                        combined = pl.col(column).list.concat(pl.col(current_column))
                    elif candidates.schema[column] == pl.String:
                        combined = pl.concat_str([pl.col(column), pl.col(current_column)], separator="")
                    else:
                        combined = pl.coalesce(column, current_column)
                    candidates = candidates.with_columns(combined.alias(column)).drop(current_column)

            if candidates is None:
                msg = "at least one indexer is required"
                raise ValueError(msg)

            if multiple_indexers:
                # Pandas outer joins sort the candidate index, which also determines tie order.
                candidates = candidates.with_columns(
                    [pl.col(column).fill_null(0).cast(pl.Int8) for column in indexers_map]
                ).sort(["gt_uid", "uid"], maintain_order=True)

            timer.log_param("n_cands", candidates.height)
            timer.label("add columns")

            if self.with_no_matches:
                if self.uid_col is None or self.uid_col not in self.gt.columns:
                    msg = "uid_col must identify a ground-truth entity ID when with_no_matches is enabled"
                    raise ValueError(msg)
                if -1 in candidates["gt_uid"].to_list():
                    msg = "ground-truth row index -1 is reserved for unmatched candidates"
                    raise ValueError(msg)
                matched_uids = set(candidates["uid"].to_list())
                missing_uids = [uid for uid in range(X.height) if uid not in matched_uids]
                if missing_uids:
                    unmatched = pl.DataFrame(
                        {
                            "uid": pl.Series(missing_uids, dtype=pl.Int64),
                            "gt_uid": pl.Series([None] * len(missing_uids), dtype=pl.Int64),
                            **{column: pl.Series([0] * len(missing_uids), dtype=pl.Int8) for column in indexers_map},
                        }
                    )
                    candidates = pl.concat([candidates, unmatched], how="diagonal_relaxed")

            gt_index, gt_row_id = _with_row_index(self.gt, "__gt_row_id")
            query_index, query_row_id = _with_row_index(X, "__query_row_id")

            if self.uid_col is not None:
                if self.uid_col in gt_index.columns:
                    gt_values = select_with_prefix(gt_index, [self.uid_col], "gt").with_columns(gt_index[gt_row_id])
                    candidates = candidates.join(
                        gt_values, left_on="gt_uid", right_on=gt_row_id, how="left", maintain_order="left"
                    )
                if self.uid_col in query_index.columns:
                    query_values = query_index.select(query_row_id, self.uid_col).rename({query_row_id: "uid"})
                    candidates = candidates.join(query_values, on="uid", how="left", maintain_order="left")

            if self.carry_on_cols:
                gt_cols = [column for column in self.carry_on_cols if column in gt_index.columns]
                if gt_cols:
                    gt_values = select_with_prefix(gt_index, gt_cols, "gt").with_columns(gt_index[gt_row_id])
                    candidates = candidates.join(
                        gt_values, left_on="gt_uid", right_on=gt_row_id, how="left", maintain_order="left"
                    )
                query_cols = [column for column in self.carry_on_cols if column in query_index.columns]
                if query_cols:
                    query_values = query_index.select(query_row_id, *query_cols).rename({query_row_id: "uid"})
                    candidates = candidates.join(query_values, on="uid", how="left", maintain_order="left")

            timer.log_param("n", X.height)
        return candidates

    def increase_window_by_one_step(self) -> None:
        for indexer in self.indexers:
            indexer.increase_window_by_one_step()

    def decrease_window_by_one_step(self) -> None:
        for indexer in self.indexers:
            indexer.decrease_window_by_one_step()

    def _set_sni_ground_truth(self) -> None:
        for indexer in self.indexers:
            if isinstance(indexer, PolarsSortedNeighbourhoodIndexer) and not indexer.store_ground_truth:
                indexer.fit(self.gt)

    def _reset_sni_ground_truth(self) -> None:
        for indexer in self.indexers:
            if isinstance(indexer, PolarsSortedNeighbourhoodIndexer) and not indexer.store_ground_truth:
                indexer.gt = None
