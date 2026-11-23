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

from fnmatch import filter as filter_names

import numpy as np
import polars as pl


def _negative_samples(candidates: pl.DataFrame, uid_col: str, correct_col: str, positive_set_col: str) -> pl.DataFrame:
    ranks = filter_names(candidates.columns, "rank_*")
    if not ranks:
        msg = "No columns with pattern rank_* present in input dataframe."
        raise AssertionError(msg)
    row_id = "__emm_training_row"
    while row_id in candidates.columns:
        row_id = f"_{row_id}"
    candidates = candidates.with_row_index(row_id)
    results = []
    for positive in (True, False):
        subset = candidates.filter(pl.col(positive_set_col) == positive)
        kept = []
        for rank in ranks:
            values = candidates[rank].drop_nulls()
            if values.dtype.is_float():
                values = values.filter(~values.is_nan())
            rank_min, rank_max = values.min(), values.max()
            if rank_min is None:
                continue
            limit = max(rank_max, abs(rank_min))
            current = subset.filter(pl.col(rank).is_not_null())
            if current.schema[rank].is_float():
                current = current.filter(~pl.col(rank).is_nan())
            if positive:
                current = current.filter(pl.col(rank).abs() < limit)
            elif rank_min < 0:
                correct_ranks = (
                    current.filter(pl.col(correct_col))
                    .group_by(uid_col, maintain_order=True)
                    .agg(pl.col(rank).last().alias("__correct_rank"))
                )
                current = (
                    current.filter(~pl.col(correct_col))
                    .sort([uid_col, rank], nulls_last=True, maintain_order=True)
                    .join(correct_ranks, on=uid_col, how="left", maintain_order="left")
                )
                same_rank = (pl.col(rank) == pl.col("__correct_rank")).any().over(uid_col)
                shift = (
                    pl.when((pl.col("__correct_rank") > 0) & (pl.col(rank) > pl.col("__correct_rank")))
                    .then(-1)
                    .when((pl.col("__correct_rank") < 0) & (pl.col(rank) < pl.col("__correct_rank")))
                    .then(1)
                    .otherwise(0)
                )
                current = current.with_columns(
                    pl.when(same_rank.fill_null(False)).then(pl.col(rank)).otherwise(pl.col(rank) + shift).alias(rank)
                ).filter(pl.col(rank).abs() < limit)
            else:
                current = (
                    current.filter(~pl.col(correct_col))
                    .sort([uid_col, rank], nulls_last=True, maintain_order=True)
                    .with_columns(pl.int_range(1, pl.len() + 1).over(uid_col).alias(rank))
                    .filter(pl.col(rank) < limit)
                )
            kept.append(current.select(row_id, pl.col(rank).alias(f"__new_{rank}")))
        if not kept:
            results.append(subset.head(0))
            continue
        selected = kept[0]
        for part in kept[1:]:
            selected = selected.join(part, on=row_id, how="full", coalesce=True)
        result = subset.join(selected, on=row_id, how="inner", maintain_order="left")
        for rank in ranks:
            new_rank = f"__new_{rank}"
            if new_rank in result.columns:
                result = result.with_columns(pl.col(new_rank).alias(rank)).drop(new_rank)
            else:
                result = result.with_columns(pl.lit(None, dtype=result.schema[rank]).alias(rank))
        results.append(result)
    return pl.concat(results, how="vertical_relaxed").drop(row_id)


def prepare_name_pairs_polars(
    candidates: pl.DataFrame,
    drop_duplicate_candidates: bool = False,
    drop_samename_nomatch: bool = False,
    create_negative_sample_fraction: float = 0,
    entity_id_col: str = "entity_id",
    gt_entity_id_col: str = "gt_entity_id",
    positive_set_col: str = "positive_set",
    correct_col: str = "correct",
    uid_col: str = "uid",
    gt_uid_col: str = "gt_uid",
    preprocessed_col: str = "preprocessed",
    gt_preprocessed_col: str = "gt_preprocessed",
    random_seed: int = 42,
) -> pl.DataFrame:
    """Label candidate pairs and construct negative examples using Polars."""
    for col in (entity_id_col, gt_entity_id_col):
        if col not in candidates.columns:
            msg = f"Column {col} not present in input dataframe."
            raise AssertionError(msg)
    candidates = candidates.with_columns(
        [pl.col(column).fill_nan(None) for column, dtype in candidates.schema.items() if dtype.is_float()]
    )
    candidates = candidates.with_columns(
        (pl.col(entity_id_col) == pl.col(gt_entity_id_col)).fill_null(False).alias(correct_col)
    )
    fraction = min(create_negative_sample_fraction, 1)
    if fraction > 0 and positive_set_col not in candidates.columns:
        ids = sorted(candidates[entity_id_col].unique().to_list())
        n_positive = int(len(ids) * (1.0 - fraction))
        selected = np.random.default_rng(random_seed).choice(ids, n_positive, replace=False).tolist()
        candidates = candidates.with_columns(pl.col(entity_id_col).is_in(selected).alias(positive_set_col))
    if drop_duplicate_candidates:
        candidates = candidates.sort(
            [uid_col, gt_preprocessed_col, correct_col], descending=True, nulls_last=True, maintain_order=True
        ).unique(subset=[uid_col, gt_preprocessed_col], keep="first", maintain_order=True)
    if drop_samename_nomatch:
        same = (pl.col(preprocessed_col) == pl.col(gt_preprocessed_col)).fill_null(False)
        candidates = candidates.filter(~(same & ~pl.col(correct_col)))

    grouped = (
        candidates.filter(pl.col(uid_col).is_not_null())
        .group_by(uid_col)
        .agg([pl.col(col).drop_nulls().n_unique().alias(col) for col in candidates.columns if col != uid_col])
    )
    max_counts = grouped.select(pl.all().exclude(uid_col).max()).row(0, named=True)
    names_cols = [uid_col, *[col for col in candidates.columns if col != uid_col and max_counts[col] == 1]]
    before = candidates.select(names_cols).unique(maintain_order=True)
    if fraction > 0:
        candidates = _negative_samples(candidates, uid_col, correct_col, positive_set_col)
    after = candidates.select(names_cols).unique(maintain_order=True)
    missing = before.join(after, on=names_cols, how="anti", nulls_equal=True).with_columns(
        pl.lit(False).alias(correct_col), pl.lit(1).alias("score_0_rank")
    )
    return pl.concat([candidates, missing], how="diagonal_relaxed").with_columns(
        pl.col(gt_preprocessed_col).fill_null(""), pl.col(gt_uid_col).is_null().alias("no_candidate")
    )
