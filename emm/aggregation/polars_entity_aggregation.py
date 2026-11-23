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

from typing import Literal

import polars as pl
from sklearn.base import TransformerMixin

from emm.aggregation.base_entity_aggregation import BaseEntityAggregation
from emm.loggers import Timer


def _without_nan(df: pl.DataFrame, column: str) -> pl.Expr:
    value = pl.col(column)
    return value.fill_nan(None) if df.schema[column].is_float() else value


def _row_index_name(df: pl.DataFrame) -> str:
    name = "__emm_aggregation_row"
    while name in df.columns:
        name = f"_{name}"
    return name


def matching_max_candidate(  # noqa: PLR0917
    df: pl.DataFrame,
    group: list[str],
    score_col: str,
    name_col: str,
    account_col: str,
    freq_col: str,
    output_col: str,
    aggregation_method: Literal["max_frequency_nm_score", "mean_score"] = "max_frequency_nm_score",
) -> pl.DataFrame:
    """Select one representative candidate for a group of names."""
    if df.is_empty():
        msg = "Provided an empty df"
        raise ValueError(msg)

    if aggregation_method == "mean_score":
        scores = _without_nan(df, score_col)
        keys = [_without_nan(df, column) for column in group]
        if df.schema[score_col] == pl.Float32:
            # Pandas uses a Float64 mean fast path for sorted groups without missing scores.
            group_values = df.select(*[key.alias(column) for key, column in zip(keys, group, strict=True)])
            sorted_groups = group_values.equals(group_values.sort(group, nulls_last=True))
            if df.select(scores.is_null().any()).item() or not sorted_groups:
                mean = (scores.sum().over(keys) / scores.count().over(keys)).cast(pl.Float32)
            else:
                mean = scores.cast(pl.Float64).mean().over(keys)
        else:
            mean = scores.mean().over(keys)
        result = df.with_columns(mean.alias(output_col))
        if result.height == 1:
            return result
        return result.sort(
            [_without_nan(result, output_col), _without_nan(result, score_col)],
            descending=True,
            nulls_last=True,
            maintain_order=True,
        ).head(1)

    if aggregation_method != "max_frequency_nm_score":
        msg = "aggregation_method not supported"
        raise ValueError(msg)

    if df.height == 1:
        return df.with_columns(pl.col(score_col).alias(output_col))
    names = df[name_col].to_list()
    missing_string_name = names[0] is None and df.schema[name_col] in (pl.String, pl.Categorical, pl.Enum)
    if not missing_string_name and all(name == names[0] for name in names):
        return (
            df.sort(_without_nan(df, score_col), descending=True, nulls_last=True, maintain_order=True)
            .head(1)
            .with_columns(pl.col(score_col).alias(output_col))
        )

    row_id = _row_index_name(df)
    work = df.with_row_index(row_id).with_columns(
        *[_without_nan(df, column).alias(column) for column in group],
        (_without_nan(df, freq_col) * _without_nan(df, score_col)).alias("freq_score"),
    )
    grouped = (
        work.group_by(group)
        .agg(
            _without_nan(work, freq_col).sum().alias(freq_col),
            _without_nan(work, "freq_score").sum().alias("freq_score"),
        )
        .with_columns((pl.col("freq_score") / pl.col(freq_col)).alias(output_col))
        .sort(group, nulls_last=True, maintain_order=True)
    )
    winner = grouped.sort(
        _without_nan(grouped, "freq_score"), descending=True, nulls_last=True, maintain_order=True
    ).head(1)
    selected = (
        work.join(winner.select(group), on=group, how="semi", nulls_equal=True, maintain_order="left")
        .sort(_without_nan(work, "freq_score"), descending=True, nulls_last=True, maintain_order=True)
        .head(1)
    )
    return (
        df.with_row_index(row_id)
        .filter(pl.col(row_id) == selected[row_id][0])
        .drop(row_id, "freq_score", strict=False)
        .with_columns(pl.lit(winner[output_col][0], dtype=winner.schema[output_col]).alias(output_col))
    )


class PolarsEntityAggregation(TransformerMixin, BaseEntityAggregation):
    """Aggregate candidate scores per account using native Polars operations."""

    def __init__(  # noqa: PLR0917
        self,
        score_col: str,
        account_col: str = "account",
        index_col: str = "entity_id",
        gt_entity_id_col: str = "gt_entity_id",
        uid_col: str = "uid",
        gt_uid_col: str = "gt_uid",
        name_col: str = "name",
        freq_col: str = "counterparty_account_count_distinct",
        output_col: str = "agg_score",
        preprocessed_col: str = "preprocessed",
        gt_name_col: str = "gt_name",
        gt_preprocessed_col: str = "gt_preprocessed",
        correct_col: str = "correct",
        aggregation_method: Literal["max_frequency_nm_score", "mean_score"] = "max_frequency_nm_score",
        blacklist: list[str] | None = None,
    ) -> None:
        BaseEntityAggregation.__init__(
            self,
            score_col=score_col,
            account_col=account_col,
            index_col=index_col,
            gt_entity_id_col=gt_entity_id_col,
            uid_col=uid_col,
            gt_uid_col=gt_uid_col,
            name_col=name_col,
            freq_col=freq_col,
            output_col=output_col,
            preprocessed_col=preprocessed_col,
            gt_name_col=gt_name_col,
            gt_preprocessed_col=gt_preprocessed_col,
            correct_col=correct_col,
            aggregation_method=aggregation_method,
            blacklist=blacklist or [],
        )

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> PolarsEntityAggregation:
        """Return self; aggregation does not require fitting."""
        return self

    def fit_transform(self, X: pl.DataFrame, y: pl.Series | None = None) -> pl.DataFrame | None:
        """Aggregate the input without fitting."""
        return self.transform(X)

    def transform(self, X: pl.DataFrame | None) -> pl.DataFrame | None:
        """Select an aggregated candidate for each account group."""
        if X is None:
            return None
        if not isinstance(X, pl.DataFrame):
            msg = f"expected a Polars DataFrame, got {type(X).__name__}"
            raise TypeError(msg)

        with Timer("PolarsEntityAggregation.transform") as timer:
            timer.log_param("n", X.height)
            group = self.get_group(X)
            gt_group = self.get_gt_group()
            valid = _without_nan(X, self.gt_uid_col).is_not_null()
            for column in group:
                valid &= _without_nan(X, column).is_not_null()
            matches = X.filter(valid)
            size_col = "__emm_aggregation_size"
            while size_col in matches.columns:
                size_col = f"_{size_col}"
            matches = matches.with_columns(pl.len().over(group).alias(size_col))
            single = (
                matches.filter(pl.col(size_col) == 1)
                .drop(size_col)
                .with_columns(
                    pl.col(self.score_col).alias(self.output_col),
                    (pl.col(self.score_col) * pl.col(self.freq_col)).alias("freq_score"),
                )
            )
            multiple = matches.filter(pl.col(size_col) > 1).drop(size_col)
            multiple = self.remove_blacklisted_names(multiple, self.preprocessed_col)
            multiple = multiple.sort(group, nulls_last=True, maintain_order=True)
            results = [single]
            results.extend(
                matching_max_candidate(
                    current,
                    group=gt_group,
                    score_col=self.score_col,
                    name_col=self.name_col,
                    account_col=self.account_col,
                    freq_col=self.freq_col,
                    output_col=self.output_col,
                    aggregation_method=self.aggregation_method,
                )
                for current in multiple.partition_by(group, maintain_order=True)
            )
            result = pl.concat(results, how="diagonal_relaxed").with_columns(
                pl.lit(True).alias("best_match"), pl.lit(1, dtype=pl.Int64).alias("best_rank")
            )
            timer.log_param("cands", result.height)
            return result

    def remove_blacklisted_names(self, df: pl.DataFrame, preprocessed_col: str = "preprocessed") -> pl.DataFrame:
        """Remove empty and blacklisted preprocessed names."""
        if preprocessed_col not in df.columns:
            return df
        return df.filter(~pl.col(preprocessed_col).is_in([*self.blacklist, ""]).fill_null(False))
