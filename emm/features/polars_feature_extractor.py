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

import re
from typing import Callable

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Jaro, Levenshtein
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from emm.features.base_feature_extractor import BaseFeatureExtractor
from emm.features.features_lef import extract_lef, matching_legal_terms
from emm.features.features_name import abbr_match, abs_len_diff, len_ratio, name_cut
from emm.features.features_rank import RANK_FEATURES_PRECISION
from emm.features.features_vocabulary import Vocabulary
from emm.loggers import Timer


def _create_vocabulary(df: pl.DataFrame, columns: list[str]) -> Vocabulary:
    names = pl.concat([df.select(pl.col(column).alias("name")) for column in columns])
    documents = names["name"].drop_nulls().unique(maintain_order=True).to_list()

    def words(min_df: float) -> set[str]:
        try:
            vectorizer = CountVectorizer(min_df=min_df)
            vectorizer.fit(documents)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc) and "After pruning, no terms remain" not in str(exc):
                raise
            return set()
        return set(vectorizer.vocabulary_)

    very_common_words = words(0.01)
    return Vocabulary(very_common_words=very_common_words, common_words=words(0.0001) - very_common_words)


def _vocabulary_features(name1: list[str], name2: list[str], vocabulary: Vocabulary) -> pl.DataFrame:
    values: dict[str, list[float]] = {
        key: []
        for key in (
            "very_common_hit",
            "common_hit",
            "rare_hit",
            "very_common_miss",
            "common_miss",
            "rare_miss",
            "n_overlap_words",
            "ratio_overlap_words",
            "num_word_difference",
        )
    }
    common = vocabulary.common_words
    very_common = vocabulary.very_common_words
    vocab = common | very_common
    for left, right in zip(name1, name2):
        left_words = set(re.findall(r"\w\w+", left))
        right_words = set(re.findall(r"\w\w+", right))
        hits = left_words & right_words
        total = left_words | right_words
        misses = total - hits
        row = (
            len(hits & very_common),
            len(hits & common),
            len(hits - vocab),
            len(misses & very_common),
            len(misses & common),
            len(misses - vocab),
            len(hits),
            len(hits) / len(total) if total else float("nan"),
            abs(len(left_words) - len(right_words)),
        )
        for key, value in zip(values, row):
            values[key].append(value)
    return pl.DataFrame([pl.Series(key, value, dtype=pl.Float32) for key, value in values.items()])


class PolarsFeatureExtractor(TransformerMixin, BaseFeatureExtractor):
    """Calculate name-pair, vocabulary, score, and optional legal-form features with Polars."""

    def __init__(  # noqa: PLR0917
        self,
        name1_col: str = "preprocessed",
        name2_col: str = "gt_preprocessed",
        uid_col: str = "uid",
        gt_uid_col: str = "gt_uid",
        score_columns: list[str] | None = None,
        extra_features: list[str | tuple[str, Callable]] | None = None,
        vocabulary: Vocabulary | None = None,
        without_rank_features: bool = False,
        with_legal_entity_forms_match: bool = False,
        fillna_value: float | None = None,
        drop_features: list[str] | None = None,
    ) -> None:
        self.name1_col = name1_col
        self.name2_col = name2_col
        self.uid_col = uid_col
        self.gt_uid_col = gt_uid_col
        self.score_columns = score_columns or []
        self.extra_features = extra_features or []
        self.vocabulary = vocabulary
        self.without_rank_features = without_rank_features
        self.with_legal_entity_forms_match = with_legal_entity_forms_match
        self.fillna_value = fillna_value
        self.drop_features = drop_features
        super().__init__()
        self.name_features = {
            "abbr_match": (abbr_match, "int8"),
            "abs_len_diff": (abs_len_diff, "int8"),
            "len_ratio": (len_ratio, "float32"),
            "token_sort_ratio": (fuzz.token_sort_ratio, "int8"),
            "token_set_ratio": (fuzz.token_set_ratio, "int8"),
            "partial_ratio": (fuzz.partial_ratio, "int8"),
            "w_ratio": (fuzz.WRatio, "int8"),
            "ratio": (fuzz.ratio, "int8"),
            "name_cut": (name_cut, "int8"),
            "norm_ed": (Levenshtein.distance, "int8"),
            "norm_jaro": (Jaro.similarity, "float32"),
        }
        self._fitted = False

    def fit(self, X: pl.DataFrame | None, y: pl.Series | None = None) -> PolarsFeatureExtractor:
        """Fit the vocabulary from Polars candidate pairs, if needed."""
        if X is not None and not isinstance(X, pl.DataFrame):
            msg = f"expected a Polars DataFrame, got {type(X).__name__}"
            raise TypeError(msg)
        if X is not None and self.vocabulary is None:
            self.vocabulary = _create_vocabulary(X, [self.name1_col, self.name2_col])
        self._fitted = True
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Calculate features and return a Polars DataFrame."""
        if not isinstance(X, pl.DataFrame):
            msg = f"expected a Polars DataFrame, got {type(X).__name__}"
            raise TypeError(msg)
        if not self._fitted:
            self.fit(X)
        if self.vocabulary is None:
            msg = "vocabulary is not fitted"
            raise ValueError(msg)

        with Timer("PolarsFeatureExtractor.transform") as timer:
            timer.log_param("cands", X.height)
            row_id = "__emm_feature_row"
            while row_id in X.columns:
                row_id = f"_{row_id}"
            data = X.with_row_index(row_id).sort(
                [self.name1_col, self.name2_col, self.uid_col], nulls_last=True, maintain_order=True
            )
            data = data.with_columns([pl.col(col).cast(pl.Float32) for col in self.score_columns])
            if isinstance(self.fillna_value, float):
                for col in data.columns:
                    dtype = data.schema[col]
                    if dtype.is_numeric() and (data[col].null_count() or dtype.is_float()):
                        expr = pl.col(col).fill_null(self.fillna_value)
                        if dtype.is_float():
                            expr = expr.fill_nan(self.fillna_value)
                        data = data.with_columns(expr.alias(col))

            names1 = data[self.name1_col].to_list()
            names2 = data[self.name2_col].to_list()
            names1 = [str(name) for name in names1]
            names2 = [str(name) for name in names2]
            result = data.select(row_id, *self.score_columns)
            for name, (func, dtype) in self.name_features.items():
                values = np.vectorize(func)(names1, names2).astype(dtype) if names1 else np.array([], dtype=dtype)
                result = result.with_columns(pl.Series(name, values))
            result = result.hstack(_vocabulary_features(names1, names2, self.vocabulary))

            for feature in self.extra_features:
                feature_name, func = feature if isinstance(feature, tuple) else (feature, None)
                gt_feature_name = f"gt_{feature_name}"
                if feature_name not in data.columns or gt_feature_name not in data.columns:
                    msg = (
                        f"missing extra features columns ('{feature_name}', '{gt_feature_name}') "
                        f"in df.columns={X.columns}"
                    )
                    raise ValueError(msg)
                left, right = data[feature_name], data[gt_feature_name]
                left_values, right_values = left.to_list(), right.to_list()
                if isinstance(self.fillna_value, float):
                    left_values = [self.fillna_value if value is None else value for value in left_values]
                    right_values = [self.fillna_value if value is None else value for value in right_values]
                if func is not None:
                    values = np.vectorize(func)(left_values, right_values)
                    result = result.with_columns(pl.Series(feature_name, values))
                else:

                    def missing(value: object) -> bool:
                        return value is None or (isinstance(value, float) and np.isnan(value))

                    values = [
                        0 if missing(a) or missing(b) else 1 if a == b else -1
                        for a, b in zip(left_values, right_values)
                    ]
                    result = result.with_columns(pl.Series(feature_name, values, dtype=pl.Int64))

            if not self.without_rank_features:
                result = result.hstack(self._rank_features(data, row_id))
            if self.with_legal_entity_forms_match:
                legal_matches = [
                    matching_legal_terms(extract_lef(left), extract_lef(right)) for left, right in zip(names1, names2)
                ]
                result = result.with_columns(pl.Series("match_legal_entity_form", legal_matches).cast(pl.Categorical))
            if self.drop_features:
                result = result.drop(self.drop_features)
            return result.sort(row_id).drop(row_id)

    def _rank_features(self, data: pl.DataFrame, row_id: str) -> pl.DataFrame:
        if data.schema[self.uid_col].is_float():
            data = data.with_columns(pl.col(self.uid_col).fill_nan(None))
        features = data.select(row_id)
        for column in self.score_columns:
            valid_score = pl.col(column).fill_nan(None)
            valid_uid = pl.col(self.uid_col).is_not_null()
            rounded = valid_score.round(RANK_FEATURES_PRECISION)
            maximum = valid_score.max().over(self.uid_col)
            minimum = valid_score.min().over(self.uid_col)
            top2 = valid_score.sort(descending=True, nulls_last=True).head(2).drop_nulls()
            non_null_count = valid_score.count().over(self.uid_col)
            group_size = pl.len().over(self.uid_col)
            expressions = {
                "rank": rounded.rank(method="ordinal", descending=True).over(self.uid_col),
                "top2_dist": pl.when(non_null_count < pl.min_horizontal(group_size, 2))
                .then(None)
                .otherwise((top2.max() - top2.min()).over(self.uid_col)),
                "dist_to_max": maximum - valid_score,
                "dist_to_min": valid_score - minimum,
                "ptp": pl.when(valid_score.is_null().any().over(self.uid_col)).then(None).otherwise(maximum - minimum),
            }
            for name, expr in expressions.items():
                dtype = pl.Int8 if name == "rank" else pl.Float32
                value = expr.fill_null(-1)
                if name == "rank":
                    value = (value + 128) % 256 - 128
                    value = pl.when(valid_uid).then(value).otherwise(None)
                else:
                    value = pl.when(valid_uid).then(value).otherwise(-1)
                features = features.with_columns(data.select(value.cast(dtype).alias(f"{column}_{name}")).to_series())
            sorted_data = data.with_columns(pl.col(column).fill_nan(None).round(RANK_FEATURES_PRECISION)).sort(
                [self.uid_col, column], descending=[False, True], nulls_last=True, maintain_order=True
            )
            value = pl.col(column)
            diffs = sorted_data.select(
                row_id,
                pl.when(pl.col(self.uid_col).is_not_null())
                .then((value - value.shift(1).over(self.uid_col)).abs().fill_null(-1))
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"{column}_diff_to_next"),
                pl.when(pl.col(self.uid_col).is_not_null())
                .then((value - value.shift(-1).over(self.uid_col)).fill_null(-1))
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"{column}_diff_to_prev"),
            )
            features = features.join(diffs, on=row_id, how="left", maintain_order="left")
        rank_columns = [
            f"{column}_{name}"
            for column in self.score_columns
            for name in ("rank", "top2_dist", "dist_to_max", "dist_to_min", "ptp")
        ]
        diff_columns = [
            f"{column}_{name}" for column in self.score_columns for name in ("diff_to_next", "diff_to_prev")
        ]
        return features.select(*rank_columns, *diff_columns)
