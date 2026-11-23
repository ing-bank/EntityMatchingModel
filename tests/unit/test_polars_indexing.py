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

from importlib import import_module

import numpy as np
import pytest

pl = pytest.importorskip("polars")

PolarsCandidateSelectionTransformer = import_module(
    "emm.indexing.polars_candidate_selection"
).PolarsCandidateSelectionTransformer
PolarsCosSimIndexer = import_module("emm.indexing.polars_cos_sim_matcher").PolarsCosSimIndexer
PolarsNormalizedTfidfVectorizer = import_module("emm.indexing.polars_normalized_tfidf").PolarsNormalizedTfidfVectorizer
PolarsSortedNeighbourhoodIndexer = import_module("emm.indexing.polars_sni").PolarsSortedNeighbourhoodIndexer


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polars_tfidf_normalization(dtype):
    vectorizer = PolarsNormalizedTfidfVectorizer(binary=False, dtype=dtype)
    ground_truth = pl.Series("name", ["a", "b", "c", "a c"])
    vectorizer.fit(ground_truth)

    assert set(vectorizer.vocabulary_) == {"a", "b", "c"}
    for name, expected in {
        "a": [1, 0, 0],
        "b": [0, 1, 0],
        "e": [0, 0, 0],
        "a b": [0.48693549, 0.87343794, 0],
        "b e": [0, 0.707107, 0],
        "a e": [0.48693549, 0, 0],
    }.items():
        result = vectorizer.transform(pl.Series("name", [name]))
        assert result.dtype == dtype
        np.testing.assert_allclose(result.toarray()[0], expected, rtol=0, atol=0.001)


def test_polars_tfidf_binary_character_ngrams():
    vectorizer = PolarsNormalizedTfidfVectorizer(binary=True, analyzer="char", ngram_range=(3, 3))
    vectorizer.fit(pl.Series("name", ["aaab", "bbbc"]))

    assert set(vectorizer.vocabulary_) == {"aaa", "aab", "bbb", "bbc"}
    result = vectorizer.transform(pl.Series("name", ["aaa", "bbb", "ccc", "ddd", "aaabbb"]))
    expected = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0]])
    np.testing.assert_allclose(result.toarray(), expected, rtol=0, atol=0.001)


def test_polars_sni_and_calc_score():
    ground_truth = pl.DataFrame({"name": ["a", "b", "c", "d", "f"]})
    indexer = PolarsSortedNeighbourhoodIndexer("name", window_length=3).fit(ground_truth)

    for query_name, expected in [
        ("a", [["a", 0], ["b", 1]]),
        ("c", [["b", -1], ["c", 0], ["d", 1]]),
        ("ca", [["c", -1], ["d", 1]]),
        ("e", [["d", -1], ["f", 1]]),
    ]:
        query = pl.DataFrame({"name": [query_name]})
        candidates = indexer.transform(query)
        actual = [
            [ground_truth["name"][gt_uid], rank]
            for gt_uid, rank in zip(candidates["gt_uid"].to_list(), candidates["rank"].to_list())
        ]
        assert actual == expected

        query_names = pl.Series("name1", [query_name] * len(actual))
        gt_names = pl.Series("name2", [row[0] for row in actual])
        scores = indexer.calc_score(query_names, gt_names)
        assert scores["rank"].to_list() == candidates["rank"].to_list()
        np.testing.assert_allclose(scores["score"].to_numpy(), candidates["score"].to_numpy())


def test_polars_sni_mapping_func_and_invalid_window():
    with pytest.raises(ValueError, match="SNI window should be odd integer"):
        PolarsSortedNeighbourhoodIndexer("name", window_length=4)

    ground_truth = pl.DataFrame({"name": ["abc", "cba", "bbb", "ddd"]})
    indexer = PolarsSortedNeighbourhoodIndexer("name", window_length=3, mapping_func=lambda value: value[::-1])
    indexer.fit(ground_truth)
    candidates = indexer.transform(pl.DataFrame({"name": ["xxc"]}))

    actual = [
        [ground_truth["name"][gt_uid], rank]
        for gt_uid, rank in zip(candidates["gt_uid"].to_list(), candidates["rank"].to_list())
    ]
    assert actual == [["abc", -1], ["ddd", 1]]


@pytest.mark.parametrize(
    ("blocking_func", "expected"), [(None, {0: {0, 2, 3}, 1: {0, 2, 3}, 2: {1}}), ("first", {0: {0, 3}, 1: {2}})]
)
def test_polars_cos_sim_indexer(blocking_func, expected):
    ground_truth = pl.DataFrame({"name": ["Tzu Sun", "Mikolaj Kopernik", "A", "Tzu"]})
    queries = pl.DataFrame({"name": ["Tzu Sun A", "A Tzu Sun", "Kopernik"]})
    indexer = PolarsCosSimIndexer(
        input_col="name", cos_sim_lower_bound=0.01, num_candidates=10, blocking_func=blocking_func
    )
    indexer.fit(ground_truth)
    candidates = indexer.transform(queries)
    grouped = candidates.group_by("uid", maintain_order=True).agg(pl.col("gt_uid"))
    actual = dict(zip(grouped["uid"].to_list(), [set(values) for values in grouped["gt_uid"].to_list()]))
    assert actual == expected


def test_polars_candidate_selection_carries_columns_and_adds_unmatched_rows():
    class NoMatchIndexer:
        def fit(self, X, y=None):
            return self

        def transform(self, X, multiple_indexers=False):
            return pl.DataFrame(schema={"uid": pl.Int64, "gt_uid": pl.Int64, "score": pl.Float32, "rank": pl.Int64})

        def column_prefix(self):
            return "none"

        def increase_window_by_one_step(self):
            pass

        def decrease_window_by_one_step(self):
            pass

    ground_truth = pl.DataFrame({"name": ["a", "b"], "entity_id": [10, 20], "extra": ["x", "y"]})
    queries = pl.DataFrame({"name": ["z", "q"], "entity_id": [30, 40], "query_extra": ["m", "n"]})
    selector = PolarsCandidateSelectionTransformer(
        [NoMatchIndexer()], uid_col="entity_id", carry_on_cols=["extra", "query_extra"], with_no_matches=True
    ).fit(ground_truth)

    candidates = selector.transform(queries)

    assert candidates["uid"].to_list() == [0, 1]
    assert candidates["gt_uid"].to_list() == [None, None]
    assert candidates["gt_entity_id"].to_list() == [None, None]
    assert candidates["entity_id"].to_list() == [30, 40]
    assert candidates["gt_extra"].to_list() == [None, None]
    assert candidates["query_extra"].to_list() == ["m", "n"]


def test_polars_candidate_selection_carry_on_cols():
    ground_truth = pl.DataFrame(
        {
            "preprocessed": ["apple", "microsoft", "google", "amazon", "netflix", "spotify"],
            "id": [1, 2, 3, 4, 5, 6],
            "a": [1, 1, 1, 1, 1, 1],
        }
    )
    queries = pl.DataFrame(
        {"preprocessed": ["aplle", "microbloft", "googol", "amz", "netfliks", "spot-off"], "b": [2, 2, 2, 2, 2, 2]}
    )
    selector = PolarsCandidateSelectionTransformer(
        [PolarsSortedNeighbourhoodIndexer("preprocessed", window_length=3)], uid_col="id", carry_on_cols=["a", "b"]
    ).fit(ground_truth)

    candidates = selector.transform(queries)

    assert "gt_a" in candidates.columns
    assert "b" in candidates.columns
    assert candidates["gt_a"].to_list() == [1] * 8
    assert candidates["b"].to_list() == [2] * 8
    assert candidates.height == 8


def test_polars_candidate_selection_merges_multiple_indexers():
    class FixedIndexer:
        def __init__(self, rows: dict[str, list], prefix: str) -> None:
            self.rows = rows
            self.prefix = prefix

        def fit(self, X, y=None):
            return self

        def transform(self, X, multiple_indexers=False):
            return pl.DataFrame(
                self.rows, schema={"uid": pl.Int64, "gt_uid": pl.Int64, "score": pl.Float32, "rank": pl.Int64}
            )

        def column_prefix(self):
            return self.prefix

        def increase_window_by_one_step(self):
            pass

        def decrease_window_by_one_step(self):
            pass

    ground_truth = pl.DataFrame({"name": ["a", "b"]})
    queries = pl.DataFrame({"name": ["x", "y"]})
    selector = PolarsCandidateSelectionTransformer(
        [
            FixedIndexer({"uid": [0, 1], "gt_uid": [0, 0], "score": [0.8, 0.3], "rank": [1, 2]}, "first"),
            FixedIndexer({"uid": [0], "gt_uid": [1], "score": [0.6], "rank": [1]}, "second"),
        ],
        with_no_matches=False,
    ).fit(ground_truth)

    candidates = selector.transform(queries)
    assert set(zip(candidates["uid"].to_list(), candidates["gt_uid"].to_list())) == {(0, 0), (0, 1), (1, 0)}
    assert list(zip(candidates["gt_uid"].to_list(), candidates["uid"].to_list())) == [(0, 0), (0, 1), (1, 0)]
    by_pair = {
        (uid, gt_uid): (first, second)
        for uid, gt_uid, first, second in zip(
            candidates["uid"].to_list(),
            candidates["gt_uid"].to_list(),
            candidates["first"].to_list(),
            candidates["second"].to_list(),
        )
    }
    assert by_pair == {(0, 0): (1, 0), (0, 1): (0, 1), (1, 0): (1, 0)}
