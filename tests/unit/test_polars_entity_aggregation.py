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

from importlib import import_module

import pandas as pd
import pytest

from tests.unit import test_entity_aggregation as pandas_tests

pl = pytest.importorskip("polars")

PolarsEntityAggregation = import_module("emm.aggregation.polars_entity_aggregation").PolarsEntityAggregation
matching_max_candidate = import_module("emm.aggregation.polars_entity_aggregation").matching_max_candidate
PandasEntityAggregation = pandas_tests.PandasEntityAggregation
sample_one_cluster_candidates = pandas_tests.sample_one_cluster_candidates
sample_two_cluster_candidates = pandas_tests.sample_two_cluster_candidates
sample_candidates = pandas_tests.sample_candidates


@pytest.mark.parametrize(
    ("fixture_name", "expected_uid", "expected_score"),
    [("sample_one_cluster_candidates", 1001, 0.533333), ("sample_two_cluster_candidates", 1001, 0.55)],
)
def test_matching_max_frequency_pandas_cases(request, fixture_name, expected_uid, expected_score):
    pairs = request.getfixturevalue(fixture_name)
    kwargs = {
        "group": ["gt_uid", "account"],
        "score_col": "nm_score",
        "name_col": "name",
        "account_col": "account",
        "freq_col": "counterparty_account_count_distinct",
        "output_col": "agg_score",
        "aggregation_method": "max_frequency_nm_score",
    }
    actual = matching_max_candidate(pl.from_pandas(pairs), **kwargs)
    expected = pandas_tests.matching_max_candidate(pairs, **kwargs)
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert set(actual.columns) == {*pairs.columns, "agg_score"}
    assert actual["gt_uid"][0] == expected_uid
    assert actual["agg_score"][0] == pytest.approx(expected_score)


def test_matching_max_frequency_missing_candidate():
    pairs = pl.DataFrame(
        {
            "uid": [1000],
            "name": ["Tzu Sun"],
            "entity_id": [1],
            "account": ["G0001"],
            "gt_entity_id": [None],
            "gt_uid": [None],
            "nm_score": [0.0019],
            "counterparty_account_count_distinct": [1],
        }
    )
    kwargs = {
        "group": ["gt_uid", "account"],
        "score_col": "nm_score",
        "name_col": "name",
        "account_col": "account",
        "freq_col": "counterparty_account_count_distinct",
        "output_col": "agg_score",
    }
    actual = matching_max_candidate(pairs, **kwargs)
    expected = pandas_tests.matching_max_candidate(pairs.to_pandas(), **kwargs)
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
    assert actual["agg_score"][0] == pytest.approx(0.0019)


@pytest.mark.parametrize("aggregation_method", ["max_frequency_nm_score", "mean_score"])
def test_entity_aggregation_pandas_cases(sample_candidates, aggregation_method):
    kwargs = {"score_col": "nm_score", "aggregation_method": aggregation_method}
    pairs = pl.from_pandas(sample_candidates)
    original = pairs.clone()
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(sample_candidates.copy())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert pairs.equals(original)
    assert set(actual["account"].to_list()) == set(pairs["account"].to_list())
    assert actual["best_match"].to_list() == [True, True]
    assert actual["best_rank"].to_list() == [1, 1]


@pytest.fixture
def mixed_candidates():
    return pl.DataFrame(
        {
            "account": ["z", "a", "a", "a", "b", "b", "c", "d", None],
            "gt_uid": [10, 11, 11, 12, 14, 13, 15, None, 17],
            "gt_entity_id": [10, 11, 11, 12, 14, 13, 15, None, 17],
            "name": ["single", "first", "second", "second", "same", "same", "bad", "none", "null"],
            "preprocessed": ["single", "first", "second", "second", "same", "same", "bad", "none", "null"],
            "score": [0.4, 0.7, 0.3, 0.9, 0.5, 0.5, 0.1, 0.1, 0.6],
            "freq": [2, 5, 1, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.mark.parametrize("aggregation_method", ["max_frequency_nm_score", "mean_score"])
def test_mixed_groups_match_pandas(mixed_candidates, aggregation_method):
    kwargs = {"score_col": "score", "freq_col": "freq", "aggregation_method": aggregation_method, "blacklist": ["bad"]}
    actual = PolarsEntityAggregation(**kwargs).transform(mixed_candidates)
    expected = PandasEntityAggregation(**kwargs).transform(mixed_candidates.to_pandas())
    pd.testing.assert_frame_equal(
        actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
    assert actual.schema["gt_uid"] == mixed_candidates.schema["gt_uid"]
    assert actual["account"].to_list() == ["z", "c", "a", "b"]
    assert actual.filter(pl.col("account") == "a")["gt_uid"][0] == (
        11 if aggregation_method == "max_frequency_nm_score" else 12
    )


def test_blacklist_only_applies_to_multiple_matches(mixed_candidates):
    pairs = pl.concat(
        [
            mixed_candidates,
            pl.DataFrame(
                {
                    "account": ["e", "e"],
                    "gt_uid": [20, 21],
                    "gt_entity_id": [20, 21],
                    "name": ["bad", "good"],
                    "preprocessed": ["bad", "good"],
                    "score": [0.9, 0.2],
                    "freq": [1, 1],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    kwargs = {"score_col": "score", "freq_col": "freq", "blacklist": ["bad"]}
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(
        actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
    assert actual.filter(pl.col("account") == "c")["name"].to_list() == ["bad"]
    assert actual.filter(pl.col("account") == "e")["name"].to_list() == ["good"]


def test_null_string_names_and_preprocessed_values():
    pairs = pl.DataFrame(
        {
            "account": ["a", "a"],
            "gt_uid": [1, 1],
            "gt_entity_id": [1, 1],
            "name": pl.Series([None, None], dtype=pl.String),
            "preprocessed": pl.Series([None, None], dtype=pl.String),
            "score": [0.3, 0.5],
            "freq": [1, 1],
        }
    )
    kwargs = {"score_col": "score", "freq_col": "freq"}
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert actual["agg_score"][0] == pytest.approx(0.4)


def test_entity_and_positive_set_grouping():
    pairs = pl.DataFrame(
        {
            "account": ["a", "a", "a", "a"],
            "entity_id": [1, 1, 1, 2],
            "positive_set": [True, True, False, True],
            "gt_uid": [1, 2, 1, 1],
            "gt_entity_id": [None, None, 1, 1],
            "name": ["first", "second", "other", "last"],
            "score": [0.8, 0.4, 0.3, 0.5],
            "freq": [1, 1, 1, 1],
        }
    )
    kwargs = {"score_col": "score", "freq_col": "freq"}
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(
        actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
    assert actual["entity_id"].to_list() == [1, 2, 1]
    assert actual["positive_set"].to_list() == [False, True, True]


def test_zero_frequency_and_null_ground_truth_group():
    pairs = pl.DataFrame(
        {
            "account": ["a", "a"],
            "gt_uid": [1, 1],
            "gt_entity_id": [None, None],
            "name": ["first", "second"],
            "score": [0.3, 0.5],
            "freq": [0, 0],
        }
    )
    kwargs = {"score_col": "score", "freq_col": "freq"}
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert actual["agg_score"].is_nan().to_list() == [True]


@pytest.mark.parametrize("with_missing_score", [False, True])
def test_mean_score_float32_rounding_selects_same_candidate_as_pandas(with_missing_score):
    gt_uids = [1, 1, 1, 2] + ([3] if with_missing_score else [])
    pairs = pl.DataFrame(
        {
            "account": ["a"] * len(gt_uids),
            "gt_uid": gt_uids,
            "gt_entity_id": gt_uids,
            "name": ["first", "second", "third", "fourth"] + (["fifth"] if with_missing_score else []),
            "score": pl.Series([0.9] * 4 + ([None] if with_missing_score else []), dtype=pl.Float32),
            "freq": [1] * len(gt_uids),
        }
    )
    kwargs = {"score_col": "score", "freq_col": "freq", "aggregation_method": "mean_score"}
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert actual["gt_uid"].to_list() == [2 if with_missing_score else 1]


@pytest.mark.parametrize("first_entity", [2, None])
def test_mean_score_float32_unsorted_groups_select_same_candidate_as_pandas(first_entity):
    pairs = pl.DataFrame(
        {
            "account": ["a"] * 4,
            "gt_uid": [1, 1, 1, 2],
            "gt_entity_id": [first_entity, first_entity, first_entity, 1],
            "name": ["first", "second", "third", "fourth"],
            "score": pl.Series([0.9] * 4, dtype=pl.Float32),
            "freq": [1] * 4,
        }
    )
    kwargs = {"score_col": "score", "freq_col": "freq", "aggregation_method": "mean_score"}
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(
        actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
    assert actual["gt_uid"].to_list() == [2]


@pytest.mark.parametrize("aggregation_method", ["max_frequency_nm_score", "mean_score"])
def test_null_scores_and_group_ties(aggregation_method):
    pairs = pl.DataFrame(
        {
            "account": ["a", "a", "b", "b"],
            "gt_uid": [2, 1, 3, 4],
            "gt_entity_id": [2, 1, 3, 4],
            "name": ["first", "second", "other", "other2"],
            "score": [0.5, 0.5, float("nan"), None],
            "freq": [1, 1, 1, 1],
        }
    )
    kwargs = {"score_col": "score", "freq_col": "freq", "aggregation_method": aggregation_method}
    actual = PolarsEntityAggregation(**kwargs).transform(pairs)
    expected = PandasEntityAggregation(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert actual.filter(pl.col("account") == "a")["gt_uid"][0] == (1 if aggregation_method != "mean_score" else 2)


@pytest.mark.parametrize("aggregation_method", ["max_frequency_nm_score", "mean_score"])
def test_custom_columns_and_output(aggregation_method):
    pairs = pl.DataFrame(
        {
            "group": ["a", "a", "b"],
            "entity": [1, 1, 2],
            "reference": [1, 1, 2],
            "reference_entity": [3, 3, 4],
            "input_name": ["first", "second", "only"],
            "raw_score": [0.8, 0.6, 0.4],
            "frequency": [3, 1, 1],
        }
    )
    aggregation = PolarsEntityAggregation(
        score_col="raw_score",
        account_col="group",
        index_col="entity",
        gt_uid_col="reference",
        gt_entity_id_col="reference_entity",
        name_col="input_name",
        freq_col="frequency",
        output_col="confidence",
        aggregation_method=aggregation_method,
    )
    result = aggregation.transform(pairs)
    assert result["group"].to_list() == ["b", "a"]
    assert result["confidence"].to_list() == pytest.approx([0.4, 0.75 if aggregation_method != "mean_score" else 0.7])
    assert "agg_score" not in result.columns


def test_empty_and_blacklisted_inputs(mixed_candidates):
    aggregator = PolarsEntityAggregation(score_col="score", freq_col="freq", blacklist=["first", "second"])
    assert aggregator.fit(mixed_candidates) is aggregator
    assert aggregator.transform(None) is None
    assert aggregator.fit_transform(mixed_candidates).equals(aggregator.transform(mixed_candidates))
    empty = aggregator.transform(mixed_candidates.head(0))
    assert empty.is_empty()
    assert {"agg_score", "freq_score", "best_match", "best_rank"} <= set(empty.columns)
    assert aggregator.transform(mixed_candidates.filter(pl.col("account") == "a")).is_empty()
    with pytest.raises(TypeError, match="expected a Polars DataFrame"):
        aggregator.transform(mixed_candidates.to_pandas())


def test_no_pandas_conversion(mixed_candidates, monkeypatch):
    def fail_conversion(*args, **kwargs):
        msg = "Polars aggregation must not convert to pandas"
        raise AssertionError(msg)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", fail_conversion)
    result = PolarsEntityAggregation(score_col="score", freq_col="freq").transform(mixed_candidates)
    assert result.height == 4


def test_unsupported_method_and_empty_group():
    pairs = pl.DataFrame({"score": [1.0]})
    kwargs = {
        "group": [],
        "score_col": "score",
        "name_col": "name",
        "account_col": "account",
        "freq_col": "freq",
        "output_col": "agg_score",
    }
    with pytest.raises(ValueError, match="empty df"):
        matching_max_candidate(pairs.head(0), **kwargs)
    with pytest.raises(ValueError, match="aggregation_method not supported"):
        matching_max_candidate(pairs, **kwargs, aggregation_method="unsupported")
