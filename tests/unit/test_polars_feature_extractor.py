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

import numpy as np
import pandas as pd
import pytest

from tests.unit import test_feature_extractor as pandas_tests

pl = pytest.importorskip("polars")

Vocabulary = import_module("emm.features.features_vocabulary").Vocabulary
PandasFeatureExtractor = import_module("emm.features.pandas_feature_extractor").PandasFeatureExtractor
PolarsFeatureExtractor = import_module("emm.features.polars_feature_extractor").PolarsFeatureExtractor
assert_polars_frame_equal = import_module("polars.testing").assert_frame_equal
pandas_kvk_fixture = pandas_tests.kvk_candidate_pairs


@pytest.fixture()
def candidate_pairs():
    return pl.DataFrame(
        {
            "name": ["rare foo company", "rare foo company", "rare foo company", "other ltd", "no candidate", ""],
            "gt_name": ["rare bar company ltd", "unrelated company", "rare limited", "other limited", "", ""],
            "score": [1.0, 0.1, 0.9, 0.95, None, None],
            "uid": [0, 0, 0, 1, 2, 3],
            "country": ["PL", "PL", "PL", None, "NL", "NL"],
            "gt_country": ["PL", "NL", "PL", "NL", None, None],
        }
    )


@pytest.mark.parametrize("without_rank_features", [False, True])
@pytest.mark.parametrize("with_legal_entity_forms_match", [False, True])
def test_polars_features_match_pandas(candidate_pairs, without_rank_features, with_legal_entity_forms_match):
    kwargs = {
        "name1_col": "name",
        "name2_col": "gt_name",
        "score_columns": ["score"],
        "extra_features": ["country"],
        "vocabulary": Vocabulary(very_common_words={"ltd"}, common_words=set()),
        "without_rank_features": without_rank_features,
        "with_legal_entity_forms_match": with_legal_entity_forms_match,
    }
    original = candidate_pairs.clone()
    actual = PolarsFeatureExtractor(**kwargs).transform(candidate_pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(candidate_pairs.to_pandas())

    pd.testing.assert_frame_equal(actual.to_pandas(), expected, check_categorical=False)
    assert candidate_pairs.equals(original)
    assert actual.height == candidate_pairs.height
    assert actual["country"].to_list() == [1, -1, 1, 0, 0, 0]
    assert actual["abs_len_diff"][0] == 4
    if not without_rank_features:
        assert actual["score_rank"].to_list() == [1, 3, 2, 1, -1, -1]
        np.testing.assert_allclose(actual["score_dist_to_max"].to_numpy(), [0, 0.9, 0.1, 0, -1, -1], atol=1e-6)
        np.testing.assert_allclose(actual["score_dist_to_min"].to_numpy(), [0.9, 0, 0.8, 0, -1, -1], atol=1e-6)
        np.testing.assert_allclose(actual["score_ptp"].to_numpy(), [0.9, 0.9, 0.9, 0, -1, -1], atol=1e-6)
        np.testing.assert_allclose(actual["score_diff_to_prev"].to_numpy(), [0.1, -1, 0.8, -1, -1, -1], atol=1e-6)
        np.testing.assert_allclose(actual["score_diff_to_next"].to_numpy(), [-1, 0.8, 0.1, -1, -1, -1], atol=1e-6)
    if with_legal_entity_forms_match:
        assert actual["match_legal_entity_form"].to_list() == [
            "no_match",
            "identical",
            "no_match",
            "no_match",
            "lef1_lef2_missing",
            "lef1_lef2_missing",
        ]


def test_polars_feature_vocabulary_fit_and_row_order(candidate_pairs):
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "score_columns": ["score"]}
    polars_extractor = PolarsFeatureExtractor(**kwargs)
    pandas_extractor = PandasFeatureExtractor(**kwargs)
    assert polars_extractor.fit(candidate_pairs) is polars_extractor
    pandas_extractor.fit(candidate_pairs.to_pandas())
    assert polars_extractor.vocabulary == pandas_extractor.vocabulary

    shuffled = candidate_pairs[[3, 1, 4, 0, 5, 2]]
    actual = polars_extractor.transform(shuffled)
    expected = pandas_extractor.transform(shuffled.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)


def test_polars_feature_fit_transform_and_extra_callable(candidate_pairs):
    data = candidate_pairs.with_columns(pl.Series("amount", [1, 10, 20, 2, 3, 4])).with_columns(
        pl.Series("gt_amount", [100, 50, 0, 5, 6, 7])
    )
    kwargs = {
        "name1_col": "name",
        "name2_col": "gt_name",
        "score_columns": ["score"],
        "extra_features": [("amount", lambda x, y: x + y)],
        "fillna_value": 0.0,
        "drop_features": ["abbr_match"],
    }
    actual = PolarsFeatureExtractor(**kwargs).fit_transform(data)
    expected = PandasFeatureExtractor(**kwargs).fit_transform(data.to_pandas())

    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
    assert actual["amount"].to_list() == [101, 60, 20, 7, 9, 11]
    assert "abbr_match" not in actual.columns
    assert np.isfinite(actual["score"].to_numpy()).all()


def test_polars_extra_features_pandas_cases():
    pairs = pl.DataFrame(
        {
            "name": ["a"] * 6,
            "gt_name": ["a", "b", "c", "d", "e", "f"],
            "uid": [0, 1, 2, 3, 4, 5],
            "country": ["PL", "PL", "PL", None, None, None],
            "gt_country": ["PL", "NL", None, "PL", None, None],
        }
    )
    extractor = PolarsFeatureExtractor(name1_col="name", name2_col="gt_name", extra_features=["country"])
    assert extractor.transform(pairs)["country"].to_list() == [1, -1, 0, 0, 0, 0]


def test_polars_feature_errors(candidate_pairs):
    extractor = PolarsFeatureExtractor(name1_col="name", name2_col="gt_name", extra_features=["missing"])
    with pytest.raises(ValueError, match="missing extra features columns"):
        extractor.transform(candidate_pairs)
    with pytest.raises(TypeError, match="expected a Polars DataFrame"):
        extractor.transform(candidate_pairs.to_pandas())


def test_polars_feature_values_for_name_and_vocabulary(candidate_pairs):
    extractor = PolarsFeatureExtractor(
        name1_col="name",
        name2_col="gt_name",
        without_rank_features=True,
        vocabulary=Vocabulary(very_common_words={"ltd"}, common_words={"company"}),
    )
    result = extractor.transform(candidate_pairs.head(1)).row(0, named=True)

    assert result["abbr_match"] == 0
    assert result["abs_len_diff"] == 4
    assert result["len_ratio"] == pytest.approx(0.8)
    assert result["token_sort_ratio"] == 72
    assert result["token_set_ratio"] == 85
    assert result["partial_ratio"] == 81
    assert result["w_ratio"] == 81
    assert result["ratio"] == 72
    assert result["name_cut"] == 0
    assert result["norm_ed"] == 7
    assert result["norm_jaro"] == pytest.approx(0.7695513)
    assert [result[key] for key in ("very_common_hit", "common_hit", "rare_hit")] == [0, 1, 1]
    assert [result[key] for key in ("very_common_miss", "common_miss", "rare_miss")] == [1, 0, 2]
    assert result["n_overlap_words"] == 2
    assert result["ratio_overlap_words"] == pytest.approx(0.4)
    assert result["num_word_difference"] == 1


@pytest.mark.parametrize(("func_name", "name1", "name2", "expected_value"), pandas_tests.NAME_FEATURE_CASES)
def test_polars_name_features_functions(func_name, name1, name2, expected_value):
    pairs = pl.DataFrame({"name": [name1], "gt_name": [name2], "uid": [0]})
    extractor = PolarsFeatureExtractor(
        name1_col="name", name2_col="gt_name", vocabulary=Vocabulary(very_common_words=set(), common_words=set())
    )
    result = extractor.transform(pairs)
    _, dtype = extractor.name_features[func_name]
    assert result.schema[func_name] == (pl.Int8 if dtype == "int8" else pl.Float32)
    if dtype == "float32":
        assert result[func_name][0] == pytest.approx(expected_value)
    else:
        assert result[func_name][0] == expected_value


def test_polars_hits_features(candidate_pairs):
    extractor = PolarsFeatureExtractor(
        name1_col="name", name2_col="gt_name", vocabulary=Vocabulary(very_common_words={"ltd"}, common_words=set())
    )
    result = extractor.transform(candidate_pairs)
    columns = [
        "very_common_hit",
        "common_hit",
        "rare_hit",
        "very_common_miss",
        "common_miss",
        "rare_miss",
        "n_overlap_words",
        "ratio_overlap_words",
        "num_word_difference",
    ]
    assert result.select(columns).columns == columns
    assert result["very_common_miss"].to_list() == [1, 0, 0, 1, 0, 0]
    assert all(result.schema[column] == pl.Float32 for column in columns)


@pytest.mark.parametrize(
    ("without_rank_features", "extra_features"), [(False, ["country"]), (True, ["country"]), (True, [])]
)
def test_polars_calc_features_three_scores(candidate_pairs, without_rank_features, extra_features):
    rng = np.random.default_rng(1)
    pairs = candidate_pairs.with_columns(
        (1 - pl.col("score")).alias("other_score"), pl.Series("random_score", rng.random(candidate_pairs.height))
    )
    score_columns = ["score", "other_score", "random_score"]
    kwargs = {
        "name1_col": "name",
        "name2_col": "gt_name",
        "uid_col": "uid",
        "score_columns": score_columns,
        "extra_features": extra_features,
        "without_rank_features": without_rank_features,
        "vocabulary": Vocabulary(very_common_words={"ltd"}, common_words=set()),
    }
    result = PolarsFeatureExtractor(**kwargs).transform(pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(result.to_pandas(), expected)
    assert result.width == 20 + len(extra_features) + (1 if without_rank_features else 8) * len(score_columns)
    if not without_rank_features:
        assert {
            "score",
            "abbr_match",
            "ratio",
            "very_common_hit",
            "score_rank",
            "score_diff_to_next",
            "country",
        } <= set(result.columns)


def test_polars_feature_multiple_scores_and_ties():
    pairs = pl.DataFrame(
        {
            "name": ["ABC1"] * 4,
            "gt_name": ["GT1", "GT2", "GT3", "GT4"],
            "score": [1.0, 0.9, 0.1, 0.1],
            "other_score": [0.0, 0.1, 0.9, 0.9],
            "uid": [0, 0, 0, 0],
            "gt_uid": [1, 2, 3, 4],
        }
    )
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "score_columns": ["score", "other_score"]}
    actual = PolarsFeatureExtractor(**kwargs).transform(pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(pairs.to_pandas())

    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
    assert len(actual.columns) == 20 + 8 * 2
    assert actual["score_rank"].to_list() == [1, 2, 3, 4]
    assert actual["other_score_rank"].to_list() == [4, 3, 1, 2]
    assert actual["score_top2_dist"].to_list() == pytest.approx([0.1] * 4)
    assert actual["score_diff_to_next"].to_list() == pytest.approx([-1, 0.1, 0.8, 0])
    assert actual["score_diff_to_prev"].to_list() == pytest.approx([0.1, 0.8, 0, -1])


def test_polars_feature_stability_after_shuffling(candidate_pairs):
    extractor = PolarsFeatureExtractor(name1_col="name", name2_col="gt_name", score_columns=["score"])
    expected = extractor.transform(candidate_pairs)
    order = [3, 1, 4, 0, 5, 2]
    actual = extractor.transform(candidate_pairs[order])

    pd.testing.assert_frame_equal(actual.to_pandas(), expected[order].to_pandas())


def test_polars_feature_stability_with_ties_and_score_noise(pandas_kvk_fixture):
    rng = np.random.default_rng(1)
    extractor = PolarsFeatureExtractor(name1_col="name", name2_col="gt_name", uid_col="uid", score_columns=["score"])
    pairs = pl.from_pandas(pandas_kvk_fixture)
    expected = extractor.transform(pairs).with_columns(pairs["i"]).sort("i").drop("i")
    for seed in range(10):
        shuffled = pl.from_pandas(pandas_kvk_fixture.sample(frac=1, random_state=seed).reset_index(drop=True))
        shuffled = shuffled.with_columns(
            (pl.col("score") + pl.Series(rng.random(shuffled.height) * 1e-7)).alias("score")
        )
        actual = extractor.transform(shuffled).with_columns(shuffled["i"]).sort("i").drop("i")
        assert_polars_frame_equal(actual, expected, rel_tol=0, abs_tol=1e-3)


def test_polars_features_do_not_convert_to_pandas(candidate_pairs, monkeypatch):
    def fail_conversion(*args, **kwargs):
        msg = "Polars feature extraction must not convert to pandas"
        raise AssertionError(msg)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", fail_conversion)
    extractor = PolarsFeatureExtractor(name1_col="name", name2_col="gt_name", score_columns=["score"])
    result = extractor.fit(candidate_pairs).transform(candidate_pairs)
    assert result.height == candidate_pairs.height


def test_polars_feature_rounded_score_ties():
    pairs = pl.DataFrame(
        {"name": ["a", "a", "a"], "gt_name": ["x", "y", "z"], "uid": [0, 0, 0], "score": [0.500001, 0.500004, 0.4]}
    )
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "score_columns": ["score"]}
    actual = PolarsFeatureExtractor(**kwargs).transform(pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)


def test_polars_rank_features_mixed_null_scores():
    pairs = pl.DataFrame(
        {
            "name": ["a"] * 5,
            "gt_name": ["a", "b", "c", "d", "e"],
            "uid": [0, 0, 0, 1, 1],
            "score": [0.8, 0.3, None, 0.5, None],
        }
    )
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "score_columns": ["score"]}
    actual = PolarsFeatureExtractor(**kwargs).transform(pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
    assert actual["score_ptp"].to_list() == [-1] * 5
    assert actual["score_top2_dist"].to_list() == pytest.approx([0.5, 0.5, 0.5, -1, -1])


def test_polars_extra_features_with_fillna(candidate_pairs):
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "extra_features": ["country"], "fillna_value": 0.0}
    actual = PolarsFeatureExtractor(**kwargs).transform(candidate_pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(candidate_pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)


@pytest.mark.parametrize("fillna_value", [None, 0.0])
def test_polars_extra_features_both_missing(fillna_value):
    pairs = pl.DataFrame(
        {
            "name": ["a", "b", "c"],
            "gt_name": ["a", "b", "c"],
            "uid": [0, 1, 2],
            "country": [None, None, "NL"],
            "gt_country": [None, "NL", None],
        }
    )
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "extra_features": ["country"], "fillna_value": fillna_value}
    actual = PolarsFeatureExtractor(**kwargs).transform(pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)


def test_polars_rank_features_with_missing_uid():
    pairs = pl.DataFrame(
        {
            "name": ["a", "b", "c"],
            "gt_name": ["a", "b", "c"],
            "uid": [1.0, float("nan"), None],
            "score": [0.1, 0.2, 0.3],
        }
    )
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "score_columns": ["score"]}
    actual = PolarsFeatureExtractor(**kwargs).transform(pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)


def test_polars_rank_features_large_group():
    pairs = pl.DataFrame(
        {"name": ["a"] * 130, "gt_name": [f"{i:03d}" for i in range(130)], "uid": [0] * 130, "score": [1.0] * 130}
    )
    kwargs = {"name1_col": "name", "name2_col": "gt_name", "score_columns": ["score"]}
    actual = PolarsFeatureExtractor(**kwargs).transform(pairs)
    expected = PandasFeatureExtractor(**kwargs).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
