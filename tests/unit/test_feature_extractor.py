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

import re

import numpy as np
import pandas as pd
import pytest

from emm.data.create_data import retrieve_kvk_test_sample
from emm.features.features_extra import calc_extra_features
from emm.features.features_lef import calc_lef_features
from emm.features.features_name import calc_name_features
from emm.features.features_rank import calc_diff_features, calc_rank_features
from emm.features.features_vocabulary import Vocabulary, compute_vocabulary_features
from emm.features.pandas_feature_extractor import PandasFeatureExtractor


@pytest.fixture()
def single_candidate_pair():
    return pd.DataFrame({"name": ["rare foo company"], "gt_name": ["rare bar company ltd"], "score": [1.0]})


@pytest.fixture()
def candidate_pairs():
    return pd.DataFrame(
        {
            "name": ["rare foo company", "rare foo company", "rare foo company", "other ltd", "no candidate", ""],
            "gt_name": ["rare bar company ltd", "unrelated company", "rare limited", "other limited", "", ""],
            "score": [1.0, 0.1, 0.9, 0.95, None, None],
            "uid": [0, 0, 0, 1, 2, 3],
            "country": ["PL", "PL", "PL", None, "NL", "NL"],
            "gt_country": ["PL", "NL", "PL", "NL", None, None],
        }
    )


@pytest.fixture()
def kvk_candidate_pairs(n=1000):
    _, df = retrieve_kvk_test_sample()
    all_names = df["Name"].values
    df["id"] = range(len(df))
    rng = np.random.default_rng(1)
    names = rng.choice(all_names, n)
    gt_names = rng.choice(all_names, n)
    scores = rng.random(n)
    # introduce ties
    scores[rng.choice(range(n), n // 10)] = 0.123
    return pd.DataFrame(
        {"i": range(n), "uid": [i // 10 for i in range(n)], "name": names, "gt_name": gt_names, "score": scores}
    )


def test_calc_name_features(single_candidate_pair):
    fe = PandasFeatureExtractor()
    res = calc_name_features(single_candidate_pair, funcs=fe.name_features, name1="name", name2="gt_name")
    assert len(res) == 1
    pd.testing.assert_series_equal(
        res.iloc[0],
        pd.Series(
            {
                "abbr_match": 0.0,
                "abs_len_diff": 4.0,
                "len_ratio": 0.8,
                # Rapidfuzz
                "token_sort_ratio": 72.0,
                "token_set_ratio": 85.0,  # 86 with fuzzywuzzy
                "partial_ratio": 81.0,
                "w_ratio": 81.0,
                "ratio": 72.0,
                "name_cut": 0.0,
                "norm_ed": 7.0,
                "norm_jaro": 0.7695513,
            },
            dtype="float32",
        ),
        check_names=False,
    )


def test_extra_features():
    df = pd.DataFrame(
        {"country": ["PL", "PL", "PL", None, None, pd.NA], "gt_country": ["PL", "NL", None, "PL", None, pd.NA]}
    )
    res = calc_extra_features(df, features=["country"])
    pd.testing.assert_series_equal(res["country"], pd.Series([1, -1, 0, 0, 0, 0]), check_names=False)

    df2 = pd.DataFrame({"v": [1, 10, 20], "gt_v": [100, 50, 0]})
    res2 = calc_extra_features(df2, features=[("v", lambda x, y: x + y)])
    pd.testing.assert_series_equal(res2["v"], pd.Series([101, 60, 20]), check_names=False)


@pytest.mark.parametrize(
    ("func_name", "name1", "name2", "expected_value"),
    [
        # warning! this is case-sensitive
        ("abbr_match", "Abcd", "Axyz", False),
        ("abbr_match", "ABC xyz", "Aaa Bbb Ccc", True),
        # if name is all lower case, approximate version is used
        ("abbr_match", "abc xyz", "aaa bbb ccc", True),
        ("abbr_match", "abc xyz", "aaa bbb xyz", False),
        ("abs_len_diff", "abc", "xyz", 0),
        ("abs_len_diff", "abc", "abcabc", 3),
        ("len_ratio", "abc", "xyz", 1.0),
        ("len_ratio", "abc", "xyzxyz", 0.5),
        ("len_ratio", "abcabcabc", "xyz", 1 / 3),
        ("name_cut", "abc", "xyz", False),
        ("name_cut", "abcabc", "abcxyz", False),
        ("name_cut", "abcxyz", "abc", True),
        ("name_cut", "abc", "abcxyz", True),
        # rapidfuzz.distance.Levenshtein.distance (regular edit distance)
        ("norm_ed", "abc", "abc", 0),
        ("norm_ed", "abc", "xyz", 3),
        ("norm_ed", "abc", "axbc", 1),
        ("norm_ed", "aybc", "axbc", 1),
        # rapidfuzz.distance.Jaro.similarity
        ("norm_jaro", "abc", "abc", 1),
        ("norm_jaro", "abc", "xyz", 0),
        ("norm_jaro", "abc", "axbc", 0.91666666),
        # fuzzywuzzy features
        ("w_ratio", "abc", "abc", 100),
        ("w_ratio", "abc", "xyz", 0),
        ("w_ratio", "abc", "axbc", 85),  # 86 with fuzzywuzzy
        ("ratio", "abc", "abc", 100),
        ("ratio", "abc", "xyz", 0),
        ("ratio", "abc", "axbc", 85),  # 86 with fuzzywuzzy
        ("token_sort_ratio", "abc bcd abc", "bcd abc abc", 100),
        ("token_set_ratio", "abc bcd abc", "abc abc xyz", 60),
        ("partial_ratio", "abc bcd abc", "abc abc xyz", 77),  # 64 with fuzzywuzzy
    ],
)
def test_name_features_functions(func_name, name1, name2, expected_value):
    fe = PandasFeatureExtractor()
    func, dtype = fe.name_features[func_name]
    value = func(name1, name2)
    value_casted = np.array([value]).astype(dtype)[0]  # Like in calc_name_features()

    if dtype == "int8":
        assert -128 <= value_casted < 128 if isinstance(value_casted, np.int8) else isinstance(value_casted, bool)
    elif dtype == "float32":
        assert isinstance(value_casted, np.float32)
    else:
        msg = f"Unsupported dtype={dtype}"
        raise Exception(msg)

    if isinstance(value_casted, np.float32):
        assert expected_value == pytest.approx(value_casted)
    else:
        assert expected_value == value_casted


def test_compute_hits_misses():
    data = compute_vocabulary_features(
        pd.DataFrame({"col1": ["rare foo company"], "col2": ["rare bar company ltd"]}),
        col1="col1",
        col2="col2",
        common_words={"company"},
        very_common_words={"ltd"},
    ).iloc[0]
    pd.testing.assert_series_equal(
        data,
        pd.Series(
            {
                "very_common_hit": 0.0,
                "common_hit": 1.0,
                "rare_hit": 1.0,
                "very_common_miss": 1.0,
                "common_miss": 0.0,
                "rare_miss": 2.0,
                "n_overlap_words": 2.0,
                "ratio_overlap_words": 0.4,
                "num_word_difference": 1.0,
            },
            name=0,
            dtype="float32",
        ),
    )


def test_calc_hits_features(candidate_pairs):
    res = compute_vocabulary_features(
        candidate_pairs, col1="name", col2="gt_name", very_common_words={"ltd"}, common_words=set()
    )
    assert all(res.index == candidate_pairs.index)
    assert res.columns.tolist() == [
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
    pd.testing.assert_series_equal(
        res["very_common_miss"], pd.Series([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype="float32"), check_names=False
    )


def test_calc_rank_features(candidate_pairs):
    fe = PandasFeatureExtractor()
    res = calc_rank_features(candidate_pairs, funcs=fe.rank_features, score_columns=["score"])
    assert all(res.index == candidate_pairs.index)
    assert res.columns.tolist() == [
        "score_rank",
        "score_top2_dist",
        "score_dist_to_max",
        "score_dist_to_min",
        "score_ptp",
    ]
    pd.testing.assert_series_equal(res["score_rank"], pd.Series([1, 3, 2, 1, -1, -1], dtype="int8"), check_names=False)
    pd.testing.assert_series_equal(
        res["score_dist_to_max"], pd.Series([0.0, 0.9, 0.1, 0.0, -1.0, -1.0], dtype="float32"), check_names=False
    )
    pd.testing.assert_series_equal(
        res["score_dist_to_min"], pd.Series([0.9, 0.0, 0.8, 0.0, -1.0, -1.0], dtype="float32"), check_names=False
    )
    pd.testing.assert_series_equal(
        res["score_ptp"], pd.Series([0.9, 0.9, 0.9, 0.0, -1.0, -1.0], dtype="float32"), check_names=False
    )


def test_calc_diff_features(candidate_pairs):
    na_value = -99.0
    fe = PandasFeatureExtractor()
    res = calc_diff_features(candidate_pairs, funcs=fe.diff_features, score_columns=["score"], fillna=na_value)
    assert all(res.index == candidate_pairs.index)
    assert res.columns.tolist() == ["score_diff_to_next", "score_diff_to_prev"]
    pd.testing.assert_series_equal(
        res["score_diff_to_prev"],
        pd.Series([0.1, na_value, 0.8, na_value, na_value, na_value], dtype="float32"),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        res["score_diff_to_next"],
        pd.Series([na_value, 0.8, 0.1, na_value, na_value, na_value], dtype="float32"),
        check_names=False,
    )


def test_calc_lef_features(candidate_pairs):
    res = calc_lef_features(candidate_pairs, name1="name", name2="gt_name", business_type=True, detailed_match=True)

    assert all(res.index == candidate_pairs.index)
    assert "match_legal_entity_form" in res.columns
    assert "match_business_type" in res.columns
    assert "legal_entity_forms" in res.columns
    assert "business_types" in res.columns

    lef_matches = res["match_legal_entity_form"].unique().tolist()
    bt_matches = res["match_business_type"].unique().tolist()

    np.testing.assert_array_equal(lef_matches, ["no_match", "identical", "lef1_lef2_missing"])
    np.testing.assert_array_equal(bt_matches, ["no_match", "identical", "lef1_lef2_missing"])


def test_calc_features(candidate_pairs):
    rng = np.random.default_rng(1)
    candidate_pairs["other_score"] = 1 - candidate_pairs["score"]
    candidate_pairs["random_score"] = rng.random(len(candidate_pairs))

    score_columns = ["score", "other_score", "random_score"]
    extra_features = ["country"]
    c = PandasFeatureExtractor(
        name1_col="name",
        name2_col="gt_name",
        uid_col="uid",
        score_columns=score_columns,
        extra_features=extra_features,
        vocabulary=Vocabulary(very_common_words={"ltd"}, common_words=set()),
    )

    res = c.transform(candidate_pairs)
    assert all(res.index == candidate_pairs.index)
    for col in ["score", "abbr_match", "ratio", "very_common_hit", "score_rank", "score_diff_to_next", "country"]:
        assert col in res.columns, f"missing column: {col}"
    assert len(res.columns) == 20 + len(extra_features) + 8 * len(score_columns)

    c = PandasFeatureExtractor(
        name1_col="name",
        name2_col="gt_name",
        uid_col="uid",
        score_columns=score_columns,
        extra_features=extra_features,
        vocabulary=Vocabulary(very_common_words={"ltd"}, common_words=set()),
        without_rank_features=True,
    )
    res_without_rank = c.transform(candidate_pairs)
    assert len(res_without_rank.columns) == 20 + len(extra_features) + 1 * len(score_columns)

    c2 = PandasFeatureExtractor(
        name1_col="name",
        name2_col="gt_name",
        uid_col="uid",
        score_columns=score_columns,
        # without extra features!
        without_rank_features=True,
    )
    res2 = c2.transform(candidate_pairs)
    assert len(res2.columns) == 20 + 1 * len(score_columns)


def test_calc_features_with_lef_match(candidate_pairs):
    score_columns = ["score"]
    c = PandasFeatureExtractor(
        name1_col="name",
        name2_col="gt_name",
        uid_col="uid",
        score_columns=score_columns,
        with_legal_entity_forms_match=True,
    )
    res = c.transform(candidate_pairs)

    assert all(res.index == candidate_pairs.index)
    assert "match_legal_entity_form" in res.columns

    lef_matches = res["match_legal_entity_form"].tolist()
    np.testing.assert_array_equal(
        lef_matches, ["no_match", "identical", "no_match", "no_match", "lef1_lef2_missing", "lef1_lef2_missing"]
    )


def test_rank_features(candidate_pairs):
    candidate_pairs["score_1"] = candidate_pairs["score"]
    candidate_pairs["score_2"] = 1 - candidate_pairs["score"]
    score_columns = ["score_1", "score_2"]
    c = PandasFeatureExtractor(name1_col="name", name2_col="gt_name", uid_col="uid", score_columns=score_columns)
    rank_features = {
        "score_1_diff_to_next",
        "score_1_diff_to_prev",
        "score_1_dist_to_max",
        "score_1_dist_to_min",
        "score_1_ptp",
        "score_1_rank",
        "score_1_top2_dist",
        "score_2_diff_to_next",
        "score_2_diff_to_prev",
        "score_2_dist_to_max",
        "score_2_dist_to_min",
        "score_2_ptp",
        "score_2_rank",
        "score_2_top2_dist",
    }
    res = c.transform(candidate_pairs)
    assert all(col in res.columns for col in rank_features)


def test_stability_of_features(kvk_candidate_pairs):
    """Double check if the values of the features does not depend on the ordering of the data"""
    rng = np.random.default_rng(1)

    c = PandasFeatureExtractor(name1_col="name", name2_col="gt_name", uid_col="uid", score_columns=["score"])
    feat = c.transform(kvk_candidate_pairs).set_index(kvk_candidate_pairs["i"].values).sort_index()
    for seed in range(10):
        # shuffle the input data
        curr_inp = kvk_candidate_pairs.copy()
        curr_inp = curr_inp.sample(frac=1, random_state=seed).reset_index(drop=True)
        # add noise
        curr_inp["score"] += rng.random(len(curr_inp)) * 1e-7
        curr_feat = c.transform(curr_inp).set_index(curr_inp["i"].values).sort_index()
        pd.testing.assert_frame_equal(feat, curr_feat, atol=1e-03)


@pytest.fixture()
def candidate_pairs_for_doc():
    return pd.DataFrame(
        {
            "name": ["ABC1", "ABC1", "ABC1", "ABC1"],
            "gt_name": ["GT1", "GT2", "GT3", "GT4"],
            "score": [1.0, 0.9, 0.1, 0.1],
            "uid": [0, 0, 0, 0],
            "gt_uid": [1, 2, 3, 4],
        }
    )


def test_calc_sample_rank_features_for_doc(tmp_path, candidate_pairs_for_doc):
    OUTPUT_FILE = tmp_path / "test_example_of_rank_features.tex"

    fe = PandasFeatureExtractor()
    rank_feat = calc_rank_features(candidate_pairs_for_doc, funcs=fe.rank_features, score_columns=["score"])
    diff_feat = calc_diff_features(candidate_pairs_for_doc, funcs=fe.diff_features, score_columns=["score"])
    assert len(rank_feat.columns) == 5
    assert len(diff_feat.columns) == 2
    res = pd.concat([candidate_pairs_for_doc[["uid", "gt_uid", "score"]], rank_feat, diff_feat], axis=1)
    assert len(res) == 4

    # Remark: when you Transpose the dataframe, the columns contains mixed types,
    # therefore the uid row will contain float instead of int
    res = res.T.rename(columns=lambda x: f"candidate {x + 1}")

    latex = res.style.to_latex()
    fixed_latex = []
    for line in latex.splitlines():
        # properly format index values
        if line.startswith((r"X\_index", r"gt\_index", r"score\_rank")):
            fixed_latex.append(re.sub(r"(\d)\.0", r"\1", line))
        else:
            fixed_latex.append(line)
    fixed_latex = "\n".join(fixed_latex)
    with OUTPUT_FILE.open("w") as f:
        f.write(fixed_latex)
