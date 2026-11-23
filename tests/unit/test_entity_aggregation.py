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

# flake8: noqa: E501
import numpy as np
import pandas as pd
import pytest

from emm.aggregation.base_entity_aggregation import matching_max_candidate
from emm.aggregation.pandas_entity_aggregation import PandasEntityAggregation
from emm.helper import spark_installed
from tests.utils import read_markdown

if spark_installed:
    from emm.aggregation.spark_entity_aggregation import SparkEntityAggregation


@pytest.fixture()
def sample_one_cluster_candidates():
    return read_markdown(
        """
|   uid | name                    | entity_id | account   |   amount | preprocessed            |   gt_uid |   nm_score |   score_2 |   score_0 |   score_1 | gt_entity_id | gt_name                                               | counterparty_account_count_distinct | count |   partition_id | country   | gt_country   |
|------:|:------------------------|----------:|:----------|---------:|:------------------------|---------:|-----------:|----------:|----------:|----------:|-------------:|:------------------------------------------------------|-------------------------------------|-------|---------------:|:----------|:-------------|
|  1000 | Tzu Sun                 |         1 | G0001     |        1 | tzu sun                 |     1000 |       0.51 |       1   |  1        |  1        |            1 | Tzu Sun                                               |                                   1 |     1 |             81 | NL        | NL           |
|  1000 | Tzu Sun                 |         1 | G0001     |        1 | tzu sun                 |     1002 |       0.49 |       0.5 |  0.603519 |  0.647196 |            1 | Tzu General Dutch Sun                                 |                                   1 |     1 |             81 | NL        | NL           |
|  1000 | Tzu Sun                 |         1 | G0001     |        1 | tzu sun                 |     1001 |       0.50 |     nan   |  0.603519 |  0.68348  |            1 | Tzu General Chinese Sun                               |                                   1 |     1 |             81 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1002 |       0.59 |       1   |  1        |  1        |            1 | Tzu General Dutch Sun                                 |                                   1 |     1 |             22 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1001 |       0.51 |       0.5 |  0.61981  |  0.886813 |            1 | Tzu General Chinese Sun                               |                                   1 |     1 |             22 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1000 |       0.49 |       0.5 |  0.603519 |  0.647196 |            1 | Tzu Sun                                               |                                   1 |     1 |             22 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1015 |       0.44 |     nan   |  0        |  0.549508 |           12 | Vereniging van Vrienden van het Allard Pierson Museum |                                   1 |     1 |             22 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1001 |       0.59 |       1   |  1        |  1        |            1 | Tzu General Chinese Sun                               |                                   1 |     1 |            124 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1002 |       0.51 |       0.5 |  0.61981  |  0.886813 |            1 | Tzu General Dutch Sun                                 |                                   1 |     1 |            124 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1000 |       0.50 |     nan   |  0.603519 |  0.68348  |            1 | Tzu Sun                                               |                                   1 |     1 |            124 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1015 |       0.42 |       0.5 |  0        |  0.482959 |           12 | Vereniging van Vrienden van het Allard Pierson Museum |                                   1 |     1 |            124 | NL        | NL           |"""
    )


def test_matching_max_freq_score_candidate(sample_one_cluster_candidates):
    # All the names and the candidates of 1 account after scoring:
    df = sample_one_cluster_candidates

    match_expected2 = pd.DataFrame(
        {"account": ["G0001"], "entity_id": [1], "gt_entity_id": [1], "gt_uid": [1001], "agg_score": [0.533333]}
    )

    match_result2 = matching_max_candidate(
        df,
        group=["gt_uid", "account"],
        score_col="nm_score",
        name_col="name",
        account_col="account",
        freq_col="counterparty_account_count_distinct",
        output_col="agg_score",
        aggregation_method="max_frequency_nm_score",
    )

    assert set(match_result2.columns) == {*df.columns.tolist(), "agg_score"}

    match_result2 = match_result2[match_expected2.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(match_result2, match_expected2)


@pytest.fixture()
def sample_two_cluster_candidates():
    return read_markdown(
        """
|   uid | name                    | entity_id | account   |   amount | preprocessed            |   gt_uid |   nm_score |   score_2 |   score_0 |   score_1 | gt_entity_id | gt_name                                               | counterparty_account_count_distinct | count |  partition_id | country   | gt_country   |
|------:|:------------------------|----------:|:----------|---------:|:------------------------|---------:|-----------:|----------:|----------:|----------:|-------------:|:------------------------------------------------------|-------------------------------------|-------|--------------:|:----------|:-------------|
|  1000 | ACME Corp               |         1 | G0001     |        1 | acme corp               |     1016 |       0.82 |       1   |  1        |  1        |            1 | ACME Corporation                                      |                                   1 |     1 |            81 | NL        | NL           |
|  1000 | ACME Corp               |         1 | G0001     |        1 | acpe corp               |     1017 |       0.54 |       0.5 |  0.603519 |  0.647196 |            1 | ACME                                                  |                                   1 |     1 |            81 | NL        | NL           |
|  1000 | ACME Corp               |         1 | G0001     |        1 | acme corp               |     1018 |       0.51 |     nan   |  0.603519 |  0.68348  |            1 | A Corp                                                |                                   1 |     1 |            81 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1002 |       0.59 |       1   |  1        |  1        |            1 | Tzu General Dutch Sun                                 |                                   1 |     1 |            22 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1001 |       0.51 |       0.5 |  0.61981  |  0.886813 |            1 | Tzu General Chinese Sun                               |                                   1 |     1 |            22 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1000 |       0.49 |       0.5 |  0.603519 |  0.647196 |            1 | Tzu Sun                                               |                                   1 |     1 |            22 | NL        | NL           |
|  1002 | Tzu General Dutch Sun   |         1 | G0001     |        1 | tzu general dutch sun   |     1015 |       0.44 |     nan   |  0        |  0.549508 |           12 | Vereniging van Vrienden van het Allard Pierson Museum |                                   1 |     1 |            22 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1001 |       0.59 |       1   |  1        |  1        |            1 | Tzu General Chinese Sun                               |                                   1 |     1 |           124 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1002 |       0.51 |       0.5 |  0.61981  |  0.886813 |            1 | Tzu General Dutch Sun                                 |                                   1 |     1 |           124 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1000 |       0.50 |     nan   |  0.603519 |  0.68348  |            1 | Tzu Sun                                               |                                   1 |     1 |           124 | NL        | NL           |
|  1001 | Tzu General Chinese Sun |         1 | G0001     |        1 | tzu general chinese sun |     1015 |       0.42 |       0.5 |  0        |  0.482959 |           12 | Vereniging van Vrienden van het Allard Pierson Museum |                                   1 |     1 |           124 | NL        | NL           |
"""
    )


def test_matching_max_freq_score_candidate_several_clusters(sample_two_cluster_candidates):
    # All the names and the candidates of 1 account after scoring:
    df = sample_two_cluster_candidates

    match_expected = pd.DataFrame(
        {"account": ["G0001"], "entity_id": [1], "gt_entity_id": [1], "gt_uid": [1001], "agg_score": [0.55]}
    )

    match_result = matching_max_candidate(
        df,
        group=["gt_uid", "account"],
        score_col="nm_score",
        name_col="name",
        account_col="account",
        freq_col="counterparty_account_count_distinct",
        output_col="agg_score",
        aggregation_method="max_frequency_nm_score",
    )
    assert set(match_result.columns) == {*df.columns.tolist(), "agg_score"}
    match_result = match_result[match_expected.columns].reset_index(drop=True)

    pd.testing.assert_frame_equal(match_result, match_expected)


def test_matching_max_freq_score_nan_candidate():
    # All the names and the candidates of 1 account after scoring:
    df = read_markdown(
        """
|   uid | name                    | entity_id | account   |   amount | preprocessed            |   gt_uid |   nm_score |   score_2 |   score_0 |   score_1 | gt_entity_id | gt_name                                              | counterparty_account_count_distinct | count |   partition_id | country   | gt_country   |
|------:|:------------------------|----------:|:----------|---------:|:------------------------|---------:|-----------:|----------:|----------:|----------:|-------------:|:-----------------------------------------------------|-------------------------------------|-------|---------------:|:----------|:-------------|
|  1000 | Tzu Sun                 |         1 | G0001     |        1 | tzu sun                 |          |     0.0019 |           |           |           |              |                                                      |                                   1 |     1 |             81 | NL        | NL           |"""
    )

    match_expected = pd.DataFrame(
        {"account": ["G0001"], "entity_id": [1], "gt_entity_id": [np.nan], "gt_uid": [np.nan], "agg_score": [0.0019]}
    )

    match_result = matching_max_candidate(
        df,
        group=["gt_uid", "account"],
        score_col="nm_score",
        name_col="name",
        account_col="account",
        freq_col="counterparty_account_count_distinct",
        output_col="agg_score",
        aggregation_method="max_frequency_nm_score",
    )
    assert set(match_result.columns) == {*df.columns.tolist(), "agg_score"}
    match_result = match_result[match_expected.columns].reset_index(drop=True)

    pd.testing.assert_frame_equal(match_result, match_expected)


@pytest.mark.parametrize("missing_name", [None, np.nan])
def test_matching_max_freq_score_null_names_and_group(missing_name):
    pairs = pd.DataFrame(
        {
            "account": ["a", "a"],
            "gt_entity_id": [None, None],
            "gt_uid": [1, 1],
            "name": pd.Series([missing_name, missing_name], dtype=object),
            "score": [0.3, 0.5],
            "freq": [1, 1],
        }
    )
    result = matching_max_candidate(
        pairs,
        group=["gt_entity_id", "gt_uid", "account"],
        score_col="score",
        name_col="name",
        account_col="account",
        freq_col="freq",
        output_col="agg_score",
    )
    assert result["score"].tolist() == [0.5]
    assert result["agg_score"].iloc[0] == pytest.approx(0.4)


@pytest.mark.parametrize(("frequencies", "expected_name"), [([4, 1], "first"), ([1, 4], "second")])
def test_matching_max_freq_score_selects_null_or_non_null_group(frequencies, expected_name):
    pairs = pd.DataFrame(
        {
            "account": ["a", "a"],
            "gt_entity_id": [None, 2],
            "gt_uid": [1, 1],
            "name": ["first", "second"],
            "score": [0.5, 0.5],
            "freq": frequencies,
        }
    )
    result = matching_max_candidate(
        pairs,
        group=["gt_entity_id", "gt_uid", "account"],
        score_col="score",
        name_col="name",
        account_col="account",
        freq_col="freq",
        output_col="agg_score",
    )
    assert result["name"].tolist() == [expected_name]


def test_matching_max_freq_score_preserves_large_group_keys():
    pairs = pd.DataFrame(
        {
            "account": ["a", "a"],
            "gt_entity_id": [np.nan, np.nan],
            "gt_uid": [2**53, 2**53 + 1],
            "name": ["first", "second"],
            "score": [0.5, 0.5],
            "freq": [1, 4],
        }
    )
    result = matching_max_candidate(
        pairs,
        group=["gt_entity_id", "gt_uid"],
        score_col="score",
        name_col="name",
        account_col="account",
        freq_col="freq",
        output_col="agg_score",
    )
    assert result["gt_uid"].tolist() == [2**53 + 1]


def test_matching_mean_score_float32_sorted_groups():
    pairs = pd.DataFrame(
        {
            "account": ["a"] * 4,
            "gt_uid": [1, 1, 1, 2],
            "name": ["first", "second", "third", "fourth"],
            "score": np.array([0.9] * 4, dtype=np.float32),
            "freq": [1] * 4,
        }
    )
    result = matching_max_candidate(
        pairs,
        group=["gt_uid", "account"],
        score_col="score",
        name_col="name",
        account_col="account",
        freq_col="freq",
        output_col="agg_score",
        aggregation_method="mean_score",
    )
    assert result["gt_uid"].tolist() == [1]
    assert result["agg_score"].dtype == np.float64


@pytest.fixture()
def sample_candidates():
    return read_markdown(
        """
|   uid | name                    | preprocessed            | account |   gt_uid |   nm_score | gt_entity_id | gt_name   | gt_preprocessed    | counterparty_account_count_distinct | count |
|------:|:------------------------|:------------------------|---------|---------:|:--------------------------|----------:|-------------------:|:------------------------------------|-------|
|  1000 | Tzu Sun                 | tzu sun                 | G0001   |        1 |       0.51 |          1   | Tzu San   | tzu san            |                                   1 |     1 |
|  1000 | Tzu Sun                 | tzu sun                 | G0001   |        2 |       0.50 |          2   | Tzu Sunn  | tzu sunn           |                                   1 |     1 |
|  1001 | Tzu Sunn                | tzu sunn                | G0001   |        2 |       1.00 |          2   | Tzu Sunn  | tzu sunn           |                                   1 |     1 |
|  2000 | Abc                     | abc                     | G0002   |        3 |       0.50 |          3   | AABBCC    | aabbcc             |                                   1 |     1 |
"""
    )


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_entity_aggregation(spark_session, sample_candidates):
    pandas_ea = PandasEntityAggregation(score_col="nm_score", account_col="account", uid_col="uid", gt_uid_col="gt_uid")
    res_from_pandas = pandas_ea.transform(sample_candidates)
    spark_ea = SparkEntityAggregation(score_col="nm_score")
    res_from_spark = spark_ea._transform(spark_session.createDataFrame(sample_candidates)).toPandas()

    for res in [res_from_pandas, res_from_spark]:
        assert "agg_score" in res.columns
        assert set(res["account"].unique()) == set(sample_candidates["account"])

    cols = ["gt_entity_id", "agg_score"]
    spark_g = res_from_spark.sort_values(by="account").set_index("account", verify_integrity=True)[cols]
    pandas_g = res_from_pandas.sort_values(by="account").set_index("account", verify_integrity=True)[cols]
    pd.testing.assert_frame_equal(spark_g, pandas_g, check_dtype=False)
