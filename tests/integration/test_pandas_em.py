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

import logging
import os

import numpy as np
import pandas as pd
import pytest
from unidecode import unidecode

from emm.helper import spark_installed
from emm.helper.io import load_pickle
from emm.helper.util import string_columns_to_pyarrow
from emm.indexing.pandas_cos_sim_matcher import PandasCosSimIndexer
from emm.indexing.pandas_normalized_tfidf import PandasNormalizedTfidfVectorizer
from emm.pipeline.pandas_entity_matching import PandasEntityMatching
from emm.preprocessing.pandas_preprocessor import PandasPreprocessor

if spark_installed:
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import CountVectorizer, NGram, RegexTokenizer

    from emm.indexing.spark_normalized_tfidf import SparkNormalizedTfidfVectorizer
    from emm.pipeline.spark_entity_matching import SparkEntityMatching
    from emm.preprocessing.spark_preprocessor import SparkPreprocessor


os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def split_gt_and_names(
    df: pd.DataFrame, gt_limit: int | None = None, names_limit: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataframe of (id,names) into ground truth (gt) and names to match (names).
    It makes sure that there is at most one name with given id,
    and names to match does not contain rows from gt.
    """
    gt = df.groupby("id", as_index=False).first()
    if gt_limit is not None:
        gt = gt.head(gt_limit).copy()

    names = df[df.index.isin(gt.index)]
    if names_limit is not None:
        names = names.head(names_limit).copy()
    return gt, names


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    "pipeline",
    [
        "preprocess_name",
        "preprocess_with_punctuation",
        "preprocess_merge_abbr",
        "preprocess_merge_legal_abbr",
        ["remove_legal_form"],
    ],
)
def test_pandas_preprocessing(spark_session, kvk_dataset, pipeline):
    pandas_pre = PandasPreprocessor(preprocess_pipeline=pipeline, input_col="name", output_col="output")
    pandas_out = pandas_pre.transform(kvk_dataset.copy())
    pandas_out = string_columns_to_pyarrow(df=pandas_out, columns=pandas_out.columns)
    spark_pre = SparkPreprocessor(preprocess_pipeline=pipeline, input_col="name", output_col="output")
    spark_out = spark_pre._transform(spark_session.createDataFrame(kvk_dataset)).toPandas()
    spark_out = string_columns_to_pyarrow(df=spark_out, columns=spark_out.columns)
    for i, pandas_row in pandas_out.iterrows():
        spark_row = spark_out.loc[i]
        assert pandas_row["output"] == spark_row["output"], f"error on name {pandas_row['name']}"
    pd.testing.assert_frame_equal(pandas_out, spark_out)


def create_spark_tfidf(binary_countvectorizer=False, tokenizer="words", ngram=1):
    stages = []

    if tokenizer == "words":
        stages += [RegexTokenizer(inputCol="name", outputCol="tokens", pattern=r"\w+", gaps=False)]
    elif tokenizer == "characters":
        stages += [RegexTokenizer(inputCol="name", outputCol="tokens", pattern=r".", gaps=False)]
    else:
        msg = f"invalid tokenizer: {tokenizer}"
        raise ValueError(msg)

    stages += [NGram(inputCol="tokens", outputCol="ngram_tokens", n=ngram)]

    stages += [CountVectorizer(inputCol="ngram_tokens", outputCol="tf", vocabSize=2**25, binary=binary_countvectorizer)]

    stages += [
        SparkNormalizedTfidfVectorizer(
            count_col="tf",
            token_col="ngram_tokens",
            output_col="features",
            binary_countvectorizer=binary_countvectorizer,
        )
    ]
    return Pipeline(stages=stages)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pandas_tfidf(dtype):
    pandas_t = PandasNormalizedTfidfVectorizer(binary=False, dtype=dtype)
    gt_names = pd.Series(["a", "b", "c", "a c"])
    pandas_t.fit(gt_names)
    assert set(pandas_t.vocabulary_.keys()) == {"a", "b", "c"}
    data = {
        "a": [1, 0, 0],
        "b": [0, 1, 0],
        "e": [0, 0, 0],
        "a b": [0.48693549, 0.87343794, 0],
        "b e": [0, 0.707107, 0],
        "a e": [0.48693549, 0, 0],
    }
    for name, exp_value in data.items():
        res = pandas_t.transform(pd.Series([name]))
        assert res.dtype == dtype
        actual_value = res.toarray()[0]
        np.testing.assert_allclose(actual_value, exp_value, rtol=0, atol=0.001)


def test_pandas_tfidf_ngram():
    pandas_t = PandasNormalizedTfidfVectorizer(binary=True, analyzer="char", ngram_range=(3, 3))
    gt_names = pd.Series(["aaab", "bbbc"])
    pandas_t.fit(gt_names)
    pandas_t.transform(pd.Series(["aaa", "bbb", "ccc", "ddd"]))
    assert set(pandas_t.vocabulary_.keys()) == {"aaa", "aab", "bbb", "bbc"}
    data = {
        "aaa": [1, 0, 0, 0],
        "bbb": [0, 0, 1, 0],
        "aaa xyz xyz": [0.37796447, 0, 0, 0],
        "_!@$": [0, 0, 0, 0],
        "aaabbb": [0.5, 0.5, 0.5, 0],
    }
    for name, exp_value in data.items():
        actual_value = pandas_t.transform(pd.Series([name])).toarray()[0]
        np.testing.assert_allclose(actual_value, exp_value, rtol=0, atol=0.001)


def test_pandas_tfidf_ngram_large(kvk_dataset):
    kvk_dataset_part = kvk_dataset[["name"]].head(10000).copy()
    kvk_dataset_part["name"] = kvk_dataset_part["name"].map(unidecode)
    gt, names = split_dataset(kvk_dataset_part)
    pandas_t = PandasNormalizedTfidfVectorizer(binary=False, analyzer="char", ngram_range=(3, 3))
    pandas_t.fit(gt)
    pandas_res = pandas_t.transform(names).toarray()
    assert len(pandas_res) == len(names)


def test_pandas_tfidf_binary():
    pandas_t = PandasNormalizedTfidfVectorizer(binary=True)
    gt_names = pd.Series(["a", "b b", "c", "a c"], name="name")
    pandas_t.fit(gt_names)
    assert set(pandas_t.vocabulary_.keys()) == {"a", "b", "c"}
    data = {
        "a a": [1, 0, 0],
        "a a b b": [0.48693549, 0.87343794, 0],
        "a a a b": [0.48693549, 0.87343794, 0],
        "a b": [0.48693549, 0.87343794, 0],
        "a e e e": [0.48693549, 0, 0],
        "a e E e": [0.48693549, 0, 0],
        "a e": [0.48693549, 0, 0],
    }
    for name, exp_value in data.items():
        actual_value = pandas_t.transform(pd.Series([name])).toarray()[0]
        np.testing.assert_allclose(actual_value, exp_value, rtol=0, atol=0.001)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize("binary", [False, True])
def test_pandas_tfidf_compatibility_with_spark(binary, spark_session, kvk_dataset):
    kvk_dataset_part = kvk_dataset[["name"]].head(10000).copy()
    kvk_dataset_part["name"] = kvk_dataset_part["name"].map(unidecode)
    gt, names = split_dataset(kvk_dataset_part)
    pandas_t = PandasNormalizedTfidfVectorizer(binary=binary)
    pandas_t.fit(gt)
    pandas_res = pandas_t.transform(names).toarray()
    assert len(pandas_res) == len(names)

    s_gt, s_names = spark_session.createDataFrame(gt), spark_session.createDataFrame(names)

    tokens = RegexTokenizer(inputCol="name", outputCol="tokens", pattern=r"\w+", gaps=False).transform(s_gt)
    tokens_set = set()
    for line in tokens.toPandas()["tokens"].values:
        tokens_set |= set(line)
    missing_tokens = tokens_set ^ set(pandas_t.vocabulary_.keys())
    assert (
        len(missing_tokens) == 0
    ), f"missing tokens: {missing_tokens} set(pandas_t.vocabulary_.keys())={set(pandas_t.vocabulary_.keys())}"
    assert len(tokens_set) == pandas_res.shape[1]

    spark_t = create_spark_tfidf(binary_countvectorizer=binary)
    spark_model = spark_t.fit(s_gt)
    spark_res_full = spark_model.transform(s_names).toPandas()
    assert all(names["name"].values == spark_res_full["name"].values)
    spark_res = np.vstack(spark_res_full.loc[:, "features"].apply(lambda x: x.toArray()).values)
    spark_res = np.nan_to_num(spark_res, nan=0)
    assert len(spark_res) == len(names)
    assert pandas_res.shape == spark_res.shape

    def sort_columns(arr):
        sorted_c_idx = np.lexsort(arr, axis=0)
        return arr[:, sorted_c_idx]

    pandas_res = sort_columns(pandas_res)
    spark_res = sort_columns(spark_res)

    for i in range(len(names)):
        np.testing.assert_allclose(pandas_res[i], spark_res[i], rtol=0, atol=0.001)


@pytest.mark.parametrize("use_blocking", [False, True])
def test_pandas_cos_sim_indexer(use_blocking):
    idx = PandasCosSimIndexer(
        input_col="name",
        cos_sim_lower_bound=0.01,
        num_candidates=10,
        blocking_func=lambda x: x[0] if use_blocking else None,
    )
    gt = pd.DataFrame({"name": ["Tzu Sun", "Mikolaj Kopernik", "A", "Tzu"]})
    names = pd.DataFrame({"name": ["Tzu Sun A", "A Tzu Sun", "Kopernik"]})
    idx.fit(gt)
    res = idx.transform(names)
    g = res.groupby("uid")["gt_uid"].apply(lambda x: set(x.unique())).to_dict()
    if use_blocking:
        assert g == {0: {0, 3}, 1: {2}}  # Kopernik not matched
    else:
        assert g == {0: {0, 2, 3}, 1: {0, 2, 3}, 2: {1}}


def split_dataset(df):
    df = df.drop_duplicates(subset=["name"]).sort_values(by="name")
    gt = df.iloc[range(0, len(df), 2)]
    names = df.iloc[range(1, len(df), 2)]
    return gt, names


def compare_name_matching_results(res1, res2, ntop, min_score, eps=0.00001):
    # drop rows without a match
    res1 = res1.dropna(subset=["gt_entity_id"]).copy()
    res2 = res2.dropna(subset=["gt_entity_id"]).copy()

    def prep(df, source):
        df["k"] = list(zip(df["entity_id"], df["gt_entity_id"]))
        df["source"] = source
        assert df["k"].nunique() == len(df)
        df["neg_score"] = -1.0 * df["score"]
        # we need to use max method to make sure that equal scores receive max rank
        df["score_rank"] = df.groupby("entity_id")["neg_score"].rank(method="max").astype(int)
        return df.set_index("k", drop=True)

    res1 = prep(res1, source="res1")
    res2 = prep(res2, source="res2")
    not_in_one = pd.concat(
        [res1[~res1.index.isin(res2.index)], res2[~res2.index.isin(res1.index)]], ignore_index=True, axis=0, sort=False
    )
    in_both = res1.join(res2["score"].rename("score2"), how="inner")
    in_both["score_diff"] = (in_both["score"] - in_both["score2"]).abs()

    # for matches that appear in only 1 source, we consider BAD if both score is large enough & score_rank is within [1..ntop]
    bad_not_in_one = not_in_one[(not_in_one.score_rank <= ntop) & (not_in_one.score >= min_score)]
    bad_not_in_one = bad_not_in_one.sort_values(by=["entity_id", "score_rank", "source"])
    bad_not_in_one = bad_not_in_one[
        ["entity_id", "gt_entity_id", "score", "score_rank", "source", "name", "gt_name", "source"]
    ]

    # for matches that appear in both sources, we consider BAD if score diff > eps
    bad_in_both = in_both[in_both["score_diff"] > eps]
    bad_in_both = bad_in_both[["name", "gt_name", "score", "score2", "score_diff"]]
    assert len(bad_not_in_one) == 0, f"some matches not found {bad_not_in_one}"
    assert len(bad_in_both) == 0, f"different scores! {bad_in_both}"


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("supervised_on", "use_blocking"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_pandas_name_matching_vs_spark(spark_session, kvk_dataset, supervised_on, use_blocking, supervised_model):
    """This test verifies the compatibility of results from Spark & Pandas name matching.
    Test is parametrized with different EM settings.
    Warning! there could be some small differences due to rounding errors & ordering of the data,
    so the results are compared using specialized function `compare_name_matching_results`.
    """
    gt, names = split_dataset(kvk_dataset)
    gt["uid"] = gt.reset_index().index
    names["uid"] = names.reset_index().index

    ntop = 10
    min_score = 0.5
    min_score_tolerance = 0.001

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "uid_col": "uid",
        "name_col": "name",
        "supervised_on": supervised_on,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                # we add tolerance to both cos_sim & num_candidates to capture pairs just under the threshold
                "cos_sim_lower_bound": min_score - min_score_tolerance,
                # to handle cases with a lot of candidate pairs with the same lowest score:
                "num_candidates": 2 * ntop,
                "blocking_func": (lambda x: x[0] if len(x) > 0 else "?") if use_blocking else None,
            }
        ],
    }
    score_col = "nm_score" if supervised_on else "score_0"

    p = PandasEntityMatching(em_params)
    p = p.fit(gt.copy())
    res_from_pandas = p.transform(names.copy()).rename(columns={score_col: "score"})

    s = SparkEntityMatching(em_params)
    s = s.fit(spark_session.createDataFrame(gt))
    res_from_spark = s.transform(spark_session.createDataFrame(names))
    res_from_spark = res_from_spark.toPandas()
    res_from_spark = res_from_spark.rename(columns={score_col: "score"})
    res_from_spark = res_from_spark[["entity_id", "name", "gt_entity_id", "gt_name", "score"]]

    # all scores should be from range 0..1 (and None for no-candidate rows)
    assert res_from_pandas["score"].round(decimals=5).between(0, 1, inclusive="both").all()
    assert res_from_spark["score"].round(decimals=5).between(0, 1, inclusive="both").all()

    compare_name_matching_results(
        res_from_pandas, res_from_spark, ntop=ntop, min_score=min_score, eps=min_score_tolerance
    )


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("supervised_on", "use_blocking"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_pandas_name_matching_vs_spark_with_no_matches(
    spark_session, kvk_dataset, supervised_on, use_blocking, supervised_model
):
    """This test verifies the compatibility of results from Spark & Pandas name matching.
    Test is parametrized with different EM settings.
    Warning! there could be some small differences due to rounding errors & ordering of the data,
    so the results are compared using specialized function `compare_name_matching_results`.
    """
    gt, names = split_dataset(kvk_dataset)
    gt["uid"] = gt.reset_index().index
    names["uid"] = names.reset_index().index

    ntop = 10
    min_score = 0.5
    min_score_tolerance = 0.001

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "uid_col": "uid",
        "name_col": "name",
        "supervised_on": supervised_on,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                # we add tolerance to both cos_sim & num_candidates to capture pairs just under the threshold
                "cos_sim_lower_bound": min_score - min_score_tolerance,
                # to handle cases with a lot of candidate pairs with the same lowest score:
                "num_candidates": 2 * ntop,
                "blocking_func": (lambda x: x[0] if len(x) > 0 else "?") if use_blocking else None,
            }
        ],
        "with_no_matches": True,
    }
    score_col = "nm_score" if supervised_on else "score_0"

    p = PandasEntityMatching(em_params)
    p = p.fit(gt.copy())
    res_from_pandas = p.transform(names.copy()).rename(columns={score_col: "score"})

    s = SparkEntityMatching(em_params)
    s = s.fit(spark_session.createDataFrame(gt))
    res_from_spark = s.transform(spark_session.createDataFrame(names))
    res_from_spark = res_from_spark.toPandas()
    res_from_spark = res_from_spark.rename(columns={score_col: "score"})
    res_from_spark = res_from_spark[["entity_id", "name", "gt_entity_id", "gt_name", "score"]]

    # double check if number of no-candidate rows is reasonable
    assert 0.5 < res_from_pandas["score"].isnull().mean() < 0.90
    assert 0.5 < res_from_spark["score"].isnull().mean() < 0.90
    # all scores should be from range 0..1 (and None for no-candidate rows)
    assert res_from_pandas["score"].fillna(0).round(decimals=5).between(0, 1, inclusive="both").all()
    assert res_from_spark["score"].fillna(0).round(decimals=5).between(0, 1, inclusive="both").all()

    compare_name_matching_results(
        res_from_pandas, res_from_spark, ntop=ntop, min_score=min_score, eps=min_score_tolerance
    )


def test_pandas_entity_matching_without_indexers():
    em_params = {"name_only": True, "supervised_on": False, "indexers": []}
    ground_truth = pd.DataFrame(
        [["Tzu Sun", 1], ["Eddie Eagle", 2], ["Adam Mickiewicz", 3], ["Mikołaj Kopernik", 4]], columns=["name", "id"]
    )
    p = PandasEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(ground_truth)
    assert "preprocessed" in res.columns


def test_pandas_entity_matching_simple_case(supervised_model):
    ntop = 10
    min_score = 0.5
    min_score_tolerance = 0.001

    em_params = {
        "name_only": False,
        "aggregation_layer": True,
        "aggregation_method": "max_frequency_nm_score",
        "entity_id_col": "id",
        "name_col": "name",
        "freq_col": "counterparty_account_count_distinct",
        "supervised_on": True,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                # we add tolerance to both cos_sim & num_candidates to capture pairs just under the threshold
                "cos_sim_lower_bound": min_score - min_score_tolerance,
                "num_candidates": ntop,
            }
        ],
    }

    ground_truth = pd.DataFrame(
        [["Tzu Sun", 1, "NL"], ["Eddie Eagle", 2, "NL"], ["Adam Mickiewicz", 3, "PL"], ["Mikołaj Kopernik", 4, "PL"]],
        columns=["name", "id", "country"],
    )

    query_data = pd.DataFrame(
        [
            ["Tzu Sun A", "A1", 100],
            ["Tzu Sun General B", "A1", 100],
            ["Eddie Eagle A", "A1", 100],
            ["Eddie Eagle B", "A2", 101],
            ["Eddie Eagle", "A3", 102],  # perfect match, but it is dominated by other
            ["Mikołaj Kopernik Tzu", "A3", 102],
            ["Mikołaj Kopernik Tzu", "A3", 102],
            ["Mikołaj Kopernik Tzu", "A3", 102],
            ["Mikołaj Kopernik Tzu", "A3", 102],
            ["Mikołaj Kopernik Tzu", "A3", 102],
        ],
        columns=["name", "account", "id"],
    )
    query_data["amount"] = 1.0
    query_data["counterparty_account_count_distinct"] = 1.0
    query_data["country"] = "PL"

    p = PandasEntityMatching(em_params)
    p = p.fit(ground_truth)
    # double check if nothing breaks without id column in query
    matched_without_id = p.transform(query_data.drop(columns="id"))
    assert "id" not in matched_without_id.columns
    matched = p.transform(query_data)
    assert "entity_id" in matched.columns
    assert "score_0" in matched.columns
    assert matched["score_0"].dtype == np.float32  # score from cossim indexer
    assert "nm_score" in matched.columns
    assert "agg_score" in matched.columns

    best_match = matched[matched.best_match].set_index("account")["gt_entity_id"]
    candidates = matched.groupby("account")["gt_entity_id"].unique()
    for account, expected_best_match, expected_candidates in [("A1", 1, {1}), ("A2", 2, {2}), ("A3", 4, {4})]:
        # These tests are based on sem_nm.pkl trained a very dummy fake pairs create_training_data()
        # therefore the expected_best_match is wrong TODO: use a model trained on proper data
        assert account in best_match.index
        assert best_match.loc[account] == expected_best_match
        assert set(candidates.loc[account]) == expected_candidates


def default_em_params():
    return {
        "name_only": True,
        "entity_id_col": "id",
        "supervised_on": False,
        "supervised_model_dir": ".",
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                "cos_sim_lower_bound": 0.5,
                "num_candidates": 5,
            }
        ],
    }


def test_pandas_name_matching_with_two_supervised_models(kvk_dataset, supervised_model):
    gt, names = split_dataset(kvk_dataset)
    p = PandasEntityMatching(
        {**default_em_params(), "supervised_on": True},
        supervised_models={
            "nm_score_with_rank": {
                "model": load_pickle(supervised_model[2].name, supervised_model[2].parent),
                "enable": True,
            },
            "nm_score_without_rank": {
                "model": load_pickle(supervised_model[4].name, supervised_model[4].parent),
                "enable": True,
            },
        },
    )
    p = p.fit(gt)
    res = p.transform(names)
    assert "nm_score_with_rank" in res.columns
    assert "nm_score_without_rank" in res.columns


def test_pandas_entity_matching_duplicates_in_gt(supervised_model):
    em_params = {
        "name_only": False,
        "aggregation_layer": True,
        "entity_id_col": "id",
        "name_col": "name",
        "freq_col": "counterparty_account_count_distinct",
        "supervised_on": True,
        "supervised_model_dir": supervised_model[0].parent,
        "supervised_model_filename": supervised_model[0].name,
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                "cos_sim_lower_bound": 0.1,
                "num_candidates": 10,
            }
        ],
    }

    ground_truth = pd.DataFrame(
        [["Tzu Sun", 1, "NL"] for _ in range(10)] + [["Eddie Eagle", 2, "NL"]], columns=["name", "id", "country"]
    )

    query_data = pd.DataFrame([["Tzu Sun", "A1", 100]], columns=["name", "account", "id"])
    query_data["amount"] = 1.0
    query_data["counterparty_account_count_distinct"] = 1.0
    query_data["country"] = "PL"

    p = PandasEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(query_data)
    assert all(res["nm_score"] < 1.0)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_pandas_entity_matching(spark_session, kvk_dataset, supervised_model):
    kvk_dataset = kvk_dataset.copy()
    kvk_dataset["country"] = "PL"
    kvk_dataset["amount"] = 1.0
    kvk_dataset["uid"] = kvk_dataset.reset_index().index
    gt, names = split_dataset(kvk_dataset)

    names["uid"] = names.reset_index().index
    del names["id"]
    names["account"] = [i // 5 for i in range(len(names))]
    names["counterparty_account_count_distinct"] = 1.0

    ntop = 10
    min_score = 0.5
    min_score_tolerance = 0.001

    em_params = {
        "name_only": False,
        "aggregation_layer": True,
        # "aggregation_method": "mean_score",
        "entity_id_col": "id",
        "uid_col": "uid",
        "name_col": "name",
        "freq_col": "counterparty_account_count_distinct",
        "supervised_on": True,
        "supervised_model_dir": supervised_model[0].parent,
        "supervised_model_filename": supervised_model[0].name,
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                # we add tolerance to both cos_sim & num_candidates to capture pairs just under the threshold
                "cos_sim_lower_bound": min_score - min_score_tolerance,
                "num_candidates": ntop,
            }
        ],
    }

    p = PandasEntityMatching(em_params)
    p = p.fit(gt.copy())
    res_from_pandas = p.transform(names.copy())

    em_params["freq_col"] = "counterparty_account_count_distinct"
    s = SparkEntityMatching(em_params)
    s = s.fit(spark_session.createDataFrame(gt))
    res_from_spark = s.transform(spark_session.createDataFrame(names))
    res_from_spark = res_from_spark.toPandas()
    res_from_spark["account"] = res_from_spark["account"].astype(int)

    best_from_pandas = (
        res_from_pandas[res_from_pandas.best_match][["account", "gt_entity_id", "agg_score"]]
        .rename(
            columns={
                "account": "account",
                "gt_entity_id": "pandas_best_match_id",
                "agg_score": "pandas_best_match_score",
            }
        )
        .set_index("account", verify_integrity=True)
        .sort_index()
    )
    best_from_spark = (
        res_from_spark[["account", "gt_entity_id", "agg_score"]]
        .rename(columns={"gt_entity_id": "spark_best_match_id", "agg_score": "spark_best_match_score"})
        .set_index("account", verify_integrity=True)
        .sort_index()
    )
    res_cmp = pd.concat([best_from_pandas, best_from_spark], axis=1)

    # make sure that results between pandas are spark are consistent
    # - there are no accounts with best match selected only by Pandas or only Spark
    assert len(res_cmp[(res_cmp["pandas_best_match_id"].isnull()) & (res_cmp["spark_best_match_id"].notnull())]) == 0
    assert len(res_cmp[(res_cmp["pandas_best_match_id"].notnull()) & (res_cmp["spark_best_match_id"].isnull())]) == 0
    # change nulls to -1 to simplify comparison
    res_cmp["spark_best_match_id"] = res_cmp["spark_best_match_id"].fillna(-1).astype(int)
    res_cmp["pandas_best_match_id"] = res_cmp["pandas_best_match_id"].fillna(-1).astype(int)
    # - the match score is calculated in the same way (up to some tolerance)
    assert (res_cmp["spark_best_match_score"] - res_cmp["pandas_best_match_score"]).dropna().abs().mean() < 0.00001
    # - the best matches are selected in the same way (up to 95%)
    # (with MLP model we add 99%, with xgboost 95% due to many rows having the same score, because sem.pkl is training on very simple fake candidates)
    assert (res_cmp["spark_best_match_id"] == res_cmp["pandas_best_match_id"]).mean() > 0.95


def test_pandas_sni():
    id = list(range(10, 100, 10))
    gt = pd.DataFrame({"id": id, "name": [f"A{x:03d}" for x in id]})
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 5}],
    }
    p = PandasEntityMatching(em_params)
    p = p.fit(gt.copy())
    for name, expected_gt in [
        ("A000", {"A010", "A020"}),
        ("A055", {"A040", "A050", "A060", "A070"}),
        ("A050", {"A030", "A040", "A050", "A060", "A070"}),
        ("A100", {"A080", "A090"}),
    ]:
        res = p.transform(pd.DataFrame({"name": [name], "id": 0}))
        assert set(res["gt_name"].unique()) == expected_gt


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize("use_mapping", [False, True])
def test_pandas_sni_on_kvk_dataset(spark_session, kvk_dataset, use_mapping):
    gt, names = split_dataset(kvk_dataset)
    gt["uid"] = gt.reset_index().index
    names["uid"] = names.reset_index().index

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "uid_col": "uid",
        "name_col": "name",
        "supervised_on": False,
        "indexers": [
            {"type": "sni", "window_length": 3, "mapping_func": ((lambda x: x[::-1]) if use_mapping else None)}
        ],
    }

    p = PandasEntityMatching(em_params)
    p = p.fit(gt.copy())
    res_from_pandas = p.transform(names.copy())

    s = SparkEntityMatching(em_params)
    s = s.fit(spark_session.createDataFrame(gt))
    res_from_spark = s.transform(spark_session.createDataFrame(names))
    res_from_spark = res_from_spark.toPandas()
    res_from_spark = res_from_spark.dropna(subset=["gt_entity_id"])
    res_from_spark["gt_entity_id"] = res_from_spark["gt_entity_id"].astype(int)
    res_from_spark = res_from_spark[["entity_id", "name", "gt_entity_id", "gt_name", "score_0"]]

    res_from_pandas = res_from_pandas.dropna(subset=["gt_uid"])
    res_from_pandas["gt_entity_id"] = res_from_pandas["gt_entity_id"].astype(int)

    def add_idx(df):
        return df.sort_values(by=["entity_id", "gt_entity_id"]).set_index(
            ["entity_id", "gt_entity_id"], verify_integrity=True
        )

    res_from_pandas = add_idx(res_from_pandas)
    res_from_spark = add_idx(res_from_spark)

    if not use_mapping:
        assert len(res_from_pandas) / len(names) > 1.5
        assert len(res_from_spark) / len(names) > 1.5

    assert len(res_from_pandas) == len(res_from_spark)
    assert all(res_from_pandas.index == res_from_spark.index)
    pd.testing.assert_series_equal(res_from_pandas["score_0"], res_from_spark["score_0"])


def test_multi_indexers(kvk_dataset):
    gt, names = split_dataset(kvk_dataset.head(1000))
    # new indexers param
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1},
            {"type": "cosine_similarity", "tokenizer": "characters", "ngram": 3},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = PandasEntityMatching(em_params)
    p = p.fit(gt)
    res = p.transform(names).dropna(subset=["gt_uid"])
    assert "sni" in res.columns
    assert "score_3" in res.columns
    assert "cossim_w1" in res.columns
    assert res["sni"].sum() > 0
    assert res["cossim_w1"].sum() > 0
    assert res["cossim_n3"].sum() > 0
    assert all(res["sni"].notnull())
    assert all(res["cossim_w1"].notnull())


def test_multi_indexers_simple_case():
    gt = pd.DataFrame({"name": ["abc", "b c d"], "id": [1, 2]})
    names = pd.DataFrame({"name": ["abc a", "abd", "xyz"], "id": [10, 20, 30]})
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "preprocessor": "preprocess_name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1},
        ],
    }
    p = PandasEntityMatching(em_params)
    p = p.fit(gt)
    res = p.transform(names).set_index(["entity_id", "gt_entity_id"], drop=True)
    for X_id, gt_id, exp_cossim_w1, exp_cossim_n1 in [(10, 1, 1, 1), (20, 1, 0, 1)]:
        assert (X_id, gt_id) in res.index
        row = res.loc[(X_id, gt_id)]
        assert row["cossim_w1"] == exp_cossim_w1
        assert row["cossim_n1"] == exp_cossim_n1


def test_multi_indexers_simple_case_with_no_matches():
    gt = pd.DataFrame({"name": ["abc", "b c d"], "id": [1, 2]})
    names = pd.DataFrame({"name": ["abc a", "abd", "xyz"], "id": [10, 20, 30]})
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "preprocessor": "preprocess_name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1},
        ],
        "with_no_matches": True,
    }
    p = PandasEntityMatching(em_params)
    p = p.fit(gt)
    res = p.transform(names).set_index(["entity_id", "gt_entity_id"], drop=True)
    for X_id, gt_id, exp_cossim_w1, exp_cossim_n1 in [(10, 1, 1, 1), (20, 1, 0, 1), (30, None, 0, 0)]:
        assert (X_id, gt_id) in res.index
        row = res.loc[(X_id, gt_id)]
        assert row["cossim_w1"] == exp_cossim_w1
        assert row["cossim_n1"] == exp_cossim_n1


def test_train_supervised_model(kvk_training_dataset):
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 5}],
    }
    p = PandasEntityMatching(em_params)
    train_gt, train_names = split_gt_and_names(kvk_training_dataset.head(10**3))
    p.fit_classifier(train_names, train_gt=train_gt)
    sm = p.model.steps[2][1]

    for model_dict in sm.supervised_models.values():
        model = model_dict["model"]
        feat_obj = model.named_steps["feat"]
        break
    assert len(feat_obj.vocabulary.very_common_words) > 0
    assert len(feat_obj.vocabulary.common_words) > 0

    test_gt, test_names = split_gt_and_names(kvk_training_dataset.tail(10**3))

    em_params["supervised_on"] = True
    em_params["supervised_model_object"] = model

    p = PandasEntityMatching(em_params)
    p.fit(test_gt)
    candidates = p.transform(test_names)
    assert (candidates["entity_id"] == candidates["gt_entity_id"]).mean() > 0.1


def test_silent_em_output(capsys, caplog, kvk_training_dataset, supervised_model):
    """Verify if Pandas EM can be run in silent mode (no output on stdout/stderr)"""
    caplog.set_level(logging.INFO)
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": True,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = PandasEntityMatching(em_params)
    gt, names = split_gt_and_names(kvk_training_dataset.head(10**3))
    p.fit(gt)
    _ = p.transform(names)

    # no output on stdout/stderr
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""

    # make sure that regular run does not produce any WARNING | ERROR
    # only log entries with <= INFO level
    for name, level, msg in caplog.record_tuples:
        assert level <= logging.INFO
        assert name.startswith("emm"), f"non-em logger used: [{name}, {level}, {msg}]"


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_calc_sm_features(spark_session, kvk_training_dataset, supervised_model):
    """Calculate Supervised model features"""
    gt, names = split_gt_and_names(kvk_training_dataset.head(10**4), gt_limit=50, names_limit=50)

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": True,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "return_sm_features": True,
    }
    p = PandasEntityMatching(em_params)
    p.fit(gt)
    res = p.transform(names)
    assert {"nm_score_feat_score_0", "nm_score_feat_norm_ed", "nm_score_feat_score_0_rank"} <= set(res.columns)

    p2 = SparkEntityMatching(em_params)
    p2.fit(spark_session.createDataFrame(gt))
    res2 = p2.transform(spark_session.createDataFrame(names))
    assert {"nm_score_feat_score_0", "nm_score_feat_norm_ed", "nm_score_feat_score_0_rank"} <= set(res2.columns)
    res2_pd = res2.toPandas()
    assert all(res.filter(regex="^nm_score_feat").columns == res2_pd.filter(regex="^nm_score_feat").columns)
