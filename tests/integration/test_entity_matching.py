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

import numpy as np
import pandas as pd
import pytest

from emm.helper import spark_installed
from tests.utils import add_features_vector_col, create_test_data

if spark_installed:
    import pyspark.sql.functions as F

    from emm.indexing.spark_cos_sim_matcher import dot_product_udf
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(("supervised_on", "model_filename"), [(False, None), (True, "sem_nm.pkl")])
def test_name_matching(spark_session, supervised_on, model_filename, supervised_model):
    """Test the whole name matching pipeline"""
    name_only = True
    nm = SparkEntityMatching(
        {
            "preprocessor": "preprocess_with_punctuation",
            "indexers": [{"type": "cosine_similarity", "tokenizer": "characters", "ngram": 3, "num_candidates": 5}],
            "entity_id_col": "id",
            "uid_col": "uid",
            "name_col": "name",
            "supervised_on": supervised_on,
            "name_only": name_only,
            "supervised_model_dir": supervised_model[2].parent if model_filename is not None else ".",
            "supervised_model_filename": model_filename,
        }
    )
    ground_truth, _ = create_test_data(spark_session)
    nm.fit(ground_truth)

    # Sanity check that ground truth is matched correctly back to ground truth, itself
    matched = nm.transform(ground_truth.select("uid", "id", "name", "country"))
    matched = matched.toPandas()

    best_matches = matched.loc[matched.groupby("uid")["score_0"].idxmax()]
    assert (best_matches["entity_id"] == best_matches["gt_entity_id"]).all()
    assert (best_matches["uid"] == best_matches["gt_uid"]).all()
    pd.testing.assert_series_equal(
        best_matches["score_0"], pd.Series(1.0, index=best_matches.index, dtype="float32"), check_names=False
    )

    # all scores are not null, since there are no-candidate rows (we match GT against GT)
    assert matched["score_0"].between(0, 1 + 1e-6, inclusive="both").all()
    if supervised_on:
        assert matched["nm_score"].between(0, 1 + 1e-6, inclusive="both").all()


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("uid_in_data", "mapping_func"),
    [
        (True, None),
        (True, lambda x: x[::-1]),  # sni with reversed names
        (False, None),
    ],
)
def test_name_matching_with_sni(spark_session, uid_in_data, mapping_func):
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "uid_col": "uid",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 3, "mapping_func": mapping_func}],
    }

    ground_truth = spark_session.createDataFrame(
        [["ABC", 1, 100], ["Eddie Eagle", 2, 101], ["Tzu Sun", 3, 102]], ["name", "id", "uid"]
    )
    if not uid_in_data:
        ground_truth = ground_truth.drop("uid")

    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    for name, expected_gt in [
        ("Dummy", {3} if mapping_func else {1, 2}),
        ("   Tzu Sun II ", {2, 3} if mapping_func else {3}),  # extra spaces in name to verify preprocessing
        ("eddie eagle", {1, 2, 3}),  # perfect match (after preprocessing)
        ("Tzu Suu", {3}),
        ("Tzu San", {2, 3}),
    ]:
        query_data = spark_session.createDataFrame([[name, 10, 1000]], ["name", "id", "uid"])
        if not uid_in_data:
            query_data = query_data.drop("uid")
        res = p.transform(query_data)
        res = res.toPandas()
        actual_gt = set(res["gt_entity_id"].values)
        assert (
            expected_gt == actual_gt
        ), f"candidates mismatch for name='{name}' expected={expected_gt} actual_gt={actual_gt}"


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_name_matching_with_sni_on_test_dataset(spark_session):
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "uid_col": "uid",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 3}],
    }
    ground_truth, names_to_match = create_test_data(spark_session)
    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(names_to_match).toPandas().set_index("name")
    assert len(res.groupby("uid")["score_0"].idxmax()) == 39
    assert set(res.loc["Eddie Arnheim noise"]["gt_preprocessed"]) == {"eddie eagle", "eddie arnheim"}
    assert len(res["gt_uid"]) == 58


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_name_matching_with_sni_on_test_dataset_with_no_matches(spark_session):
    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "uid_col": "uid",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 3}],
        "with_no_matches": True,
    }
    ground_truth, names_to_match = create_test_data(spark_session)
    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(names_to_match).toPandas().set_index("name")
    assert len(res.groupby("uid")["score_0"].idxmax()) == names_to_match.count()
    assert set(res.loc["Eddie Arnheim noise"]["gt_preprocessed"]) == {"eddie eagle", "eddie arnheim"}
    # Not matched due to many similar names in names_to_match (one before 'Tzu Chines Sun' and one after 'Tzu Chinese General'):
    assert np.isnan(res.loc["Tzu Chines Sun a"]["gt_uid"])
    assert res["gt_uid"].isnull().sum() == 236
    assert len(res["gt_uid"].dropna()) == 58


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_entity_matching_with_sni(spark_session, supervised_model):
    em_params = {
        "name_only": False,
        "aggregation_layer": True,
        "account_col": "account",
        "entity_id_col": "id",
        "name_col": "name",
        "uid_col": "uid",
        "indexers": [{"type": "sni", "window_length": 3}],
        "supervised_on": True,
        "supervised_model_dir": supervised_model[0].parent,
        "supervised_model_filename": supervised_model[0].name,
    }

    ground_truth = spark_session.createDataFrame(
        [["ABC", 1, 100], ["Eddie Eagle", 2, 101], ["Tzu Sun", 3, 102]], ["name", "id", "uid"]
    )

    query_data = spark_session.createDataFrame(
        [["Tzu Sun", "A1", 100, 1.0, -1, 1], ["Tzu San", "A1", 100, 1.0, -1, 2], ["A Tzu San", "A1", 100, 1.0, -1, 3]],
        ["name", "account", "amount", "counterparty_account_count_distinct", "id", "uid"],
    )

    def add_em_feat(x):
        return x.withColumn("country", F.lit("PL"))

    ground_truth = add_em_feat(ground_truth)
    query_data = add_em_feat(query_data)

    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(query_data)
    res = res.toPandas()
    best_matches = res.loc[res.groupby("entity_id")["agg_score"].idxmax()]
    assert len(res) == 1
    best_matches = best_matches.iloc[0]
    assert best_matches["account"] == "A1"
    assert best_matches["gt_entity_id"] == 3


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("supervised_on", "aggregation_layer"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_entity_matching_with_custom_columns(supervised_on, aggregation_layer, spark_session, supervised_model):
    em_params = {
        "name_only": False,
        "aggregation_layer": aggregation_layer,
        "account_col": "custom_account",
        "uid_col": "custom_uid",
        "entity_id_col": "custom_index",
        "name_col": "custom_name",
        "freq_col": "custom_amount",
        "indexers": [{"type": "cosine_similarity", "tokenizer": "words"}],
        "supervised_on": supervised_on,
        "supervised_model_dir": supervised_model[0].parent,
        "supervised_model_filename": supervised_model[0].name,
    }

    ground_truth = spark_session.createDataFrame(
        [["ABC", 1, 100], ["Eddie Eagle", 2, 101], ["Tzu Sun", 3, 102]], ["custom_name", "custom_index", "custom_uid"]
    )

    query_data = spark_session.createDataFrame(
        [["Tzu Sun", "A1", 100, -1, 1], ["Tzu San", "A1", 100, -1, 2], ["A Tzu San", "A1", 100, -1, 3]],
        ["custom_name", "custom_account", "custom_amount", "custom_index", "custom_uid"],
    )

    def add_em_feat(x):
        return x.withColumn("country", F.lit("PL"))

    ground_truth = add_em_feat(ground_truth)
    query_data = add_em_feat(query_data)

    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(query_data)
    res = res.toPandas()
    assert len(res) > 0
    if aggregation_layer:
        assert "account" in res.columns
        assert "gt_entity_id" in res.columns
        assert "agg_score" in res.columns
    else:
        assert "name" in res.columns
        assert "gt_name" in res.columns
        assert "gt_preprocessed" in res.columns
        assert "preprocessed" in res.columns
        assert "score_0" in res.columns


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize("uid_col", [("custom_uid"), ("id"), ("uid")])
def test_name_matching_with_multiple_indexers(spark_session, uid_col, tmp_path):
    spark_session.sparkContext.setCheckpointDir(str(tmp_path / "checkpoints"))

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "uid_col": uid_col,
        "supervised_on": False,
        "indexers": [
            {
                "type": "cosine_similarity",
                "cos_sim_lower_bound": 0.5,
                "num_candidates": 3,
                "tokenizer": "words",
                "ngram": 1,
            },
            {
                "type": "cosine_similarity",
                "cos_sim_lower_bound": 0.5,
                "num_candidates": 3,
                "tokenizer": "characters",
                "ngram": 1,
            },
        ],
    }

    ground_truth = spark_session.createDataFrame(
        [["Tzu Sun", 10, 100], ["Eddie Eagle", 20, 200]], ["name", "id", uid_col if uid_col != "id" else "not_used"]
    )

    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    for name, expected_id in [("Tzu Sun II", 10), ("Zhi San", 10)]:
        expected_uid = expected_id if uid_col == "id" else expected_id * 10

        query_data = spark_session.createDataFrame(
            [[name, 10, 100]], ["name", "id", uid_col if uid_col != "id" else "not_used"]
        )
        res = p.transform(query_data)
        res = res.toPandas()
        res = res.iloc[0]
        actual_id = res["gt_entity_id"]
        actual_uid = res["gt_uid"]

        assert expected_id == actual_id, f"candidates mismatch for name='{name}'"
        assert expected_uid == actual_uid, f"candidates mismatch for name='{name}'"


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_entity_matching(spark_session, supervised_model):
    """Test the whole entity matching pipeline"""
    nm = SparkEntityMatching(
        {
            "preprocessor": "preprocess_with_punctuation",
            "indexers": [{"type": "cosine_similarity", "tokenizer": "characters", "ngram": 3, "num_candidates": 5}],
            "entity_id_col": "id",
            "uid_col": "uid",
            "name_col": "name",
            "name_only": False,
            "supervised_on": True,
            "aggregation_layer": True,
            "supervised_model_dir": supervised_model[0].parent,
            "supervised_model_filename": supervised_model[0].name,
        }
    )

    ground_truth_pd = pd.DataFrame(
        [["Tzu Sun", 1, "NL"], ["Eddie Eagle", 2, "NL"], ["Adam Mickiewicz", 3, "PL"], ["Mikołaj Kopernik", 4, "PL"]],
        columns=["name", "id", "country"],
    )
    ground_truth_pd["uid"] = ground_truth_pd.index + 100
    ground_truth = spark_session.createDataFrame(ground_truth_pd)

    query_data_pd = pd.DataFrame(
        [
            ["Tzu Sun", "A1"],
            ["Tzu Sun General B", "A1"],
            ["Eddie Eagle A", "A1"],
            ["Eddie Eagle B", "A2"],
            ["Eddie Eagle", "A3"],  # perfect match, but it is dominated by other 3
            ["Mikołaj Kopernik Tzu", "A3"],
            ["Mikołaj Kopernik Tzu", "A3"],
            ["Mikołaj Kopernik Tzu", "A3"],
        ],
        columns=["name", "account"],
    )
    query_data_pd["uid"] = query_data_pd.index + 10000
    query_data = spark_session.createDataFrame(query_data_pd)
    query_data = (
        query_data.withColumn("id", F.lit(-1))
        .withColumn("country", F.lit("PL"))
        .withColumn("amount", F.lit(1.0))
        .withColumn("counterparty_account_count_distinct", F.lit(1.0))
    )
    nm.fit(ground_truth)

    matched = nm.transform(query_data).toPandas()
    assert len(matched) == query_data.toPandas()["account"].nunique()
    assert matched["account"].nunique() == len(matched)
    matched = matched.set_index("account")
    for account, expected_best_match, _expected_candidates in [("A1", 1, {1, 2}), ("A2", 2, {2}), ("A3", 4, {2, 4})]:
        # These tests are based on sem.pkl trained a very dummy fake pairs create_training_data()
        # therefore the expected_best_match is wrong TODO: use a model trained on proper data
        assert account in matched.index
        match = matched.loc[account]
        assert match["gt_uid"] is not None
        assert match["gt_entity_id"] == expected_best_match


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize("tokenizer", ["words", "characters"])
def test_non_latin_name_matching(spark_session, tokenizer):
    nm = SparkEntityMatching(
        {
            "preprocessor": "preprocess_with_punctuation",
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": tokenizer,
                    "ngram": 3 if tokenizer == "characters" else 1,
                    "num_candidates": 1,
                    "cos_sim_lower_bound": 0.1,
                }
            ],
            "entity_id_col": "id",
            "uid_col": "uid",
            "name_col": "name",
            "supervised_on": False,
            "name_only": True,
        }
    )
    ground_truth = ["a b c", "bździągwa", "ϰaὶ τότ ἐyὼ Kύϰλωπa πpooηύδωv ἄyχi πapaoτάς"]

    ground_truth_sdf = spark_session.createDataFrame(enumerate(ground_truth, start=0), ["id", "name"])
    ground_truth_sdf = ground_truth_sdf.withColumn("uid", ground_truth_sdf["id"])
    nm.fit(ground_truth_sdf)
    queries = [
        "a b",  # sanity check, easy case, latin characters only
        "bzdziagwa",  # no accented characters
        "a b c ϰaὶ τότ ἐyὼ Kύϰλωπa πpooηύδωv ἄyχi πapaoτάς",  # extra "a b c", but all greek words match
    ]
    queries_sdf = spark_session.createDataFrame(enumerate(queries, start=100), ["id", "name"])
    queries_sdf = queries_sdf.withColumn("uid", queries_sdf["id"])
    matched = nm.transform(queries_sdf)
    matched = matched.toPandas()

    best_matches = matched.loc[matched.groupby("entity_id")["score_0"].idxmax()]  # since id == uid
    assert len(best_matches) == len(queries)
    # extract best candidate for each query
    candidates = best_matches["gt_uid"].values
    expected_candidates = [0, 1, 2]
    for query, c, expected in zip(queries, candidates, expected_candidates):
        assert c is not None, f"no match for {query}, expected {ground_truth[expected]}"
        assert c == expected, f"wrong match for {query}, got {ground_truth[c]}, expected {ground_truth[expected]}"


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_full_positive(spark_session):
    """Test manual vectorization and dot product, used for full positive correct template"""
    ground_truth, names_to_match = create_test_data(spark_session)
    names_to_match = names_to_match.withColumn(
        "random_col", F.lit(1)
    )  # testing that all columns in names_to_match are carried on
    ground_truth = ground_truth.drop("country")
    ground_truth.persist()

    # Vectorize pipeline only
    nm_params = {
        "preprocessor": "preprocess_with_punctuation",
        "indexers": [{"type": "cosine_similarity", "tokenizer": "words", "ngram": 1, "num_candidates": 5}],
        "entity_id_col": "id",
        "uid_col": "uid",
        "name_col": "name",
        "keep_all_cols": True,
        "supervised_on": False,
        "name_only": True,
    }
    # this object is only used for vectorization due to cosine_similarity=False and keep_all_cols=True
    nm_vec = SparkEntityMatching(nm_params)

    # Turn off cossim
    stages = nm_vec.pipeline.getStages()
    stages[1].indexers[0].cossim = None

    nm_vec.fit(ground_truth)

    # EM object with Full pipeline, to generate some candidates
    nm_cand = SparkEntityMatching(nm_params)
    nm_cand.fit(ground_truth)

    candidates = nm_cand.transform(names_to_match)
    assert "random_col" in candidates.columns  # testing that all columns in names_to_match are carried on
    assert "preprocessed" in candidates.columns
    candidates = candidates.select(["uid", "entity_id", "name", "preprocessed", "gt_uid", "gt_name", "score_0"])

    # Explode candidates
    candidates_exp = candidates
    candidates_exp = candidates_exp.withColumnRenamed("uid", "correct__uid")
    candidates_exp = candidates_exp.withColumnRenamed("entity_id", "correct__id")
    candidates_exp = candidates_exp.withColumnRenamed("name", "correct__name")
    candidates_exp = candidates_exp.withColumnRenamed("gt_uid", "candidate__gt_uid")
    candidates_exp = candidates_exp.withColumnRenamed("gt_name", "candidate__gt_name")

    # Get the vector feature for name to match
    candidates_exp2 = add_features_vector_col(nm_vec, candidates_exp, "correct__uid", "correct__name")
    candidates_exp2 = candidates_exp2.drop("entity_id").withColumnRenamed("features", "correct__features")

    # Get the vector feature for the candidates
    candidates_exp2 = add_features_vector_col(nm_vec, candidates_exp2, "candidate__gt_uid", "candidate__gt_name")
    candidates_exp2 = candidates_exp2.withColumnRenamed("features", "candidate__features")

    # Compute the dot product between the 2 vectors
    candidates_exp3 = candidates_exp2.withColumn(
        "dot_product", dot_product_udf(F.col("correct__features"), F.col("candidate__features"))
    )

    # It should be the same values
    candidates_exp3_pd = candidates_exp3.toPandas()
    candidates_exp3_pd = candidates_exp3_pd.fillna(0)
    np.testing.assert_allclose(
        candidates_exp3_pd["score_0"].values, candidates_exp3_pd["dot_product"].values, rtol=1e-06
    )


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_entity_matching_duplicates_in_gt(spark_session, supervised_model):
    em_params = {
        "name_only": False,
        "aggregation_layer": True,
        "entity_id_col": "id",
        "uid_col": "uid",
        "name_col": "name",
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

    ground_truth_pd = pd.DataFrame(
        [["Tzu Sun", 1, "NL"] for _ in range(10)] + [["Eddie Eagle", 2, "NL"]], columns=["name", "id", "country"]
    )
    ground_truth_pd["uid"] = ground_truth_pd.index + 100
    ground_truth = spark_session.createDataFrame(ground_truth_pd)

    query_data_pd = pd.DataFrame(
        [["Tzu Sun", "A1", 100, 1.0, 1.0, "NL"]],
        columns=["name", "account", "id", "amount", "counterparty_account_count_distinct", "country"],
    )
    query_data_pd["uid"] = query_data_pd.index + 10000
    query_data = spark_session.createDataFrame(query_data_pd)

    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(query_data)
    res = res.toPandas()
    assert all(res["agg_score"] < 1.0)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_name_matching_with_blocking(spark_session):
    nm = SparkEntityMatching(
        {
            "preprocessor": "preprocess_with_punctuation",
            "entity_id_col": "id",
            "name_col": "name",
            "supervised_on": False,
            "name_only": True,
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": "words",
                    "ngram": 1,
                    "num_candidates": 5,
                    "cos_sim_lower_bound": 0.01,
                    "blocking_func": lambda x: x.strip().lower()[0],  # block using first character
                }
            ],
            "with_no_matches": True,
        }
    )
    gt = spark_session.createDataFrame([["a Tzu", 1], ["b Tzu", 2], ["d Sun", 3]], ["name", "id"])
    names = spark_session.createDataFrame(
        [
            ["a Tzu", 100],  # should be matched only to "a Tzu" id:1
            ["c Tzu", 101],  # should not be matched
        ],
        ["name", "id"],
    )
    nm.fit(gt)
    res = nm.transform(names).toPandas()
    res = res.set_index("entity_id")
    assert res.loc[100]["gt_entity_id"] == 1
    # should not be matched
    assert np.isnan(res.loc[101]["gt_entity_id"])


indexer1 = {"type": "cosine_similarity", "num_candidates": 3, "tokenizer": "words", "ngram": 1}
indexer2 = {"type": "cosine_similarity", "num_candidates": 3, "tokenizer": "characters", "ngram": 2}
indexer3 = {"type": "sni", "num_candidates": 3, "window_length": 3}


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("name_only", "supervised_on", "keep_all_cols", "indexers"),
    [
        (False, True, True, [indexer1, indexer2]),
        (False, False, True, [indexer1, indexer2]),
        (True, False, False, [indexer1]),
        (True, False, True, [indexer1]),
        (False, False, False, [indexer1]),
        (True, True, False, [indexer1]),
        (False, True, False, [indexer1]),
        (True, False, False, [indexer3]),
    ],
)
def test_em_output_columns(spark_session, name_only, supervised_on, keep_all_cols, indexers, supervised_model):
    UID_COL = "custom_uid"
    ENTITY_ID_COL = "custom_index"
    NAME_COL = "custom_name"
    ACCOUNT_COL = "custom_account"

    aggregation_layer = not name_only
    em_params = {
        "name_only": name_only,
        "aggregation_layer": aggregation_layer,
        "uid_col": UID_COL,
        "entity_id_col": ENTITY_ID_COL,
        "name_col": NAME_COL,
        "account_col": ACCOUNT_COL,
        "keep_all_cols": keep_all_cols,
        "supervised_on": supervised_on,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name if name_only else supervised_model[0].name,
        "indexers": indexers,
    }

    ground_truth = spark_session.createDataFrame(
        [[1000, 1, "Tzu Sun", "NL"]], [UID_COL, ENTITY_ID_COL, NAME_COL, "country"]
    )

    names_to_match = spark_session.createDataFrame(
        [[2000, 11, "A1", "Tzu Sun I", "PL", 100, 1.0, "a"]],
        [
            UID_COL,
            ENTITY_ID_COL,
            ACCOUNT_COL,
            NAME_COL,
            "country",
            "amount",
            "counterparty_account_count_distinct",
            "extra",
        ],
    )

    if name_only:
        names_to_match = names_to_match.drop("country", "amount", ACCOUNT_COL, "counterparty_account_count_distinct")

    p = SparkEntityMatching(em_params)
    p = p.fit(ground_truth)
    res = p.transform(names_to_match)

    expected_columns = {
        "uid",
        "name",
        "preprocessed",
        "entity_id",
        "gt_entity_id",
        "gt_uid",
        "gt_name",
        "gt_country",
        "gt_preprocessed",
        "extra",
    }

    if supervised_on:
        expected_columns |= {"nm_score"}

    if supervised_on or aggregation_layer:
        expected_columns |= {"best_match", "best_rank"}

    if not name_only:
        expected_columns |= {"account", "country", "gt_country"}  # 'country' already in input_columns

    if aggregation_layer:
        expected_columns |= {"agg_score", "counterparty_account_count_distinct"}
        # => EM => grouped per (account,id) => we don't have the uid, extra, or the intermediary anymore
        expected_columns -= {"uid", "extra", "amount", "country", "gt_country"}
        if not supervised_on:
            expected_columns |= {"score_0"}
    else:
        if keep_all_cols:
            indexers_type = [indexer["type"] for indexer in indexers]
            if "cosine_similarity" in indexers_type:
                expected_columns |= {"tokens", "ngram_tokens", "tf", "idf", "features"}

        for i in range(len(indexers)):
            expected_columns |= {f"score_{i}"}
            expected_columns |= {f"rank_{i}"}

        if supervised_on:
            expected_columns |= {"nm_score"}

    assert set(res.columns) == expected_columns
