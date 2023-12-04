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

from emm import PandasEntityMatching
from emm.helper import spark_installed

if spark_installed:
    from emm import SparkEntityMatching


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_spark_entity_matching_add_supervised_model(spark_session, supervised_model):
    gt = pd.DataFrame(
        [(1, "John Smith LLC"), (2, "ING LLC"), (3, "John Doe LLC"), (4, "Tzu Sun G.M.B.H"), (5, "Random GMBH")],
        columns=["id", "name"],
    )
    gt = spark_session.createDataFrame(gt)

    namestomatch = pd.DataFrame(
        [
            (10, "John Smith"),
            (11, "I.n.G. LLC"),
            (12, "Jon DOEE LLC"),  # this will not be matched due to misspellings
        ],
        columns=["id", "name"],
    )
    namestomatch = spark_session.createDataFrame(namestomatch)

    # with supervised model
    nms = SparkEntityMatching(
        {
            "name_only": True,
            "name_col": "name",
            "entity_id_col": "id",
            "supervised_on": True,
            "supervised_model_dir": supervised_model[2].parent,
            "supervised_model_filename": supervised_model[2].name,
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": "words",
                    "ngram": 1,
                    "cos_sim_lower_bound": 0.5,
                    "num_candidates": 10,
                }
            ],
        }
    )
    nms.fit(gt)

    # without supervised model
    nm = SparkEntityMatching(
        {
            "name_only": True,
            "name_col": "name",
            "entity_id_col": "id",
            "supervised_on": False,
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": "words",
                    "ngram": 1,
                    "cos_sim_lower_bound": 0.5,
                    "num_candidates": 10,
                }
            ],
        }
    )
    nm.fit(gt)
    # add supervised model later, but now after fitting indexers
    nm.add_supervised_model(supervised_model[2])

    # calculate and compare two versions
    ress = nms.transform(namestomatch)
    res = nm.transform(namestomatch)

    ress = ress.toPandas()
    res = res.toPandas()

    assert len(res) == len(ress)
    assert set(ress.columns) == set(res.columns)
    assert "nm_score" in ress.columns
    assert "nm_score" in res.columns
    np.testing.assert_almost_equal(res["nm_score"].sum(), ress["nm_score"].sum())


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_spark_entity_matching_add_aggregation_layer(spark_session, supervised_model):
    gt = pd.DataFrame(
        [["Tzu Sun", 1, "NL"], ["Eddie Eagle", 2, "NL"], ["Adam Mickiewicz", 3, "PL"], ["Mikołaj Kopernik", 4, "PL"]],
        columns=["name", "id", "country"],
    )
    gt = spark_session.createDataFrame(gt)

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
    query_data = spark_session.createDataFrame(query_data)

    em_params = {
        "name_only": False,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                # we add tolerance to both cos_sim & num_candidates to capture pairs just under the threshold
                "cos_sim_lower_bound": 0.5,
                "num_candidates": 10,
            }
        ],
        "aggregation_layer": True,
        "aggregation_method": "max_frequency_nm_score",
        "freq_col": "counterparty_account_count_distinct",
        "account_col": "account",
    }

    em_params2 = em_params.copy()
    del em_params2["aggregation_layer"]
    del em_params2["aggregation_method"]
    del em_params2["freq_col"]
    del em_params2["account_col"]

    # aggregation layer has already been added
    pa = SparkEntityMatching(em_params)
    pa.fit(gt)
    resa = pa.transform(query_data)

    # aggregation layer has already been added
    p = SparkEntityMatching(em_params2)
    p.fit(gt)
    p.add_aggregation_layer(
        aggregation_method="max_frequency_nm_score",
        account_col="account",
        freq_col="counterparty_account_count_distinct",
    )
    resb = p.transform(query_data)

    resa = resa.toPandas()
    resb = resb.toPandas()

    assert len(resb) == len(resa)
    assert len(resb.columns) == len(resa.columns)
    assert set(resb.columns) == set(resa.columns)
    assert "agg_score" in resa.columns
    assert "agg_score" in resb.columns
    np.testing.assert_almost_equal(resb["agg_score"].sum(), resa["agg_score"].sum())


def test_pandas_entity_matching_add_supervised_model(supervised_model):
    gt = pd.DataFrame(
        [(1, "John Smith LLC"), (2, "ING LLC"), (3, "John Doe LLC"), (4, "Tzu Sun G.M.B.H"), (5, "Random GMBH")],
        columns=["id", "name"],
    )

    namestomatch = pd.DataFrame(
        [
            (10, "John Smith"),
            (11, "I.n.G. LLC"),
            (12, "Jon DOEE LLC"),  # this will not be matched due to misspellings
        ],
        columns=["id", "name"],
    )

    # with supervised model
    nms = PandasEntityMatching(
        {
            "name_only": True,
            "name_col": "name",
            "entity_id_col": "id",
            "freq_col": "counterparty_account_count_distinct",
            "supervised_on": True,
            "supervised_model_dir": supervised_model[2].parent,
            "supervised_model_filename": supervised_model[2].name,
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": "words",
                    "ngram": 1,
                    "cos_sim_lower_bound": 0.5,
                    "num_candidates": 10,
                }
            ],
        }
    )
    nms.fit(gt)

    # without supervised model
    nm = PandasEntityMatching(
        {
            "name_only": True,
            "name_col": "name",
            "entity_id_col": "id",
            "freq_col": "counterparty_account_count_distinct",
            "supervised_on": False,
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": "words",
                    "ngram": 1,
                    "cos_sim_lower_bound": 0.5,
                    "num_candidates": 10,
                }
            ],
        }
    )
    nm.fit(gt)
    # add supervised model later, but now after fitting indexers
    nm.add_supervised_model(supervised_model[2])

    # calculate and compare two versions
    ress = nms.transform(namestomatch)
    res = nm.transform(namestomatch)

    assert len(res) == len(ress)
    assert set(ress.columns) == set(res.columns)
    assert "nm_score" in ress.columns
    assert "nm_score" in res.columns
    np.testing.assert_almost_equal(res["nm_score"].sum(), ress["nm_score"].sum())


def test_pandas_entity_matching_add_aggregation_layer(supervised_model):
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

    em_params = {
        "name_only": False,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "indexers": [
            {
                "type": "cosine_similarity",
                "tokenizer": "words",
                "ngram": 1,
                # we add tolerance to both cos_sim & num_candidates to capture pairs just under the threshold
                "cos_sim_lower_bound": 0.5,
                "num_candidates": 10,
            }
        ],
        "aggregation_layer": True,
        "aggregation_method": "mean_score",
        "freq_col": "counterparty_account_count_distinct",
        "account_col": "account",
    }

    em_params2 = em_params.copy()
    del em_params2["aggregation_layer"]
    del em_params2["aggregation_method"]
    del em_params2["freq_col"]

    # aggregation layer has already been added
    pa = PandasEntityMatching(em_params)
    pa = pa.fit(ground_truth)
    resa = pa.transform(query_data)

    # aggregation layer has already been added
    p = PandasEntityMatching(em_params2)
    p = p.fit(ground_truth)
    res = p.transform(query_data)

    p.add_aggregation_layer(
        aggregation_method="mean_score", account_col="account", freq_col="counterparty_account_count_distinct"
    )
    resb = p.transform(query_data)

    assert len(res) > len(resa)
    assert len(resb) == len(resa)
    assert len(resb.columns) == len(resa.columns)
    assert set(resb.columns) == set(resa.columns)
    assert "agg_score" in resa.columns
    assert "agg_score" in resb.columns
    np.testing.assert_almost_equal(resb["agg_score"].sum(), resa["agg_score"].sum())
