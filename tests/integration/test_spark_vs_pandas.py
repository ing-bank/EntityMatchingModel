import pandas as pd
import pytest

from emm import PandasEntityMatching
from emm.helper import blocking_functions, spark_installed

if spark_installed:
    from emm import SparkEntityMatching


@pytest.fixture(params=[True, False])
def name_only(request):
    return request.param


@pytest.fixture(params=[True, False])
def supervised_on(request):
    return request.param


@pytest.fixture(params=[True, False])
def aggregation_layer(request):
    return request.param


@pytest.fixture(params=["max_frequency_nm_score", "mean_score"])
def aggregation_method(request):
    return request.param


@pytest.fixture(params=[True, False])
def only_nocandidate(request):
    return request.param


@pytest.fixture(params=[True, False])
def freq_col_bug(request):
    return request.param


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_pandas_and_spark_everything_no_candidates(
    spark_session,
    name_only,
    supervised_on,
    aggregation_layer,
    aggregation_method,
    only_nocandidate,
    freq_col_bug,
    supervised_model,
):
    gt = pd.DataFrame(
        [["Tzu Sun", 1, "NL"], ["Eddie Eagle", 2, "NL"], ["Adam Mickiewicz", 3, "NL"], ["Mikołaj Kopernik", 4, "NL"]],
        columns=["name", "id", "country"],
    )
    names = pd.DataFrame(
        [
            ["Tzu Sun A", 1, 5, "NL"],
            ["A Tzu Sun", 2, 5, "NL"],
            ["Kopernik", 3, 5, "NL"],
            ["NOCANDIDATE10", 4, 5, "NL"],
            ["NOCANDIDATE11", 4, None, "NL"],
            ["NOCANDIDATE20", 5, 0, "NL"],
        ],
        columns=["name", "account", "counterparty_account_count_distinct", "country"],
    )

    if only_nocandidate:
        names = names.tail(3)

    if freq_col_bug:
        # To test weird situations:
        # - only no candidate with 0 or None weight for aggregation
        # - account mixing no candidates and candidate
        names.loc[:, "account"] = 1
        names.loc[:, "counterparty_account_count_distinct"] = 0
        names.iloc[0, names.columns.get_loc("counterparty_account_count_distinct")] = None
        names.iloc[-1, names.columns.get_loc("counterparty_account_count_distinct")] = None

    em_params = {
        "name_only": name_only,
        "supervised_on": supervised_on,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "aggregation_layer": aggregation_layer,
        "aggregation_method": aggregation_method,
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1, "num_candidates": 10},
            {
                "type": "cosine_similarity",
                "tokenizer": "characters",
                "ngram": 2,
                "num_candidates": 10,
                "blocking_func": blocking_functions.first,
            },
            {
                "type": "sni",  # Sorted Neighbourhood Indexing,
                "window_length": 3,
            },
        ],
    }

    em_pandas = PandasEntityMatching(em_params)
    em_pandas = em_pandas.fit(gt)
    res_from_pandas = em_pandas.transform(names)

    gt_sd = spark_session.createDataFrame(gt)
    names_sd = spark_session.createDataFrame(names)

    em_spark = SparkEntityMatching(em_params)
    em_spark.fit(gt_sd)
    res_from_spark = em_spark.transform(names_sd)
    res_from_spark = res_from_spark.toPandas()

    assert len(res_from_pandas) == len(res_from_spark)
