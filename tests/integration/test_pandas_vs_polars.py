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
from unidecode import unidecode

from emm.helper import blocking_functions
from emm.helper.util import string_columns_to_pyarrow
from emm.indexing.pandas_normalized_tfidf import PandasNormalizedTfidfVectorizer
from emm.pipeline.pandas_entity_matching import PandasEntityMatching
from emm.preprocessing.pandas_preprocessor import PandasPreprocessor
from tests.integration.test_pandas_em import compare_name_matching_results, split_dataset, split_gt_and_names

pl = pytest.importorskip("polars")
PolarsEntityMatching = import_module("emm.pipeline.polars_entity_matching").PolarsEntityMatching
PolarsPreprocessor = import_module("emm.preprocessing.polars_preprocessor").PolarsPreprocessor
PolarsNormalizedTfidfVectorizer = import_module("emm.indexing.polars_normalized_tfidf").PolarsNormalizedTfidfVectorizer


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
def test_pandas_polars_preprocessing(kvk_dataset, pipeline):
    pandas_pre = PandasPreprocessor(preprocess_pipeline=pipeline, input_col="name", output_col="output")
    pandas_out = string_columns_to_pyarrow(df=pandas_pre.transform(kvk_dataset.copy()))
    polars_pre = PolarsPreprocessor(preprocess_pipeline=pipeline, input_col="name", output_col="output")
    polars_out = string_columns_to_pyarrow(df=polars_pre.transform(pl.from_pandas(kvk_dataset)).to_pandas())
    for i, pandas_row in pandas_out.iterrows():
        assert pandas_row["output"] == polars_out.loc[i, "output"], f"error on name {pandas_row['name']}"
    pd.testing.assert_frame_equal(pandas_out, polars_out)


@pytest.mark.parametrize("binary", [False, True])
def test_pandas_polars_tfidf(binary, kvk_dataset):
    data = kvk_dataset[["name"]].head(10000).copy()
    data["name"] = data["name"].map(unidecode)
    gt, names = split_dataset(data)
    pandas_t = PandasNormalizedTfidfVectorizer(binary=binary).fit(gt)
    polars_t = PolarsNormalizedTfidfVectorizer(binary=binary).fit(pl.from_pandas(gt))
    pandas_res = pandas_t.transform(names).toarray()
    polars_res = polars_t.transform(pl.from_pandas(names)).toarray()
    assert len(pandas_res) == len(names)
    assert len(polars_res) == len(names)
    assert set(pandas_t.vocabulary_) == set(polars_t.vocabulary_)
    assert len(pandas_t.vocabulary_) == pandas_res.shape[1]
    assert pandas_res.shape == polars_res.shape
    for i in range(len(names)):
        np.testing.assert_allclose(pandas_res[i], polars_res[i], rtol=0, atol=0.001)


@pytest.mark.parametrize(
    ("supervised_on", "use_blocking"), [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("with_no_matches", [False, True])
def test_pandas_polars_name_matching(kvk_dataset, supervised_on, use_blocking, with_no_matches, supervised_model):
    gt, names = split_dataset(kvk_dataset)
    gt["uid"] = gt.reset_index().index
    names["uid"] = names.reset_index().index
    ntop, min_score, tolerance = 10, 0.5, 0.001
    params = {
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
                "cos_sim_lower_bound": min_score - tolerance,
                "num_candidates": 2 * ntop,
                "blocking_func": (lambda x: x[0] if len(x) > 0 else "?") if use_blocking else None,
            }
        ],
        "with_no_matches": with_no_matches,
        "n_jobs": 1,
    }
    score_col = "nm_score" if supervised_on else "score_0"
    pandas_result = PandasEntityMatching(params.copy()).fit(gt.copy()).transform(names.copy())
    polars_result = PolarsEntityMatching(params.copy()).fit(pl.from_pandas(gt)).transform(pl.from_pandas(names))
    pandas_result = pandas_result.rename(columns={score_col: "score"})
    polars_result = polars_result.to_pandas().rename(columns={score_col: "score"})
    polars_result = polars_result[["entity_id", "name", "gt_entity_id", "gt_name", "score"]]

    if with_no_matches:
        assert 0.5 < pandas_result["score"].isnull().mean() < 0.90
        assert 0.5 < polars_result["score"].isnull().mean() < 0.90
        pandas_scores = pandas_result["score"].fillna(0)
        polars_scores = polars_result["score"].fillna(0)
    else:
        pandas_scores = pandas_result["score"]
        polars_scores = polars_result["score"]
    assert pandas_scores.round(decimals=5).between(0, 1, inclusive="both").all()
    assert polars_scores.round(decimals=5).between(0, 1, inclusive="both").all()
    compare_name_matching_results(pandas_result, polars_result, ntop=ntop, min_score=min_score, eps=tolerance)


def test_pandas_polars_aggregation(kvk_dataset, supervised_model):
    data = kvk_dataset.copy()
    data["country"] = "PL"
    data["amount"] = 1.0
    data["uid"] = data.reset_index().index
    gt, names = split_dataset(data)
    names["uid"] = names.reset_index().index
    del names["id"]
    names["account"] = [i // 5 for i in range(len(names))]
    names["counterparty_account_count_distinct"] = 1.0
    params = {
        "name_only": False,
        "aggregation_layer": True,
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
                "cos_sim_lower_bound": 0.499,
                "num_candidates": 10,
            }
        ],
        "n_jobs": 1,
    }
    pandas_result = PandasEntityMatching(params.copy()).fit(gt.copy()).transform(names.copy())
    polars_result = (
        PolarsEntityMatching(params.copy()).fit(pl.from_pandas(gt)).transform(pl.from_pandas(names)).to_pandas()
    )
    pandas_best = (
        pandas_result[pandas_result.best_match][["account", "gt_entity_id", "agg_score"]]
        .rename(columns={"gt_entity_id": "pandas_id", "agg_score": "pandas_score"})
        .set_index("account", verify_integrity=True)
        .sort_index()
    )
    polars_best = (
        polars_result[["account", "gt_entity_id", "agg_score"]]
        .rename(columns={"gt_entity_id": "polars_id", "agg_score": "polars_score"})
        .set_index("account", verify_integrity=True)
        .sort_index()
    )
    result = pd.concat([pandas_best, polars_best], axis=1)
    assert not (result["pandas_id"].isnull() & result["polars_id"].notnull()).any()
    assert not (result["pandas_id"].notnull() & result["polars_id"].isnull()).any()
    assert (result["polars_score"] - result["pandas_score"]).dropna().abs().mean() < 0.00001
    assert (result["polars_id"].fillna(-1).astype(int) == result["pandas_id"].fillna(-1).astype(int)).mean() > 0.95


@pytest.mark.parametrize("use_mapping", [False, True])
def test_pandas_polars_sni(kvk_dataset, use_mapping):
    gt, names = split_dataset(kvk_dataset)
    gt["uid"] = gt.reset_index().index
    names["uid"] = names.reset_index().index
    params = {
        "name_only": True,
        "entity_id_col": "id",
        "uid_col": "uid",
        "name_col": "name",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 3, "mapping_func": (lambda x: x[::-1]) if use_mapping else None}],
    }
    pandas_result = PandasEntityMatching(params.copy()).fit(gt.copy()).transform(names.copy())
    polars_result = (
        PolarsEntityMatching(params.copy()).fit(pl.from_pandas(gt)).transform(pl.from_pandas(names)).to_pandas()
    )
    pandas_result = pandas_result.dropna(subset=["gt_uid"])
    polars_result = polars_result.dropna(subset=["gt_entity_id"])
    pandas_result["gt_entity_id"] = pandas_result["gt_entity_id"].astype(int)
    polars_result["gt_entity_id"] = polars_result["gt_entity_id"].astype(int)

    def add_idx(df):
        return df.sort_values(by=["entity_id", "gt_entity_id"]).set_index(
            ["entity_id", "gt_entity_id"], verify_integrity=True
        )

    pandas_result, polars_result = add_idx(pandas_result), add_idx(polars_result)
    if not use_mapping:
        assert len(pandas_result) / len(names) > 1.5
        assert len(polars_result) / len(names) > 1.5
    assert len(pandas_result) == len(polars_result)
    assert all(pandas_result.index == polars_result.index)
    pd.testing.assert_series_equal(pandas_result["score_0"], polars_result["score_0"])


def test_pandas_polars_supervised_features(kvk_training_dataset, supervised_model):
    gt, names = split_gt_and_names(kvk_training_dataset.head(10**4), gt_limit=50, names_limit=50)
    params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": True,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "return_sm_features": True,
        "n_jobs": 1,
    }
    pandas_result = PandasEntityMatching(params.copy()).fit(gt).transform(names)
    polars_result = PolarsEntityMatching(params.copy()).fit(pl.from_pandas(gt)).transform(pl.from_pandas(names))
    required = {"nm_score_feat_score_0", "nm_score_feat_norm_ed", "nm_score_feat_score_0_rank"}
    assert required <= set(pandas_result.columns)
    assert required <= set(polars_result.columns)
    assert list(pandas_result.filter(regex="^nm_score_feat").columns) == [
        column for column in polars_result.columns if column.startswith("nm_score_feat")
    ]
    features = list(pandas_result.filter(regex="^nm_score_feat").columns)
    keys = ["entity_id", "name", "gt_entity_id", "gt_name"]
    pandas_features = pandas_result[keys + features].sort_values(keys).reset_index(drop=True)
    polars_features = polars_result.select(keys + features).to_pandas().sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(pandas_features, polars_features, check_dtype=False, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name_only", [True, False])
@pytest.mark.parametrize("supervised_on", [True, False])
@pytest.mark.parametrize("aggregation_layer", [True, False])
@pytest.mark.parametrize("aggregation_method", ["max_frequency_nm_score", "mean_score"])
@pytest.mark.parametrize("only_nocandidate", [True, False])
@pytest.mark.parametrize("freq_col_bug", [True, False])
def test_pandas_polars_everything_no_candidates(  # noqa: PLR0917
    name_only, supervised_on, aggregation_layer, aggregation_method, only_nocandidate, freq_col_bug, supervised_model
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
        names = names.tail(3).copy()
    if freq_col_bug:
        names.loc[:, "account"] = 1
        names.loc[:, "counterparty_account_count_distinct"] = 0
        names.iloc[0, names.columns.get_loc("counterparty_account_count_distinct")] = None
        names.iloc[-1, names.columns.get_loc("counterparty_account_count_distinct")] = None
    params = {
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
            {"type": "sni", "window_length": 3},
        ],
        "n_jobs": 1,
    }
    pandas_result = PandasEntityMatching(params.copy()).fit(gt).transform(names)
    polars_result = PolarsEntityMatching(params.copy()).fit(pl.from_pandas(gt)).transform(pl.from_pandas(names))
    assert len(pandas_result) == polars_result.height
    columns = ["name", "account", "gt_entity_id", "score_0"]
    if supervised_on:
        columns.append("nm_score")
    if aggregation_layer:
        columns.append("agg_score")
    keys = ["name", "account", "gt_entity_id"]
    pandas_values = pandas_result[columns].sort_values(keys).reset_index(drop=True).convert_dtypes()
    polars_values = polars_result.select(columns).to_pandas().sort_values(keys).reset_index(drop=True).convert_dtypes()
    pd.testing.assert_frame_equal(pandas_values, polars_values, check_dtype=False, atol=1e-5, rtol=1e-5)
