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

from emm.data.negative_data_creation import create_positive_negative_samples
from emm.data.prepare_name_pairs import prepare_name_pairs_pd
from emm.pipeline.pandas_entity_matching import PandasEntityMatching
from tests.unit import test_negative_sample_creation

pl = pytest.importorskip("polars")
PolarsEntityMatching = import_module("emm.pipeline.polars_entity_matching").PolarsEntityMatching
prepare_name_pairs_polars = import_module("emm.data.polars_prepare_name_pairs").prepare_name_pairs_polars
negative_samples_polars = import_module("emm.data.polars_prepare_name_pairs")._negative_samples
namepairs_df = test_negative_sample_creation.namepairs_df


@pytest.fixture
def data():
    ground_truth = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alpha company", "beta bank", "gamma business", "delta company", "epsilon bank"],
            "country": ["NL", "NL", "BE", "NL", "BE"],
            "category": ["a", "b", "c", "d", "e"],
        }
    )
    queries = ground_truth.with_columns(pl.Series("account", ["x", "x", "y", "z", "z"]), pl.lit(1).alias("freq"))
    return ground_truth, queries


def assert_candidates_match(actual, expected):
    actual = actual.to_pandas()
    common = sorted(set(actual.columns) & set(expected.columns))
    assert set(actual.columns) == set(expected.columns)
    order = [column for column in ("uid", "gt_uid") if column in common]
    pd.testing.assert_frame_equal(
        actual[common].sort_values(order, na_position="last").reset_index(drop=True).convert_dtypes(),
        expected[common].sort_values(order, na_position="last").reset_index(drop=True).convert_dtypes(),
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "indexers",
    [
        [{"type": "sni", "window_length": 5}],
        [{"type": "cosine_similarity", "tokenizer": "words", "num_candidates": 3}],
        [{"type": "cosine_similarity", "tokenizer": "words", "num_candidates": 3}, {"type": "sni", "window_length": 3}],
        [{"type": "naive"}],
    ],
)
def test_pipeline_candidates_match_pandas(data, indexers):
    gt, names = data
    params = {"name_only": False, "indexers": indexers, "n_jobs": 1, "carry_on_cols": ["category"]}
    actual = PolarsEntityMatching(params.copy()).fit(gt).transform(names)
    expected = PandasEntityMatching(params.copy()).fit(gt.to_pandas()).transform(names.to_pandas())
    assert_candidates_match(actual, expected)


def test_sni_pandas_window_examples():
    ids = list(range(10, 100, 10))
    ground_truth = pl.DataFrame({"id": ids, "name": [f"A{x:03d}" for x in ids]})
    model = PolarsEntityMatching(name_only=True, indexers=[{"type": "sni", "window_length": 5}]).fit(ground_truth)
    for name, expected in [
        ("A000", {"A010", "A020"}),
        ("A055", {"A040", "A050", "A060", "A070"}),
        ("A050", {"A030", "A040", "A050", "A060", "A070"}),
        ("A100", {"A080", "A090"}),
    ]:
        assert set(model.transform(pl.DataFrame({"name": [name], "id": [0]}))["gt_name"]) == expected


def test_no_indexers_series_and_column_validation(data):
    gt, _ = data
    model = PolarsEntityMatching(name_only=True, indexers=[])
    with pytest.raises(TypeError, match="not been trained"):
        model.transform(gt)
    model.fit(gt)
    assert model.transform(pl.Series("name", ["Alpha Company"]))["preprocessed"].to_list() == ["alpha company"]
    with pytest.raises(ValueError, match="Column"):
        model.transform(pl.DataFrame({"other": [1]}))
    with pytest.raises(TypeError, match="Polars DataFrame"):
        model.fit(gt.to_pandas())


@pytest.mark.parametrize("with_no_matches", [False, True])
def test_multiple_indexers_unmatched_and_carry_columns(with_no_matches):
    gt = pl.DataFrame({"name": ["abc", "b c d"], "id": [1, 2], "category": ["x", "y"]})
    names = pl.DataFrame({"name": ["abc a", "abd", "xyz"], "id": [10, 20, 30], "category": ["a", "b", "c"]})
    params = {
        "name_only": True,
        "preprocessor": "preprocess_name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1},
        ],
        "with_no_matches": with_no_matches,
        "carry_on_cols": ["category"],
        "n_jobs": 1,
    }
    actual = PolarsEntityMatching(params.copy()).fit(gt).transform(names)
    expected = PandasEntityMatching(params.copy()).fit(gt.to_pandas()).transform(names.to_pandas())
    assert_candidates_match(actual, expected)
    assert actual.filter(pl.col("entity_id") == 30).height == int(with_no_matches)
    assert {"gt_category", "category", "cossim_w1", "cossim_n1"} <= set(actual.columns)


@pytest.mark.parametrize("method", ["mean_score", "max_frequency_nm_score"])
def test_aggregation_created_or_added_later(data, method):
    gt, names = data
    names = names.drop("id")
    params = {
        "name_only": False,
        "freq_col": "freq",
        "indexers": [{"type": "sni", "window_length": 5}],
        "aggregation_method": method,
    }
    configured = PolarsEntityMatching({**params, "aggregation_layer": True}).fit(gt).transform(names)
    model = PolarsEntityMatching(params.copy()).fit(gt)
    model.add_aggregation_layer(freq_col="freq", aggregation_method=method)
    later = model.transform(names)
    assert configured.equals(later)
    assert configured.height == 3
    assert {"agg_score", "best_rank", "best_match"} <= set(configured.columns)


@pytest.mark.parametrize("method", ["mean_score", "max_frequency_nm_score"])
def test_aggregation_matches_pandas_pipeline(data, method):
    gt, names = data
    names = names.drop("id").rename({"freq": "counterparty_account_count_distinct"})
    params = {
        "name_only": False,
        "indexers": [{"type": "sni", "window_length": 5}],
        "aggregation_layer": True,
        "aggregation_method": method,
    }
    actual = PolarsEntityMatching(params.copy()).fit(gt).transform(names)
    expected = PandasEntityMatching(params.copy()).fit(gt.to_pandas()).transform(names.to_pandas())
    assert_candidates_match(actual, expected)


def test_custom_columns_and_shared_id(data):
    gt, names = data
    gt = gt.rename({"id": "key", "name": "title"})
    names = names.rename({"id": "key", "name": "title", "account": "acct", "freq": "count"})
    model = PolarsEntityMatching(
        name_col="title",
        entity_id_col="key",
        uid_col="key",
        account_col="acct",
        freq_col="count",
        name_only=True,
        aggregation_layer=True,
        indexers=[{"type": "sni", "window_length": 3}],
    ).fit(gt)
    result = model.transform(names)
    assert {"account", "entity_id", "gt_entity_id", "agg_score"} <= set(result.columns)


def test_training_pairs_with_custom_uid_column(data):
    gt, names = data
    gt = gt.with_columns(pl.col("id").alias("row_key"))
    names = names.with_columns(pl.col("id").alias("row_key"))
    model = PolarsEntityMatching(name_only=True, uid_col="row_key", indexers=[{"type": "sni", "window_length": 5}]).fit(
        gt
    )
    pairs = model.create_training_name_pairs(names)
    assert {"uid", "gt_uid", "correct", "no_candidate"} <= set(pairs.columns)


def test_window_changes_and_training_pairs_match_pandas(data):
    gt, names = data
    params = {
        "name_only": True,
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "num_candidates": 3},
            {"type": "sni", "window_length": 3},
        ],
        "n_jobs": 1,
    }
    actual_model = PolarsEntityMatching(params.copy()).fit(gt)
    expected_model = PandasEntityMatching(params.copy()).fit(gt.to_pandas())
    for fraction in (0, 0.4):
        actual = actual_model.create_training_name_pairs(names, create_negative_sample_fraction=fraction)
        expected = expected_model.create_training_name_pairs(
            names.to_pandas(), create_negative_sample_fraction=fraction
        )
        assert_candidates_match(actual, expected)
    selector = actual_model.model.named_steps["candidate_selection"]
    assert selector.indexers[0].num_candidates == 3
    assert selector.indexers[1].window_length == 3
    actual_model.increase_window_by_one_step()
    assert (selector.indexers[0].num_candidates, selector.indexers[1].window_length) == (4, 5)
    actual_model.decrease_window_by_one_step()
    assert (selector.indexers[0].num_candidates, selector.indexers[1].window_length) == (3, 3)


def test_negative_reranking_matches_pandas_sample(namepairs_df):
    indexed = namepairs_df.assign(test_row=np.arange(len(namepairs_df)))
    expected = create_positive_negative_samples(indexed)
    actual = negative_samples_polars(pl.from_pandas(indexed), "uid", "correct", "positive_set")
    columns = ["test_row", "rank_0", "rank_1", "rank_2"]
    pd.testing.assert_frame_equal(
        actual.select(columns).to_pandas().sort_values("test_row").reset_index(drop=True),
        expected[columns].sort_values("test_row").reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize("drop_duplicates", [False, True])
@pytest.mark.parametrize("fraction", [0, 0.5])
def test_prepare_training_pairs_matches_pandas(data, drop_duplicates, fraction):
    gt, names = data
    params = {"name_only": True, "indexers": [{"type": "sni", "window_length": 5}]}
    pipeline = PandasEntityMatching(params).fit(gt.to_pandas())
    candidates = pipeline.transform(names.to_pandas()).drop(columns=["name", "gt_name"])
    expected = prepare_name_pairs_pd(
        candidates.copy(), drop_duplicate_candidates=drop_duplicates, create_negative_sample_fraction=fraction
    )
    actual = prepare_name_pairs_polars(
        pl.from_pandas(candidates), drop_duplicate_candidates=drop_duplicates, create_negative_sample_fraction=fraction
    )
    assert_candidates_match(actual, expected)


def test_training_pairs_restore_names_without_candidates():
    candidates = pl.DataFrame(
        {
            "uid": [0, 1],
            "gt_uid": [1.0, float("nan")],
            "entity_id": [1, 2],
            "gt_entity_id": [1.0, float("nan")],
            "preprocessed": ["alpha", "beta"],
            "gt_preprocessed": ["alpha", None],
            "rank_0": [1.0, float("nan")],
            "score_0": [1.0, float("nan")],
            "positive_set": [False, True],
        }
    )
    expected = prepare_name_pairs_pd(candidates.to_pandas(), create_negative_sample_fraction=0.5)
    actual = prepare_name_pairs_polars(candidates, create_negative_sample_fraction=0.5)
    assert_candidates_match(actual, expected)
    assert actual["no_candidate"].to_list() == [False, True]


def test_train_classifier_features_add_model_and_top_n(data, monkeypatch):
    gt, names = data

    def forbid_pandas(*args, **kwargs):
        msg = "Polars pipeline must not convert to pandas"
        raise AssertionError(msg)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", forbid_pandas)
    params = {"name_only": True, "indexers": [{"type": "sni", "window_length": 5}]}
    model = PolarsEntityMatching(params.copy()).fit(gt)
    model.fit_classifier(names, n_jobs=1)
    predicted = model.transform(names)
    assert "nm_score" in predicted.columns
    assert predicted.filter(pl.col("best_match")).height == names.height
    assert model.transform(names, top_n=1).height == names.height
    model.set_return_sm_features(True)
    assert "nm_score_feat_norm_ed" in model.transform(names).columns
    trained = model.supervised_models["nm_score"]["model"]
    another = PolarsEntityMatching(params.copy()).fit(gt)
    another.add_supervised_model(model=trained, return_features=True)
    assert another.transform(names)["nm_score"].to_list() == pytest.approx(predicted["nm_score"].to_list())


def test_classifier_training_with_aggregation_and_existing_pairs(data, tmp_path):
    gt, names = data
    names = names.with_columns(pl.col("freq").alias("counterparty_account_count_distinct"))
    params = {
        "name_only": True,
        "indexers": [{"type": "sni", "window_length": 5}],
        "aggregation_layer": True,
        "aggregation_method": "mean_score",
    }
    model = PolarsEntityMatching(params.copy()).fit(gt)
    pair_source = PolarsEntityMatching({**params, "aggregation_layer": False}).fit(gt)
    pairs = pair_source.create_training_name_pairs(names)
    model.fit_classifier(train_name_pairs=pairs, n_jobs=1)
    result = model.transform(names)
    assert {"nm_score", "agg_score", "best_match"} <= set(result.columns)
    path = str(tmp_path / "trained.joblib")
    model.save(path)
    assert PolarsEntityMatching.load(path).transform(names).equals(result)


def test_pandas_trained_classifier_matches_pandas_pipeline(data, supervised_model):
    gt, names = data
    params = {
        "name_only": True,
        "indexers": [{"type": "sni", "window_length": 5}],
        "supervised_on": True,
        "supervised_model_dir": supervised_model[2].parent,
        "supervised_model_filename": supervised_model[2].name,
        "return_sm_features": True,
    }
    actual = PolarsEntityMatching(params.copy()).fit(gt).transform(names)
    expected = PandasEntityMatching(params.copy()).fit(gt.to_pandas()).transform(names.to_pandas())
    assert_candidates_match(actual, expected)


def test_classifier_diagnostics_and_invalid_training(data, caplog):
    gt, names = data
    names = names.with_columns(pl.when(pl.col("id") == 5).then(99).otherwise(pl.col("id")).alias("id"))
    model = PolarsEntityMatching(name_only=True, indexers=[{"type": "sni", "window_length": 5}]).fit(gt)
    with pytest.raises(TypeError, match="callable"):
        model.fit_classifier(names, train_function=None)
    model.fit_classifier(gt, n_jobs=1)
    with caplog.at_level("INFO", logger="emm"):
        model.test_classifier(names, test_gt=gt)
    assert "AUC of the supervised model" in caplog.text


def test_serialization_round_trip_and_failure_restores_state(data, tmp_path):
    gt, names = data
    model = PolarsEntityMatching(name_only=True, indexers=[{"type": "sni", "window_length": 3}]).fit(
        gt, copy_ground_truth=True
    )
    expected = model.transform(names)
    path = str(tmp_path / "model.joblib")
    model.save(path)
    assert model.ground_truth_df is not None
    assert model.transform(names).equals(expected)
    loaded = PolarsEntityMatching.load(path)
    assert loaded.transform(names).equals(expected)

    def fail_dump(*args):
        msg = "serialization failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="serialization failed"):
        model.save(path, dump_func=fail_dump)
    assert model.ground_truth_df is not None
    assert model.transform(names).equals(expected)
