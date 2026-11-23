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

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from emm.helper.io import load_pickle

pl = pytest.importorskip("polars")

PandasSupervisedLayerTransformer = import_module(
    "emm.supervised_model.pandas_supervised_model"
).PandasSupervisedLayerTransformer
PolarsSupervisedLayerTransformer = import_module(
    "emm.supervised_model.polars_supervised_model"
).PolarsSupervisedLayerTransformer
PolarsFeatureExtractor = import_module("emm.features.polars_feature_extractor").PolarsFeatureExtractor
create_new_model_pipeline = import_module("emm.supervised_model.base_supervised_model").create_new_model_pipeline
assert_polars_frame_equal = import_module("polars.testing").assert_frame_equal


@pytest.fixture()
def candidates():
    return pl.DataFrame(
        {
            "uid": [0, 0, 1, 1, 2, 3],
            "gt_uid": [10, 11, 12, 13, None, 14],
            "preprocessed": ["alpha ltd", "alpha ltd", "beta inc", "beta inc", "none", "gamma ltd"],
            "gt_preprocessed": ["alpha ltd", "other ltd", "beta inc", "other inc", "", "gamma ltd"],
            "score_0": [0.95, 0.3, 0.9, 0.2, None, 0.95],
        }
    )


@pytest.fixture()
def trained_model(candidates):
    pairs = candidates.filter(pl.col("gt_uid").is_not_null())
    model = create_new_model_pipeline(xgb_args={"n_estimators": 8, "n_jobs": 1, "min_child_weight": 0})
    model.fit(pairs.to_pandas(), np.array([1, 0, 1, 0, 1]))
    return model


def test_calc_features_in_polars_supervised_layer(sample_sm_input, supervised_model):
    sm = load_pickle(supervised_model[2].name, supervised_model[2].parent)
    models = {"nm_score": {"model": sm, "enable": True}}
    expected = PandasSupervisedLayerTransformer(models, return_features=True).transform(sample_sm_input.copy())
    actual = PolarsSupervisedLayerTransformer(models, return_features=True).transform(pl.from_pandas(sample_sm_input))

    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert {
        "score_0",
        "nm_score",
        "nm_score_feat_abbr_match",
        "nm_score_feat_ratio",
        "nm_score_feat_score_0_rank",
    } <= set(actual.columns)
    assert all(actual[column].null_count() == 0 for column in actual.columns if column.startswith("nm_score_feat"))


@pytest.mark.parametrize("native_features", [False, True])
def test_score_and_features_match_pandas(candidates, trained_model, native_features):
    model = trained_model
    if native_features:
        feature = PolarsFeatureExtractor(
            score_columns=["score_0"], vocabulary=trained_model.named_steps["feat"].vocabulary
        )
        model = Pipeline([("feat", feature), ("classifier", trained_model.named_steps["classifier"])])
    expected_models = {"nm_score": {"model": trained_model, "enable": True}}
    models = {"nm_score": {"model": model, "enable": True}}
    original = candidates.clone()
    actual = PolarsSupervisedLayerTransformer(models, return_features=True).transform(candidates)
    expected = PandasSupervisedLayerTransformer(expected_models, return_features=True).transform(candidates.to_pandas())

    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert candidates.equals(original)
    assert {"nm_score", "best_match", "best_rank", "nm_score_feat_abbr_match", "nm_score_feat_score_0_rank"} <= set(
        actual.columns
    )
    assert actual.filter(pl.col("gt_uid").is_null())["nm_score"].to_list() == [None]


def test_disabled_model_returns_features_without_scoring(candidates):
    model = create_new_model_pipeline(xgb_args={"n_estimators": 4, "n_jobs": 1})
    models = {"X": {"model": model, "enable": False}}
    actual = PolarsSupervisedLayerTransformer(models, return_features=True).transform(candidates)
    expected = PandasSupervisedLayerTransformer(models, return_features=True).transform(candidates.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
    assert "X_feat_abbr_match" in actual.columns
    assert "X" not in actual.columns


def test_untrained_pandas_feature_vocabulary_is_retained(candidates):
    model = create_new_model_pipeline()
    transformer = PolarsSupervisedLayerTransformer({"X": {"model": model, "enable": False}}, return_features=True)
    transformer.transform(candidates)
    feature = model.named_steps["feat"]
    vocabulary = feature.vocabulary
    assert feature._fitted
    transformer.transform(candidates.head(1))
    assert feature.vocabulary == vocabulary


def test_scoring_fits_pandas_feature_step(candidates, trained_model):
    feature = trained_model.named_steps["feat"]
    feature._fitted = False
    transformer = PolarsSupervisedLayerTransformer({"nm_score": {"model": trained_model, "enable": True}})
    transformer.transform(candidates)
    assert feature._fitted


@pytest.mark.parametrize("native_features", [False, True])
def test_categorical_features_with_trained_xgboost(candidates, native_features, monkeypatch):
    pairs = candidates.filter(pl.col("gt_uid").is_not_null())
    trained = create_new_model_pipeline(
        feature_args={"with_legal_entity_forms_match": True},
        xgb_args={"n_estimators": 8, "n_jobs": 1, "min_child_weight": 0},
    )
    trained.fit(pairs.to_pandas(), np.array([1, 0, 1, 0, 1]))
    expected = PandasSupervisedLayerTransformer(
        {"nm_score": {"model": trained, "enable": True}}, return_features=True
    ).transform(pairs.to_pandas())
    model = trained
    if native_features:
        feature = PolarsFeatureExtractor(
            score_columns=["score_0"],
            vocabulary=trained.named_steps["feat"].vocabulary,
            with_legal_entity_forms_match=True,
        )
        model = Pipeline([("feat", feature), ("classifier", trained.named_steps["classifier"])])

    def fail_conversion(*args, **kwargs):
        msg = "supervised scoring must not convert Polars to pandas"
        raise AssertionError(msg)

    transformer = PolarsSupervisedLayerTransformer({"nm_score": {"model": model, "enable": True}}, return_features=True)
    with monkeypatch.context() as patch:
        patch.setattr(pl.DataFrame, "to_pandas", fail_conversion)
        actual = transformer.transform(pairs)
    pd.testing.assert_frame_equal(
        actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True), check_categorical=False
    )
    assert expected["nm_score"].nunique() > 1


class ScoreModel:
    def predict_proba(self, X):
        score = X["score"].to_numpy()
        return np.column_stack((1 - score, score))


def test_multiple_models_and_missing_candidates():
    pairs = pl.DataFrame({"uid": [0, 0, 1, 2], "gt_uid": [1.0, None, 3.0, float("nan")], "score": [0.3, 0.5, 0.8, 0.4]})
    models = {
        "first": {"model": ScoreModel(), "enable": True},
        "second": {"model": ScoreModel(), "enable": True},
        "disabled": {"model": ScoreModel(), "enable": False},
    }
    actual = PolarsSupervisedLayerTransformer(models, best_score_col="second").transform(pairs)
    expected = PandasSupervisedLayerTransformer(models, best_score_col="second").transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert "disabled" not in actual.columns


def test_no_candidates_and_existing_scores():
    model = {"nm_score": {"model": ScoreModel(), "enable": True}}
    transformer = PolarsSupervisedLayerTransformer(model)
    pairs = pl.DataFrame({"uid": [0, 1], "gt_uid": [None, None], "nm_score": [0.7, 0.4]})
    actual = transformer.transform(pairs)
    expected = PandasSupervisedLayerTransformer(model).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert actual["nm_score"].to_list() == [0.0, 0.0]
    assert actual["best_match"].to_list() == [False, False]


def test_partially_missing_candidates_preserve_existing_scores():
    pairs = pl.DataFrame({"uid": [0, 1], "gt_uid": [1, None], "score": [0.8, 0.1], "nm_score": [0.0, 0.4]})
    models = {"nm_score": {"model": ScoreModel(), "enable": True}}
    actual = PolarsSupervisedLayerTransformer(models).transform(pairs)
    expected = PandasSupervisedLayerTransformer(models).transform(pairs.to_pandas())
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert actual["nm_score"].to_list() == [0.8, 0.4]


def test_empty_candidates():
    pairs = pl.DataFrame(schema={"uid": pl.Int64, "gt_uid": pl.Int64})
    transformer = PolarsSupervisedLayerTransformer({"nm_score": {"model": ScoreModel(), "enable": True}})
    actual = transformer.transform(pairs)
    assert actual.is_empty()
    assert {"nm_score", "best_rank", "best_match"} <= set(actual.columns)


@pytest.mark.parametrize("custom_sort", [False, True])
def test_best_score_ties_missing_groups_and_custom_sort(custom_sort):
    pairs = pl.DataFrame(
        {
            "uid": [1.0, 1.0, 0.0, 0.0, 0.0, None, float("nan")],
            "gt_uid": [1, 3, 1, 2, 3, 4, 5],
            "nm_score": [0.5, 0.5, 0.0, None, 0.8, 0.7, 0.6],
            "order": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    models = {"nm_score": {"model": ScoreModel(), "enable": True}}
    kwargs = {"group_cols": ["uid"]}
    if custom_sort:
        kwargs.update({"sort_cols": ["order"], "sort_asc": [False]})
    actual = PolarsSupervisedLayerTransformer(models).select_best_score(pairs, **kwargs)
    expected = PandasSupervisedLayerTransformer(models).select_best_score(pairs.to_pandas(), **kwargs)
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    if not custom_sort:
        assert actual.filter(pl.col("uid") == 1)["gt_uid"].to_list()[0] == 3


def test_native_scoring_does_not_convert_to_pandas(candidates, trained_model, monkeypatch):
    def fail_conversion(*args, **kwargs):
        msg = "supervised scoring must not convert Polars to pandas"
        raise AssertionError(msg)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", fail_conversion)
    transformer = PolarsSupervisedLayerTransformer(
        {"nm_score": {"model": trained_model, "enable": True}}, return_features=True
    )
    assert transformer.transform(candidates).height == candidates.height


@pytest.mark.parametrize("native_features", [False, True])
def test_trained_supervised_layer_serialization(tmp_path, candidates, trained_model, native_features):
    model = trained_model
    if native_features:
        feature = PolarsFeatureExtractor(
            score_columns=["score_0"], vocabulary=trained_model.named_steps["feat"].vocabulary
        )
        model = Pipeline([("feat", feature), ("classifier", trained_model.named_steps["classifier"])])
    transformer = PolarsSupervisedLayerTransformer({"nm_score": {"model": model, "enable": True}}, return_features=True)
    expected = transformer.transform(candidates)
    path = tmp_path / "supervised.joblib"
    joblib.dump(transformer, path)
    assert path.is_file()
    restored = joblib.load(path)

    assert restored.best_score_col == "nm_score"
    assert restored.return_features
    assert restored.supervised_models["nm_score"]["enable"]
    assert_polars_frame_equal(restored.transform(candidates), expected)


def test_disabled_feature_model_serialization_preserves_vocabulary(tmp_path, candidates):
    model = create_new_model_pipeline()
    transformer = PolarsSupervisedLayerTransformer({"X": {"model": model, "enable": False}}, return_features=True)
    transformer.transform(candidates)
    vocabulary = model.named_steps["feat"].vocabulary
    subset = candidates.head(1)
    expected = transformer.transform(subset)
    path = tmp_path / "features.joblib"
    joblib.dump(transformer, path)
    restored = joblib.load(path)

    assert restored.supervised_models["X"]["model"].named_steps["feat"].vocabulary == vocabulary
    assert_polars_frame_equal(restored.transform(subset), expected)


@pytest.mark.parametrize("model_index", [0, 2, 4])
def test_serialization_with_repository_supervised_models(tmp_path, candidates, supervised_model, model_index):
    model_path = supervised_model[model_index]
    model = load_pickle(model_path.name, model_path.parent)
    pairs = candidates.with_columns(pl.lit("NL").alias("country"), pl.lit("NL").alias("gt_country"))
    models = {"nm_score": {"model": model, "enable": True}}
    expected = PandasSupervisedLayerTransformer(models, return_features=True).transform(pairs.to_pandas())
    transformer = PolarsSupervisedLayerTransformer(models, return_features=True)
    path = tmp_path / "supervised.joblib"
    joblib.dump(transformer, path)
    restored = joblib.load(path)

    actual = restored.transform(pairs)
    pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True), expected.reset_index(drop=True))
    assert_polars_frame_equal(actual, transformer.transform(pairs))


def test_transform_lifecycle_and_validation(candidates):
    transformer = PolarsSupervisedLayerTransformer({})
    assert transformer.fit(candidates) is transformer
    assert transformer.fit_transform(candidates) is None
    assert transformer.transform(None) is None
    assert transformer.transform(candidates).equals(candidates)
    with pytest.raises(TypeError, match="expected a Polars DataFrame"):
        transformer.transform(candidates.to_pandas())


def test_custom_pandas_rank_functions_are_rejected(candidates):
    model = create_new_model_pipeline()
    model.named_steps["feat"].rank_features["rank"] = lambda *args: None
    transformer = PolarsSupervisedLayerTransformer({"X": {"model": model, "enable": False}}, return_features=True)
    with pytest.raises(ValueError, match="custom pandas rank or difference features"):
        transformer.transform(candidates)
