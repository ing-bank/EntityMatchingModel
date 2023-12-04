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

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from emm.helper import spark_installed
from emm.pipeline.pandas_entity_matching import PandasEntityMatching

if spark_installed:
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(("spark_dump", "spark_load"), [(False, False), (True, True)])
def test_serialization(spark_dump, spark_load, spark_session, kvk_dataset, supervised_model):
    df = kvk_dataset.head(200).rename(columns={"name": "custom_name", "id": "custom_id"})
    gt, names = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    if spark_dump or spark_load:
        sdf_gt = spark_session.createDataFrame(gt)
        sdf_names = spark_session.createDataFrame(names)
    with tempfile.TemporaryDirectory() as tmpdir:
        emo_fn = os.path.join(tmpdir, "emo.joblib")
        em_params = {
            "name_col": "custom_name",
            "entity_id_col": "custom_id",
            "supervised_on": True,
            "name_only": True,
            "aggregation_layer": False,
            "supervised_model_dir": supervised_model[2].parent,
            "supervised_model_filename": supervised_model[2].name,
            "indexers": [{"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1}],
        }
        m = SparkEntityMatching(em_params) if spark_dump else PandasEntityMatching(em_params)
        m.fit(sdf_gt if spark_dump else gt)
        res = m.transform(sdf_names if spark_dump else names)
        if spark_dump:
            res = res.toPandas().sort_values(by=["uid", "best_rank"])
            res.reset_index(drop=True, inplace=True)
        # make sure that there are a lot of matches
        assert res["gt_entity_id"].notnull().mean() > 0.9
        m.save(emo_fn)
        assert os.path.exists(emo_fn), "missing serialized model file"

        m2 = SparkEntityMatching.load(emo_fn) if spark_load else PandasEntityMatching.load(emo_fn)
        assert m2.parameters["name_col"] == "custom_name"
        assert m2.parameters["supervised_on"]
        if not spark_load:
            assert "supervised" in m2.pipeline.named_steps
        res2 = m2.transform(sdf_names if spark_load else names)

        if spark_load:
            res2 = res2.toPandas().sort_values(by=["uid", "best_rank"])
            res2.reset_index(drop=True, inplace=True)

        # the results should the be the exactly the same
        if spark_load == spark_dump:
            pd.testing.assert_frame_equal(res, res2)
        else:
            # simplified check, at least the number of results should be the same
            assert len(res) == len(res2)
            assert res["nm_score"].sum() == pytest.approx(res2["nm_score"].sum())


def test_serialization_of_full_model_pandas(kvk_dataset, supervised_model):
    df = kvk_dataset.head(200).rename(columns={"name": "custom_name", "id": "custom_id"})
    gt, names = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    with tempfile.TemporaryDirectory() as tmpdir:
        emo_fn = os.path.join(tmpdir, "emo_full.joblib")
        em_params = {
            "name_col": "custom_name",
            "entity_id_col": "custom_id",
            "supervised_on": True,
            "name_only": True,
            "aggregation_layer": False,
            "supervised_model_dir": supervised_model[2].parent,
            "supervised_model_filename": supervised_model[2].name,
            "indexers": [{"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1}],
        }
        m = PandasEntityMatching(em_params)
        m.fit(gt, copy_ground_truth=True)
        res = m.transform(names)
        assert m.pipeline.named_steps["candidate_selection"].gt is not None
        # make sure that there are a lot of matches
        assert res["gt_entity_id"].notnull().mean() > 0.9
        m.save(emo_fn)
        assert os.path.exists(emo_fn), "missing serialized model file"

        m2 = PandasEntityMatching.load(emo_fn, override_parameters={"name_col": "custom_name2"})
        assert m2.parameters["name_col"] == "custom_name2"
        assert m2.parameters["supervised_on"]
        # no fitting! ground_truth has been stored.
        names2 = names.rename(columns={"custom_name": "custom_name2"})
        res2 = m2.transform(names2)

        # the results should the be the exactly the same
        pd.testing.assert_frame_equal(res, res2)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_serialization_of_full_model_pandas_to_spark(spark_session, kvk_dataset, supervised_model):
    df = kvk_dataset.head(200).rename(columns={"name": "custom_name", "id": "custom_id"})
    gt, names = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    gt2 = spark_session.createDataFrame(gt)
    names2 = spark_session.createDataFrame(names)

    with tempfile.TemporaryDirectory():
        em_params = {
            "name_col": "custom_name",
            "entity_id_col": "custom_id",
            "supervised_on": True,
            "name_only": True,
            "aggregation_layer": False,
            "supervised_model_dir": supervised_model[2].parent,
            "supervised_model_filename": supervised_model[2].name,
            "indexers": [{"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1}],
        }
        m = PandasEntityMatching(em_params)
        m.fit(gt, copy_ground_truth=True)
        res = m.transform(names)
        res = res.sort_index(axis=1)
        res = res.sort_values(by=["uid", "best_rank"])
        res.reset_index(drop=True, inplace=True)

        m2 = SparkEntityMatching(em_params)
        m2.fit(gt2, copy_ground_truth=True)
        res2 = m2.transform(names2)
        res2 = res2.toPandas()
        res2 = res2.sort_index(axis=1)
        res2 = res2.sort_values(by=["uid", "best_rank"])
        res2.reset_index(drop=True, inplace=True)

        # simplified check, at least the number of results should be the same
        assert len(res) == len(res2)
        assert res["nm_score"].sum() == pytest.approx(res2["nm_score"].sum())


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_serialization_spark_save_load(spark_session, kvk_dataset, supervised_model):
    df = kvk_dataset.head(200).rename(columns={"name": "custom_name", "id": "custom_id"})
    gt, names = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    gt2 = spark_session.createDataFrame(gt)
    names2 = spark_session.createDataFrame(names)

    with tempfile.TemporaryDirectory() as tmpdir:
        emo_fn = os.path.join(tmpdir, "emo_full.joblib")
        em_params = {
            "name_col": "custom_name",
            "entity_id_col": "custom_id",
            "supervised_on": True,
            "name_only": True,
            "aggregation_layer": False,
            "supervised_model_dir": supervised_model[2].parent,
            "supervised_model_filename": supervised_model[2].name,
            "indexers": [{"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1}],
        }
        m = SparkEntityMatching(em_params)
        m.fit(gt2, copy_ground_truth=True)
        res = m.transform(names2)
        res = res.toPandas()

        m.write().overwrite().save(emo_fn)
        assert os.path.exists(emo_fn), "missing serialized model file"

        m2 = SparkEntityMatching.load(emo_fn)
        res2 = m2.transform(names2)
        res2 = res2.toPandas()

        # simplified check, at least the number of results should be the same
        assert len(res) == len(res2)
        assert res["nm_score"].sum() == pytest.approx(res2["nm_score"].sum())


def test_serialization_pandas_save_load(kvk_dataset):
    df = kvk_dataset.head(200).rename(columns={"name": "custom_name", "id": "custom_id"})
    gt, names = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]

    with tempfile.TemporaryDirectory() as tmpdir:
        emo_fn = Path(tmpdir) / "emo_pandas_full.joblib"
        em_params = {
            "name_col": "custom_name",
            "entity_id_col": "custom_id",
            "name_only": True,
            "supervised_on": False,
            "aggregation_layer": False,
            "indexers": [
                {"type": "cosine_similarity", "tokenizer": "characters", "ngram": 1},
                {"type": "sni", "window_length": 1},
            ],
        }
        m = PandasEntityMatching(em_params)
        m.fit(gt, copy_ground_truth=True)
        res = m.transform(names)

        assert m.model.steps[1][1].gt is not None
        # make sure that there are a lot of matches
        assert res["gt_entity_id"].notnull().mean() > 0.9

        m.save(str(emo_fn))
        assert Path(emo_fn).exists(), "missing serialized model file"

        m2 = PandasEntityMatching.load(str(emo_fn))
        assert m2.parameters["name_col"] == "custom_name"
        res2 = m2.transform(names)

        # the results should the be the exactly the same
        pd.testing.assert_frame_equal(res, res2)
