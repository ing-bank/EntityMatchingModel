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

import pandas as pd
import pytest

from emm.helper import spark_installed
from emm.pipeline.pandas_entity_matching import PandasEntityMatching

from .test_pandas_em import split_gt_and_names

if spark_installed:
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


def test_increase_window_pandas(kvk_training_dataset):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:25]

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = PandasEntityMatching(em_params)
    p.fit(gt)

    p.increase_window_by_one_step()
    res = p.transform(names)
    assert len(res) == 327
    assert res["rank_0"].max() == 11
    assert res["rank_1"].max() == 3
    assert res["rank_1"].min() == -3


def test_decrease_window_pandas(kvk_training_dataset):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:25]

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = PandasEntityMatching(em_params)
    p.fit(gt)

    p.decrease_window_by_one_step()
    res = p.transform(names)
    assert len(res) == 227
    assert res["rank_0"].max() == 9
    assert res["rank_1"].max() == 1
    assert res["rank_1"].min() == -1


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_increase_window_spark(kvk_training_dataset, spark_session):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:25]

    sgt = spark_session.createDataFrame(gt)
    snames = spark_session.createDataFrame(names)

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = SparkEntityMatching(em_params)
    p.fit(sgt)

    p.increase_window_by_one_step()
    res = p.transform(snames)
    assert res.count() == 327


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_decrease_window_spark(kvk_training_dataset, spark_session):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:25]

    sgt = spark_session.createDataFrame(gt)
    snames = spark_session.createDataFrame(names)

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = SparkEntityMatching(em_params)
    p.fit(sgt)

    p.decrease_window_by_one_step()
    res = p.transform(snames)
    assert res.count() == 227


def test_create_name_pairs_pandas(kvk_training_dataset):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:25]

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = PandasEntityMatching(em_params)
    p.fit(gt)

    train = p.create_training_name_pairs(names, create_negative_sample_fraction=0.5, random_seed=42)

    assert isinstance(train, pd.DataFrame)
    assert len(train) == 277
    assert "correct" in train.columns
    assert "no_candidate" in train.columns
    assert "positive_set" in train.columns
    vc = train["positive_set"].value_counts().to_dict()
    assert vc[False] == 152
    assert vc[True] == 125


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_create_name_pairs_spark(kvk_training_dataset, spark_session):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:25]

    sgt = spark_session.createDataFrame(gt)
    snames = spark_session.createDataFrame(names)

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = SparkEntityMatching(em_params)
    p.fit(sgt)

    train = p.create_training_name_pairs(snames, create_negative_sample_fraction=0.5, random_seed=42)

    assert isinstance(train, pd.DataFrame)
    assert len(train) == 277
    assert "correct" in train.columns
    assert "no_candidate" in train.columns
    assert "positive_set" in train.columns
    vc = train["positive_set"].value_counts().to_dict()
    assert vc[False] == 152
    assert vc[True] == 125


def test_fit_classifier_pandas(kvk_training_dataset):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    train_names = names[:100]
    test_names = names[100:110]

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = PandasEntityMatching(em_params)
    p.fit(gt)

    assert len(p.model.steps) == 2
    assert "supervised" not in p.model.named_steps

    res_in = p.transform(test_names)
    assert "nm_score" not in res_in.columns
    assert len(res_in) == 123

    p.fit_classifier(train_names)

    assert len(p.model.steps) == 3
    assert "supervised" in p.model.named_steps

    res_sm = p.transform(test_names)
    assert "nm_score" in res_sm.columns


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_fit_classifier_spark(kvk_training_dataset, spark_session):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    train_names = names[:100]
    test_names = names[100:110]

    sgt = spark_session.createDataFrame(gt)
    strain_names = spark_session.createDataFrame(train_names)
    stest_names = spark_session.createDataFrame(test_names)

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = SparkEntityMatching(em_params)
    p.fit(sgt)

    assert len(p.model.stages) == 2

    p.fit_classifier(strain_names)
    assert len(p.model.stages) == 3

    res_sm = p.transform(stest_names)
    assert "nm_score" in res_sm.columns
    assert res_sm.count() == 123
