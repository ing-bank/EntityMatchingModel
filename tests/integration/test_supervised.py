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
from emm.helper.io import load_pickle
from emm.pipeline.pandas_entity_matching import PandasEntityMatching
from emm.supervised_model.base_supervised_model import calc_features_from_sm, features_schema_from_sm
from emm.supervised_model.pandas_supervised_model import PandasSupervisedLayerTransformer

from .test_pandas_em import split_gt_and_names

if spark_installed:
    from emm.pipeline.spark_entity_matching import SparkEntityMatching
    from emm.supervised_model.spark_supervised_model import SparkSupervisedLayerEstimator


@pytest.fixture()
def sample_sm_input():
    return pd.DataFrame(
        {
            "uid": [1, 2],
            "gt_uid": [10, 11],
            "entity_id": [1000, 1001],
            "gt_entity_id": [2000, 2001],
            "name": ["Abc!", "iNg BV"],
            "gt_name": ["Xyz   Abc", "ING limited"],
            "preprocessed": ["abc", "ing bv"],
            "gt_preprocessed": ["xyz abc", "ing ltd"],
            "score_0": [0.8, 0.99],
        }
    )


def test_calc_features_helper_function(sample_sm_input, supervised_model):
    sm = load_pickle(supervised_model[2].name, supervised_model[2].parent)
    feat = calc_features_from_sm(sm, sample_sm_input)
    assert {"score_0", "abbr_match", "ratio", "score_0_rank"} <= set(feat.columns)
    assert feat["partial_ratio"].max() == 100
    assert feat["norm_ed"].max() >= 4


def test_features_schema_helper_function(supervised_model):
    sm = load_pickle(supervised_model[2].name, supervised_model[2].parent)
    schema = features_schema_from_sm(sm)
    assert schema[0] == ("score_0", np.float32)
    assert schema[1] == ("abbr_match", np.int8)
    assert len(schema) > 10


def test_calc_features_in_pandas_supervised_layer(sample_sm_input, supervised_model):
    sm = load_pickle(supervised_model[2].name, supervised_model[2].parent)
    tr = PandasSupervisedLayerTransformer({"nm_score": {"model": sm, "enable": True}}, return_features=True)
    res = tr.transform(sample_sm_input)
    # standard columns from supervised layer
    assert {"score_0", "nm_score"} <= set(res.columns)
    # features
    assert {"nm_score_feat_abbr_match", "nm_score_feat_ratio", "nm_score_feat_score_0_rank"} <= set(res.columns)
    assert all(res.filter(regex="^nm_score_feat").isnull().sum() == 0)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_calc_features_in_spark_supervised_layer(spark_session, sample_sm_input, supervised_model):
    sm = load_pickle(supervised_model[2].name, supervised_model[2].parent)
    estimator = SparkSupervisedLayerEstimator({"nm_score": {"model": sm, "enable": True}}, return_features=True)
    model = estimator.fit(dataset=None)
    res = model.transform(spark_session.createDataFrame(sample_sm_input))
    # standard columns from supervised layer
    assert {"score_0", "nm_score"} <= set(res.columns)
    # features
    assert {"nm_score_feat_abbr_match", "nm_score_feat_ratio", "nm_score_feat_score_0_rank"} <= set(res.columns)
    res_pd = res.toPandas()
    assert all(res_pd.filter(regex="^nm_score_feat").isnull().sum() == 0)


def test_return_sm_features_pandas(kvk_training_dataset):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:10]

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": True,
        "return_sm_features": True,
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = PandasEntityMatching(em_params)
    p.fit(gt)
    res = p.transform(names)

    features = [
        "X_feat_abbr_match",
        "X_feat_abs_len_diff",
        "X_feat_len_ratio",
        "X_feat_token_sort_ratio",
        "X_feat_token_set_ratio",
        "X_feat_partial_ratio",
        "X_feat_w_ratio",
        "X_feat_ratio",
        "X_feat_name_cut",
        "X_feat_norm_ed",
        "X_feat_norm_jaro",
        "X_feat_very_common_hit",
        "X_feat_common_hit",
        "X_feat_rare_hit",
        "X_feat_very_common_miss",
        "X_feat_common_miss",
        "X_feat_rare_miss",
        "X_feat_n_overlap_words",
        "X_feat_ratio_overlap_words",
        "X_feat_num_word_difference",
        "X_feat_score_0_rank",
        "X_feat_score_0_top2_dist",
        "X_feat_score_0_dist_to_max",
        "X_feat_score_0_dist_to_min",
        "X_feat_score_0_ptp",
        "X_feat_score_0_diff_to_next",
        "X_feat_score_0_diff_to_prev",
    ]

    assert len(res) == 118
    assert all(feat in res.columns for feat in features)
    assert (res["X_feat_norm_jaro"] > 0).all()


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_return_sm_features_spark(kvk_training_dataset, spark_session):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:10]

    sgt = spark_session.createDataFrame(gt)
    snames = spark_session.createDataFrame(names)

    em_params = {
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": True,
        "return_sm_features": True,
        "indexers": [
            {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1},
            {"type": "sni", "window_length": 5},
        ],
    }
    p = SparkEntityMatching(em_params)
    p.fit(sgt)

    res = p.transform(snames)

    features = [
        "X_feat_abbr_match",
        "X_feat_abs_len_diff",
        "X_feat_len_ratio",
        "X_feat_token_sort_ratio",
        "X_feat_token_set_ratio",
        "X_feat_partial_ratio",
        "X_feat_w_ratio",
        "X_feat_ratio",
        "X_feat_name_cut",
        "X_feat_norm_ed",
        "X_feat_norm_jaro",
        "X_feat_very_common_hit",
        "X_feat_common_hit",
        "X_feat_rare_hit",
        "X_feat_very_common_miss",
        "X_feat_common_miss",
        "X_feat_rare_miss",
        "X_feat_n_overlap_words",
        "X_feat_ratio_overlap_words",
        "X_feat_num_word_difference",
        "X_feat_score_0_rank",
        "X_feat_score_0_top2_dist",
        "X_feat_score_0_dist_to_max",
        "X_feat_score_0_dist_to_min",
        "X_feat_score_0_ptp",
        "X_feat_score_0_diff_to_next",
        "X_feat_score_0_diff_to_prev",
    ]

    assert res.count() == 118
    assert all(feat in res.columns for feat in features)
