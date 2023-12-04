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

import time

import pytest

from emm.helper import spark_installed

if spark_installed:
    from emm.data.create_data import create_noised_data
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


def companies_data(spark_session):
    random_seed = 42

    companies_ground_truth, companies_noised = create_noised_data(
        spark_session, noise_level=0.3, noise_count=1, split_pos_neg=False, random_seed=random_seed
    )

    companies_ground_truth.persist()
    companies_noised.persist()

    companies_noised_pd = companies_noised.toPandas()

    # This is always the same (even without fixing the set):
    assert companies_ground_truth.count() == 6800
    assert companies_noised.count() == 6800
    mem_used = companies_noised_pd.memory_usage(deep=True).sum() / 1024**2
    assert abs(mem_used - 1.44) < 0.02
    return companies_ground_truth, companies_noised, companies_noised_pd


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_artificial_integration(spark_session, supervised_model):
    companies_ground_truth, companies_noised, _ = companies_data(spark_session)
    em_obj = SparkEntityMatching(
        {
            "preprocessor": "preprocess_merge_abbr",
            "indexers": [{"type": "cosine_similarity", "tokenizer": "words", "ngram": 1, "num_candidates": 10}],
            "entity_id_col": "Index",
            "uid_col": "uid",
            "name_col": "Name",
            "supervised_on": True,
            "supervised_model_dir": supervised_model[0].parent,
            "supervised_model_filename": supervised_model[0].name,
            "partition_size": None,
        }
    )

    start = time.time()
    em_obj.fit(companies_ground_truth)
    nm_results = em_obj.transform(companies_noised).toPandas()

    nm_results["score_0_row_n"] = (
        nm_results.sort_values(["uid", "score_0"], ascending=[True, False]).groupby(["uid"]).cumcount()
    )
    nm_results_best = nm_results[nm_results["score_0_row_n"] == 0].copy()

    assert nm_results["uid"].nunique() == len(nm_results_best)  # no names to match should be lost

    time_spent = time.time() - start
    assert time_spent < 270  # less than 4 minutes 30 s

    nm_results_best["hit"] = nm_results_best["uid"] == nm_results_best["gt_uid"]
    accuracy = float(sum(nm_results_best["hit"])) / len(nm_results_best["hit"])
    assert accuracy > 0.55  # at least 55% accuracy expected when noise level is 0.3

    # similarity scores must be between 0 and 1 (exclusive 0) or None
    assert nm_results["score_0"].fillna(0).between(0, 1 + 1e-6, inclusive="both").all()

    companies_ground_truth.unpersist()
    companies_noised.unpersist()
