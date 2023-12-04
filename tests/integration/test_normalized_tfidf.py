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
import pytest

from emm.helper import spark_installed
from tests.utils import create_test_data

if spark_installed:
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


def check_transformation(row, expected_tokens, expected_vector_norm_range, expected_vector_len):
    norm = sum(np.power(row.features.values, 2))
    assert expected_vector_norm_range[0] <= norm <= expected_vector_norm_range[1]
    assert row.ngram_tokens == expected_tokens
    assert row.features.indices.size == expected_vector_len


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_normalized_tfidf(spark_session):
    # prepare data up to the point that the CosSimMatcher is used
    em = SparkEntityMatching(
        parameters={
            "preprocessor": "preprocess_merge_abbr",
            "indexers": [{"type": "cosine_similarity", "tokenizer": "words", "ngram": 1}],
            "entity_id_col": "id",
            "uid_col": "uid",
            "name_col": "name",
            "name_only": True,
            "supervised_on": False,
            "keep_all_cols": True,
        }
    )
    ground_truth, _ = create_test_data(spark_session)

    # Disable the cosine similarity
    stages = em.pipeline.getStages()
    stages[1].indexers[0].cossim = None

    em.fit(ground_truth)

    test_list = [
        {
            "row": (0, "Tzu Anmot Eddie", "NL", "0001", 1.0),  # 3 tokens, all in the vocabulary
            "expected_tokens": ["tzu", "anmot", "eddie"],
            "expected_vector_norm_range": [1 - 1e-7, 1 + 1e-7],
            "expected_vector_len": 3,
        },
        {
            "row": (1, "Tzu General Chinese Moon", "NL", "0002", 1.0),  # 4 token, only 3 in vocabulary
            "expected_tokens": ["tzu", "general", "chinese", "moon"],
            "expected_vector_norm_range": [0 + 1e-7, 1 - 1e-7],
            "expected_vector_len": 3,
        },
        {
            "row": (2, "Super Awesome WBAA Moon", "NL", "0003", 1.0),  # 4 token, 0 in vocabulary
            "expected_tokens": ["super", "awesome", "wbaa", "moon"],
            "expected_vector_norm_range": [0, 0],
            "expected_vector_len": 0,
        },
        {
            "row": (3, "", "NL", "0004", 1.0),  # empty string
            "expected_tokens": [],
            "expected_vector_norm_range": [0, 0],
            "expected_vector_len": 0,
        },
    ]

    names_to_match = spark_session.createDataFrame(
        [el["row"] for el in test_list], schema=["uid", "name", "country", "account", "amount"]
    )
    names_to_match = em.transform(names_to_match)
    names_to_match = names_to_match.toPandas()

    for i, test in enumerate(test_list):
        check_transformation(
            names_to_match.iloc[i],
            test["expected_tokens"],
            test["expected_vector_norm_range"],
            test["expected_vector_len"],
        )
