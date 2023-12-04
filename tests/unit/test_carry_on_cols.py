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

from emm import PandasEntityMatching
from emm.helper import spark_installed

if spark_installed:
    from emm import SparkEntityMatching


def test_carry_on_cols_pandas():
    # test to pass on column 'a'
    ground_truth = pd.DataFrame(
        {
            "name": ["Apple", "Microsoft", "Google", "Amazon", "Netflix", "Spotify"],
            "id": [1, 2, 3, 4, 5, 6],
            "a": [1, 1, 1, 1, 1, 1],
        }
    )
    names = pd.DataFrame(
        {"name": ["Aplle", "Microbloft", "Googol", "amz", "Netfliks", "Spot-off"], "b": [2, 2, 2, 2, 2, 2]}
    )

    indexers = [
        {"type": "sni", "window_length": 3}  # sorted neighbouring indexing window of size 3.
    ]
    emm_config = {
        "name_only": True,  # only consider name information for matching
        "entity_id_col": "id",  # important to set both index and name columns
        "name_col": "name",
        "indexers": indexers,
        "supervised_on": False,  # no initial supervised model to select best candidates right now
        "carry_on_cols": ["a", "b"],
    }

    # fitting of first the ground truth, then the training names to match.
    model = PandasEntityMatching(emm_config)
    model.fit(ground_truth)

    candidates = model.transform(names)

    assert "gt_a" in candidates
    assert "b" in candidates
    assert (candidates["gt_a"] == 1).all()
    assert (candidates["b"] == 2).all()
    assert len(candidates) == 8


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_carry_on_cols_spark(spark_session):
    # test to pass on column 'a'
    ground_truth = spark_session.createDataFrame(
        [
            ("Apple", 1, 1),
            ("Microsoft", 2, 1),
            ("Google", 3, 1),
            ("Amazon", 4, 1),
            ("Netflix", 5, 1),
            ("Spotify", 6, 1),
        ],
        ["name", "id", "a"],
    )
    names = spark_session.createDataFrame(
        [("Aplle", 2), ("MicorSoft", 2), ("Gugle", 2), ("amz", 2), ("Netfliks", 2), ("Spot-off", 2)], ["name", "b"]
    )

    # two example name-pair candidate generators: character-based cosine similarity and sorted neighbouring indexing
    indexers = [
        {"type": "sni", "window_length": 3}  # sorted neighbouring indexing window of size 3.
    ]
    emm_config = {
        "name_only": True,  # only consider name information for matching
        "entity_id_col": "id",  # important to set both index and name columns
        "name_col": "name",
        "indexers": indexers,
        "supervised_on": False,  # no initial supervised model to select best candidates right now
        "carry_on_cols": ["a", "b"],
    }

    # fitting of first the ground truth, then the training names to match.
    model = SparkEntityMatching(emm_config)
    model.fit(ground_truth)

    spark_candidates = model.transform(names)
    candidates = spark_candidates.toPandas()

    assert "gt_a" in candidates
    assert "b" in candidates
    assert (candidates["gt_a"] == 1).all()
    assert (candidates["b"] == 2).all()
    assert len(candidates) == 8
