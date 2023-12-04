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
from example import example, example_pandas, example_spark
from tests.utils import read_markdown


def test_readme_example():
    # test the example in the readme
    (n_ground_truth, n_noised_names, n_names_to_match, n_best_match, n_correct, _) = example()

    assert n_ground_truth == 6800
    assert n_noised_names == 6800
    assert n_names_to_match == 1800
    assert n_best_match == 1800
    # number depends slightly on version of xgboost
    assert n_correct > 1600


def test_example_pandas():
    best_candidates_pd = example_pandas()
    best_candidates_pd.sort_values(["name"], inplace=True)
    best_candidates_pd.reset_index(drop=True, inplace=True)

    cand_excepted_pd = read_markdown(
        """
| name       | gt_name   |    gt_entity_id |
|:-----------|:----------|----------------:|
| Apl        | Apple     |               1 |
| Aplle      | Apple     |               1 |
| Microbloft | Microsoft |               2 |
| Netflfli   | Netflix   |               5 |
| amz        | Amazon    |               4 |
| googol     | Google    |               3 |
"""
    )
    pd.testing.assert_frame_equal(best_candidates_pd, cand_excepted_pd, check_dtype=False)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_example_spark(spark_session):
    best_candidates_pd = example_spark(spark_session)
    best_candidates_pd.sort_values(["name"], inplace=True)
    best_candidates_pd.reset_index(drop=True, inplace=True)

    cand_excepted_pd = read_markdown(
        """
| name       | gt_name   |    gt_entity_id |
|:-----------|:----------|----------------:|
| Apl        | Apple     |               1 |
| Aplle      | Apple     |               1 |
| Microbloft | Microsoft |               2 |
| Netflfli   | Netflix   |               5 |
| amz        | Amazon    |               4 |
| googol     | Google    |               3 |
"""
    )
    pd.testing.assert_frame_equal(best_candidates_pd, cand_excepted_pd, check_dtype=False)
