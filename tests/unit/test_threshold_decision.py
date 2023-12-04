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

from emm import PandasEntityMatching
from emm.threshold.threshold_decision import get_threshold_curves_parameters


def test_threshold(supervised_model):
    dataset_scored = pd.read_csv(supervised_model[1])

    new_params = get_threshold_curves_parameters(dataset_scored, "nm_score", False)
    emo = PandasEntityMatching(new_params)
    agg_name = emo.get_threshold_agg_name(False)

    # Asking a high value, that is actually giving precision = 1
    threshold1 = emo.calc_threshold(agg_name=agg_name, type_name="positive", metric_name="precision", min_value=0.95)
    assert threshold1 == pytest.approx(0.86, abs=0.02)

    # Asking an impossible value, that is falling back the maximum precision, that is same threshold as above
    threshold2 = emo.calc_threshold(agg_name=agg_name, type_name="positive", metric_name="precision", min_value=2)
    assert threshold1 == threshold2

    # Asking a medium value
    threshold3 = emo.calc_threshold(agg_name=agg_name, type_name="positive", metric_name="precision", min_value=0.3)
    assert threshold3 == pytest.approx(0.00, abs=0.02)

    # Asking a very low value
    threshold4 = emo.calc_threshold(agg_name=agg_name, type_name="positive", metric_name="precision", min_value=0)
    assert threshold4 == pytest.approx(0.0, abs=0.02)

    # Other metrics
    assert emo.calc_threshold(
        agg_name=agg_name, type_name="all", metric_name="precision", min_value=0.41
    ) == pytest.approx(0.01639187, abs=0.02)
    assert emo.calc_threshold(agg_name=agg_name, type_name="all", metric_name="TNR", min_value=0.5) == pytest.approx(
        0.86160237, abs=0.02
    )
    assert emo.calc_threshold(agg_name=agg_name, type_name="all", metric_name="TPR", min_value=0.5) == pytest.approx(
        0.0, abs=0.02
    )
    assert emo.calc_threshold(
        agg_name=agg_name, type_name="all", metric_name="fullrecall", min_value=0.5
    ) == pytest.approx(0.0, abs=0.02)
    assert emo.calc_threshold(
        agg_name=agg_name, type_name="all", metric_name="predicted_matches_rate", min_value=0.5
    ) == pytest.approx(0.0, abs=0.02)

    thresholds_all = np.array([0.986111, 0.868565, 0.009231, 0.0])
    np.testing.assert_allclose(
        thresholds_all, emo.parameters["threshold_curves"][agg_name]["all"]["thresholds"], atol=0.0033
    )

    thresholds_neg = np.array([0.037389, 0.0])
    np.testing.assert_allclose(
        thresholds_neg, emo.parameters["threshold_curves"][agg_name]["negative"]["thresholds"], atol=0.0033
    )
