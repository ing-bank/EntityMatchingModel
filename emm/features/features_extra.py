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

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def calc_extra_features(df: pd.DataFrame, features: list[str | tuple[str, Callable]]) -> pd.DataFrame:
    """Compute features for provided column

    Args:
        df: the input dataframe
        features: a list of strings indicating column names (for exact matches), a tuple with column name and function

    Returns:
        Feature dataframe
    """
    res = pd.DataFrame(index=df.index)
    for feat in features:
        vectorized = isinstance(feat, tuple)
        feat_name, func = feat if vectorized else (feat, None)
        gt_feat_name = f"gt_{feat_name}"
        if not (feat_name in df.columns and gt_feat_name in df.columns):
            msg = (
                f"missing extra features columns ('{feat_name}', '{gt_feat_name}') in df.columns={df.columns.tolist()}"
            )
            raise ValueError(msg)

        # cannot do comparisons with pd.NA, resulting in TypeError, so replace them with None
        df[feat_name] = df[feat_name].replace({pd.NA: None})

        if vectorized:
            res[feat_name] = np.vectorize(func)(df[feat_name], df[gt_feat_name])
        else:
            x = df[feat_name].eq(df[gt_feat_name]).astype(int)
            x[x == 0] = -1
            x[(df[feat_name].isnull()) | (df[gt_feat_name].isnull())] = 0
            res[feat_name] = x

    return res
