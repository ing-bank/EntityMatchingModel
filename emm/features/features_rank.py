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

# to reduce effect of numeric errors in scores, before calculating rank features the scores
# are rounded with `round(RANK_FEATURES_PRECISION)`
# this does not eliminate the effect completely (since there could be some scores near the rounding border)
# but in most case this works fine
RANK_FEATURES_PRECISION = 5


def rank(df, c, uid_col):
    return group_by_uid(df, c, uid_col).apply(
        lambda x: x.round(RANK_FEATURES_PRECISION).rank(ascending=False, method="first")
    )


def top2_dist(df, c, uid_col):
    return group_by_uid(df, c, uid_col).transform(lambda x: ptp(x.nlargest(2)))


def dist_to_max(df, c, uid_col):
    return group_by_uid(df, c, uid_col).transform("max") - df[c]


def dist_to_min(df, c, uid_col):
    return df[c] - group_by_uid(df, c, uid_col).transform("min")


def feat_ptp(df, c, uid_col):
    return group_by_uid(df, c, uid_col).transform(ptp)


def diff_to_next(df, c, uid_col):
    return group_by_uid(df, c, uid_col).apply(lambda x: x.round(RANK_FEATURES_PRECISION).diff(1).abs())


def diff_to_prev(df, c, uid_col):
    return group_by_uid(df, c, uid_col).apply(lambda x: x.round(RANK_FEATURES_PRECISION).diff(-1))


def calc_rank_features(
    df: pd.DataFrame,
    funcs: dict[str, Callable],
    score_columns: list[str] | None,
    uid_col: str = "uid",
    fillna: int = -1,
) -> pd.DataFrame:
    if score_columns is None:
        score_columns = ["cossim_n3", "cossim_w"]
    res = pd.DataFrame(index=df.index)
    for c in score_columns:
        for column, func in funcs.items():
            res[f"{c}_{column}"] = func(df, c, uid_col).fillna(fillna).astype("int8" if "rank" in column else "float32")
    return res


def calc_diff_features(
    df: pd.DataFrame,
    funcs: dict[str, Callable],
    score_columns: list[str] | None,
    uid_col: str = "uid",
    fillna: int = -1,
) -> pd.DataFrame:
    if score_columns is None:
        score_columns = ["cossim_n3", "cossim_w"]
    res = pd.DataFrame(index=df.index)
    for c in score_columns:
        curr = df[[uid_col, c]].copy()
        curr[c] = curr[c].round(RANK_FEATURES_PRECISION)
        curr = curr.sort_values(by=[uid_col, c], ascending=[True, False])

        for column, func in funcs.items():
            res[f"{c}_{column}"] = func(curr, c, uid_col).fillna(fillna).astype("float32")
    return res


def group_by_uid(df, c, uid_col):
    # aggregates candidates using name to match UID (uid_col)
    return df.groupby(uid_col, group_keys=False)[c]


def ptp(a: np.array):
    """Numpy `ptp` that is safe if input contains no elements or only NaN.

    Range of values (maximum - minimum) along an axis.
    """
    if a is None or len(a) == 0 or np.min(a) is None:
        return 0
    return np.ptp(a)
