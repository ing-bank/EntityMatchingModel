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

from typing import Any

import pandas as pd
from sklearn.base import TransformerMixin

from emm.indexing.base_indexer import BaseIndexer


class PandasNaiveIndexer(TransformerMixin, BaseIndexer):
    """Naive O(n^2) indexer for small datasets. Not for production use."""

    def __init__(self, indexer_id: int | None = None) -> None:
        """Naive O(n^2) indexer for small datasets. Not for production use."""
        BaseIndexer.__init__(self)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TransformerMixin:
        """Dummy function, no fitting required."""
        self.gt = X
        return self

    def transform(
        self, X: pd.DataFrame, spark_session: Any | None = None, multiple_indexers: bool = False
    ) -> pd.DataFrame:
        """Create all possible name-pairs

        Args:
            X: dataframe with (processed) input names to match to the ground truth.
            spark_session: ignored
            multiple_indexers: ignored

        Returns:
            dataframe with all possible candidate name pairs.
        """
        gt = pd.DataFrame()
        gt["gt_uid"] = self.gt.index.values

        query = pd.DataFrame()
        query["uid"] = X.index.values

        candidates = gt.merge(query, how="cross")
        candidates["score"] = 1

        gb = candidates.groupby("uid")
        candidates["rank"] = gb["gt_uid"].rank(method="dense", ascending=True)
        return candidates


NaiveIndexer = PandasNaiveIndexer
