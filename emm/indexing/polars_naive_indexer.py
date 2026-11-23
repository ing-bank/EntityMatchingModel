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

import polars as pl
from sklearn.base import TransformerMixin

from emm.indexing.base_indexer import BaseIndexer


class PolarsNaiveIndexer(TransformerMixin, BaseIndexer):
    """Create all candidate pairs for small datasets."""

    def __init__(self, indexer_id: int | None = None) -> None:
        BaseIndexer.__init__(self)
        self.gt: pl.DataFrame | None = None

    def fit(self, X: pl.DataFrame, y: Any = None) -> PolarsNaiveIndexer:
        self.gt = X
        return self

    def transform(self, X: pl.DataFrame, multiple_indexers: bool = False) -> pl.DataFrame:
        if self.gt is None:
            msg = "model is not fitted yet"
            raise ValueError(msg)
        gt = pl.DataFrame({"gt_uid": pl.Series(range(self.gt.height), dtype=pl.Int64)})
        query = pl.DataFrame({"uid": pl.Series(range(X.height), dtype=pl.Int64)})
        return gt.join(query, how="cross").with_columns(
            pl.lit(1).alias("score"), (pl.col("gt_uid") + 1).cast(pl.Float64).alias("rank")
        )

    def column_prefix(self) -> str:
        return "naive"
