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

import re
from typing import Any

import polars as pl

from emm.loggers import Timer
from emm.supervised_model.base_supervised_model import create_new_model_pipeline
from emm.supervised_model.polars_supervised_model import _polars_model


def train_model_polars(  # noqa: PLR0917
    train_df: pl.DataFrame,
    vocabulary: Any = None,
    name_only: bool = False,
    without_rank_features: bool = False,
    positive_set_col: str = "positive_set",
    custom_model: Any = None,
    score_columns: list[str] | None = None,
    with_legal_entity_forms_match: bool = False,
    drop_features: list[str] | None = None,
    n_jobs: int = -1,
    positive_only: bool = False,
    extra_features: list | None = None,
    **feature_kws: Any,
) -> Any:
    """Train the standard supervised pipeline on native Polars features."""
    for column in ("correct", "no_candidate"):
        if column not in train_df.columns:
            msg = f"column {column} not in dataset. Did you run prepare_dataset()?"
            raise ValueError(msg)
    if score_columns is None:
        score_columns = [column for column in train_df.columns if re.match(r"^(score)_\d+$", column)]
    train_fit = train_df.filter(~pl.col("no_candidate"))
    if positive_only and positive_set_col in train_fit.columns:
        train_fit = train_fit.filter(pl.col(positive_set_col))
    if custom_model is None:
        feature_args = {
            "vocabulary": vocabulary,
            "score_columns": score_columns,
            "without_rank_features": without_rank_features,
            "with_legal_entity_forms_match": with_legal_entity_forms_match,
            "drop_features": drop_features,
            "extra_features": extra_features,
            **feature_kws,
        }
        custom_model = create_new_model_pipeline(
            name_only=name_only, feature_args=feature_args, xgb_args={"n_jobs": n_jobs}
        )
    model = _polars_model(custom_model)
    with Timer("Fitting supervised model pipeline"):
        model.fit(X=train_fit, y=train_fit["correct"].to_numpy())
    return model
