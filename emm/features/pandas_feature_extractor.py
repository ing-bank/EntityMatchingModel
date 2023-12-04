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

from functools import partial
from typing import Callable

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Jaro, Levenshtein
from sklearn.base import TransformerMixin

from emm.features.base_feature_extractor import BaseFeatureExtractor
from emm.features.features_extra import calc_extra_features
from emm.features.features_lef import calc_lef_features
from emm.features.features_name import abbr_match, abs_len_diff, calc_name_features, len_ratio, name_cut
from emm.features.features_rank import (
    calc_diff_features,
    calc_rank_features,
    diff_to_next,
    diff_to_prev,
    dist_to_max,
    dist_to_min,
    feat_ptp,
    rank,
    top2_dist,
)
from emm.features.features_vocabulary import Vocabulary, compute_vocabulary_features, create_vocabulary
from emm.loggers import Timer


class PandasFeatureExtractor(TransformerMixin, BaseFeatureExtractor):
    """Sklearn based transformer for calculating numeric features for candidate pairs (used by supervised model)

    Args:
        name1_col: column with name from names to match
        name2_col: column with name from ground truth
        uid_col: column with unique ID of row from names to match
        score_columns: list of columns with raw scores from indexers
        extra_features: list of columns used for extra features (i.e. country)
        without_rank_features: if False then score rank based features will be calculated (can be overridden in transform function)
        with_legal_entity_forms_match: if True, add match of legal entity forms feature
        fillna_value: fill nans with float value. default is None.
        drop_features: list of features to drop at end of calculation, before passing to sm. default is None.
    """

    def __init__(
        self,
        name1_col: str = "preprocessed",
        name2_col: str = "gt_preprocessed",
        uid_col: str = "uid",
        gt_uid_col: str = "gt_uid",
        score_columns: list[str] | None = None,
        extra_features: list[str | tuple[str, Callable]] | None = None,
        vocabulary: Vocabulary | None = None,
        without_rank_features: bool = False,
        with_legal_entity_forms_match: bool = False,
        fillna_value: float | None = None,
        drop_features: list[str] | None = None,
    ) -> None:
        self.name1_col = name1_col
        self.name2_col = name2_col
        self.uid_col = uid_col
        self.gt_uid_col = gt_uid_col
        self.score_columns = score_columns or []
        self.extra_features = extra_features or []
        self.vocabulary = vocabulary
        self.without_rank_features = without_rank_features
        self.with_legal_entity_forms_match = with_legal_entity_forms_match
        self.fillna_value = fillna_value
        self.drop_features = drop_features
        super().__init__()

        self.name_features = {
            "abbr_match": (abbr_match, "int8"),
            "abs_len_diff": (abs_len_diff, "int8"),
            "len_ratio": (len_ratio, "float32"),
            "token_sort_ratio": (fuzz.token_sort_ratio, "int8"),
            "token_set_ratio": (fuzz.token_set_ratio, "int8"),
            "partial_ratio": (fuzz.partial_ratio, "int8"),
            "w_ratio": (fuzz.WRatio, "int8"),
            "ratio": (fuzz.ratio, "int8"),
            "name_cut": (name_cut, "int8"),
            "norm_ed": (Levenshtein.distance, "int8"),
            "norm_jaro": (Jaro.similarity, "float32"),
        }
        self.rank_features = {
            "rank": rank,
            "top2_dist": top2_dist,
            "dist_to_max": dist_to_max,
            "dist_to_min": dist_to_min,
            "ptp": feat_ptp,
        }
        self.diff_features = {
            # this assumes that scores are sorted in ascending order
            "diff_to_next": diff_to_next,
            "diff_to_prev": diff_to_prev,
        }
        self.funcs = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PandasFeatureExtractor:
        if X is not None and self.vocabulary is None:
            self.vocabulary = create_vocabulary(X, columns=[self.name1_col, self.name2_col])
        self._fitted = True
        return self

    def _get_funcs(self):
        funcs = [
            partial(calc_name_features, funcs=self.name_features, name1=self.name1_col, name2=self.name2_col),
            partial(
                compute_vocabulary_features,
                col1=self.name1_col,
                col2=self.name2_col,
                very_common_words=self.vocabulary.very_common_words,
                common_words=self.vocabulary.common_words,
            ),
        ]
        if len(self.extra_features) > 0:
            funcs.append(partial(calc_extra_features, features=self.extra_features))

        if not self.without_rank_features:
            # warning! those features are very sensitive to changes in the scores
            funcs += [
                partial(
                    calc_rank_features, funcs=self.rank_features, score_columns=self.score_columns, uid_col=self.uid_col
                ),
                partial(
                    calc_diff_features, funcs=self.diff_features, score_columns=self.score_columns, uid_col=self.uid_col
                ),
            ]
        if self.with_legal_entity_forms_match:
            funcs.append(partial(calc_lef_features, name1=self.name1_col, name2=self.name2_col))
        return funcs

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(X)
        if self.funcs is None:
            self.funcs = self._get_funcs()

        """Transforms dataframe with candidate pairs to a data frame with calculated features.
        The `X` dataframe should contain at least `name1_col,name2_col,uid_col` and `score_columns`.

        Args:
            X: dataframe with candidate pairs (one row per candidate pair)

        Returns:
            The resulting dataframe contains:

            * score columns (unmodified)
            * name features (i.e. edit distance)
            * hits features (i.e. number of common tokens)
            * rank features (i.e. rank of the candidate pair, based on each score)
            * diff features (i.e. distance to next/prev score)
            * legal entity form features (i.e. contains ltd)
        """
        with Timer("CalcFeatures.transform") as timer:
            timer.log_param("cands", len(X))

            # make ordering of the input data deterministic, we store original index, to be able to return features in right ordering
            org_index = X.index.copy()
            X = X.sort_values(by=[self.name1_col, self.name2_col, self.uid_col])
            for c in self.score_columns:
                X[c] = X[c].astype("float32")
            for c in [self.name1_col, self.name2_col]:
                X[c] = X[c].astype(str)

            if self.fillna_value is not None and isinstance(self.fillna_value, float):
                X = X.fillna(self.fillna_value)

            results = [func(X) for func in self.funcs]

            # Concatenate all features as columns
            res = pd.concat([X[self.score_columns], *results], axis=1, sort=False)

            # handy for forward/backward compatibility; extra features can be removed here.
            if isinstance(self.drop_features, list) and len(self.drop_features) > 0:
                res = res.drop(columns=self.drop_features, axis=1)

            # Reset the index
            return res.reindex(org_index)
