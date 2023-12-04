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
from typing import Literal

import pandas as pd
from sklearn.base import TransformerMixin

from emm.aggregation.base_entity_aggregation import BaseEntityAggregation, matching_max_candidate
from emm.loggers import Timer


class PandasEntityAggregation(TransformerMixin, BaseEntityAggregation):
    """Pandas name-matching aggregation code"""

    def __init__(
        self,
        score_col: str,
        account_col: str = "account",
        index_col: str = "entity_id",
        gt_entity_id_col: str = "gt_entity_id",
        uid_col: str = "uid",
        gt_uid_col: str = "gt_uid",
        name_col: str = "name",
        freq_col: str = "counterparty_account_count_distinct",
        output_col: str = "agg_score",
        preprocessed_col: str = "preprocessed",
        gt_name_col: str = "gt_name",
        gt_preprocessed_col: str = "gt_preprocessed",
        aggregation_method: Literal["max_frequency_nm_score", "mean_score"] = "max_frequency_nm_score",
        blacklist: list[str] | None = None,
    ) -> None:
        """Pandas name-matching aggregation code

        Last and optional step in PandasEntityMatching.

        Optionally, the EMM package can also be used to match a group of company names that belong together,
        to a company name in the ground truth. (For example, all names used to address an external bank account.)

        This step makes use of name-matching scores from the supervised layer. We refer to this as the aggregation step.
        (This step is not needed for standalone name matching.)

        The `account_col` column indicates which names-to-match belong together.
        The combination of scores is based on `score_col`, e.g. the name-matching score `nm_score`.

        Two aggregation methods are available:

        - "mean_score": takes the mean score from all names-to-match to find the best ground-truth name.
        - "max_frequency_nm_score": weights the nm_score with the frequency and takes the maximum to find the best
            ground-truth name.

        Args:
            score_col: name-matching score "nm_score" or first cosine similarity score "score_0".
            account_col: account column, default is "account".
            index_col: id column, default is "entity_id".
            gt_entity_id_col: ground truth id column, default is "gt_entity_id".
            uid_col: uid column, default is "uid".
            gt_uid_col: ground truth uid column, default is "gt_uid".
            name_col: name column, default is "name".
            freq_col: name frequency column, default is "counterparty_account_count_distinct".
            output_col: Name of column to store the final score
            preprocessed_col: Name of column of preprocessed input
            gt_name_col: ground truth name column, default is "gt_name".
            gt_preprocessed_col: column name of preprocessed ground truth names, default is "preprocessed".
            aggregation_method: default is "max_frequency_nm_score", alternative is "mean_score".
            blacklist: blacklist of names to skip in clustering.
        """
        BaseEntityAggregation.__init__(
            self,
            score_col=score_col,
            account_col=account_col,
            index_col=index_col,
            gt_entity_id_col=gt_entity_id_col,
            uid_col=uid_col,
            gt_uid_col=gt_uid_col,
            name_col=name_col,
            freq_col=freq_col,
            output_col=output_col,
            preprocessed_col=preprocessed_col,
            gt_name_col=gt_name_col,
            gt_preprocessed_col=gt_preprocessed_col,
            aggregation_method=aggregation_method,
            blacklist=blacklist or [],
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TransformerMixin:
        """Dummy function, no fitting is required."""
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Only calls transform(), no fitting required"""
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Combine scores of a group of name-pair candidates that belong together.

        Natch a group of company names that belong together, to a company name in the ground truth.

        Args:
            X: dataframe of scored candidates

        Returns:
            dataframe of scored candidates, only one row per account
        """
        if X is None:
            return None

        with Timer("PandasEntityAggregation.transform") as timer:
            timer.log_param("n", len(X))

            group = self.get_group(X)
            gt_group = self.get_gt_group()

            # filter out accounts with no matches (nans) and filter out accounts with just one name.
            # no need to pass those to apply_func.
            grouped_match = X[~X[self.gt_uid_col].isna()].groupby(group)
            mpl_match_df = grouped_match.filter(lambda x: len(x) > 1)

            one_match_df = grouped_match.filter(lambda x: len(x) == 1)
            one_match_df["agg_score"] = one_match_df[self.score_col]
            one_match_df["freq_score"] = one_match_df[self.score_col] * one_match_df[self.freq_col]

            group_func = partial(
                matching_max_candidate,
                group=gt_group,
                score_col=self.score_col,
                name_col=self.name_col,
                account_col=self.account_col,
                freq_col=self.freq_col,
                output_col=self.output_col,
                aggregation_method=self.aggregation_method,
            )

            # filter out all processed names that are in blacklist or empty.
            mpl_match_df = self.remove_blacklisted_names(df=mpl_match_df, preprocessed_col=self.preprocessed_col)

            # reset index to drop index generated by apply(group_func)
            cl_match_df = mpl_match_df.groupby(group, as_index=False).apply(group_func).reset_index(drop=True)

            # concat all account matches
            res = pd.concat([one_match_df, cl_match_df])

            assert self.output_col in res.columns
            # currently we leave only 1 row per account, so by definition it is best match
            res["best_match"] = True
            res["best_rank"] = 1
            timer.log_param("cands", len(res))
        return res

    def remove_blacklisted_names(self, df: pd.DataFrame, preprocessed_col: str = "preprocessed"):
        # filter out all processed names that are in blacklist or empty.
        # idea: these are too generic/not-good to use for account matching anyway.
        if preprocessed_col in df.columns:
            # preprocessed column should always be present
            return df.loc[~df[preprocessed_col].isin([*self.blacklist, ""])]
        return df
