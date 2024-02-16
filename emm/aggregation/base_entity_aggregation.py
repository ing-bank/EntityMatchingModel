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

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from emm.base.pipeline import Pipeline
from emm.preprocessing.abbreviation_util import preprocess

if TYPE_CHECKING:
    import pandas as pd


def _mean_score_aggregation(df, group, score_col, output_col):
    # set dropna to False to keep no_candidate rows
    df[output_col] = df.groupby(group, dropna=False)[score_col].transform("mean")

    # Short circuit if possible
    if df.shape[0] == 1:
        return df.head(1)
    return df.sort_values(by=[output_col, score_col], ascending=False).head(1)


def is_series_unique(series: pd.Series) -> bool:
    # does series consist of one value only?
    a = series.to_numpy()
    return (a[0] == a).all()


def _max_frequency_nm_score_aggregation(
    df: pd.DataFrame, group, name_col: str, account_col: str, freq_col: str, score_col: str, output_col: str
) -> pd.DataFrame:
    # 1. handle trivial cases: just 1 row or just 1 name per account
    # Short circuit if possible
    if df.shape[0] == 1:
        df[output_col] = df[score_col]
        return df.reset_index(drop=True)

    if is_series_unique(df[name_col]):
        best_match_df = df.nlargest(1, [score_col], keep="first").copy()
        best_match_df[output_col] = best_match_df[score_col]
        return best_match_df

    # 2a. weigh the score of each name-pair by the frequency of the account-name
    # meaning: high-frequency account-names contribute more to the ultimate match
    df["freq_score"] = df[freq_col] * df[score_col]

    # 2b. calculate the normalized (aggregate) matching score of each account-gt pair.
    # the score is a weighted average of all names contributing to the same gt-id.
    # set dropna to False to keep no_candidate rows
    # note: group = ["gt_entity_id", "gt_uid", account_col]
    # when running in spark/pandas-apply, grouping on account has already been done.
    df_grouped = df.groupby(group, dropna=False)
    am_df = df_grouped[[freq_col, "freq_score"]].sum()
    am_df[output_col] = am_df["freq_score"] / am_df[freq_col]
    am_df = am_df.reset_index()

    # 2c. the best match is a combination of both name-frequency and name-matching score.
    # pick as match the *highest* summed score (freq_score) of all names in the account contributing to this gt-id.
    # we take this as the most likely gt-id for the account.
    best_match_df = am_df.nlargest(1, ["freq_score"], keep="first")

    # 3a. pick the most frequent name of each account-grid combi: one-name summary information
    group_key = tuple(best_match_df[group].to_numpy()[0])
    one_accountname_df = df_grouped.get_group(group_key).sort_values(["freq_score"], ascending=False).head(1)
    df = one_accountname_df.drop(columns=["freq_score"]).copy()
    df["agg_score"] = best_match_df["agg_score"].to_numpy()[0]
    return df


def matching_max_candidate(
    df: pd.DataFrame,
    group: list[str],
    score_col: str,
    name_col: str,
    account_col: str,
    freq_col: str,
    output_col: str,
    aggregation_method: Literal["max_frequency_nm_score", "mean_score"] = "max_frequency_nm_score",
) -> pd.DataFrame:
    """This function aggregates all the names and its candidates of an account.
    If aggregation_method = 'mean_score'
    - Average the scores per GT and return the maximum.

    Returns dataframe with a single row.

    Args:
        df: Pandas DataFrame containing all the names of an account
        group: Grouping columns used for calculating agg_score, usually or (gt_entity_id, gt_uid)
        score_col: Score column on which the aggregation is performed
        name_col: name column used for name clustering
        account_col: account column used for name clustering
        freq_col: Frequency column used for the name clustering and weighted averages
        output_col: Name of column to store the final score
        aggregation_method: Aggregation method to use: name_clustering, mean_score, or max_frequency_nm_score
    """
    if df.empty:
        msg = "Provided an empty df"
        raise ValueError(msg)

    df = df.copy()

    if aggregation_method == "mean_score":
        return _mean_score_aggregation(df, group, score_col, output_col)
    if aggregation_method == "max_frequency_nm_score":
        return _max_frequency_nm_score_aggregation(df, group, name_col, account_col, freq_col, score_col, output_col)
    msg = "aggregation_method not supported"
    raise ValueError(msg)


class BaseEntityAggregation(Pipeline):
    def __init__(
        self,
        score_col: str,
        account_col: str = "account",
        index_col: str = "entity_id",
        gt_entity_id_col: str = "gt_entity_id",
        uid_col: str = "uid",
        gt_uid_col: str = "gt_uid",
        name_col: str = "name_col",
        freq_col: str = "counterparty_account_count_distinct",
        output_col: str = "agg_score",
        preprocessed_col: str = "preprocessed",
        gt_name_col: str = "gt_name",
        gt_preprocessed_col: str = "gt_preprocessed",
        aggregation_method: Literal["max_frequency_nm_score", "mean_score"] = "max_frequency_nm_score",
        blacklist: list | None = None,
        positive_set_col: str = "positive_set",
    ) -> None:
        self.score_col = score_col
        self.account_col = account_col
        self.index_col = index_col
        self.gt_entity_id_col = gt_entity_id_col
        self.uid_col = uid_col
        self.gt_uid_col = gt_uid_col
        self.name_col = name_col
        self.freq_col = freq_col
        self.output_col = output_col
        self.preprocessed_col = preprocessed_col
        self.gt_name_col = gt_name_col
        self.gt_preprocessed_col = gt_preprocessed_col
        self.aggregation_method = aggregation_method
        self.blacklist = blacklist or []
        self.positive_set_col = positive_set_col

        # perform very basic preprocessing to blacklist, remove abbreviations, to lower, etc.
        self.blacklist = [preprocess(name) for name in self.blacklist]
        super().__init__()

    def get_group(self, dataframe) -> list[str]:
        group = [self.account_col]

        # We aggregate on index_col
        if self.index_col in dataframe.columns:
            group += [self.index_col]

        # Useful for collect_metrics()
        if self.positive_set_col in dataframe.columns:
            group += [self.positive_set_col]

        # Notice we lose the name_to_match 'uid' column here
        return group

    def get_gt_group(self) -> list[str]:
        if self.aggregation_method == "max_frequency_nm_score":
            return [self.gt_entity_id_col, self.gt_uid_col, self.account_col]
        if self.aggregation_method == "mean_score":
            return [self.gt_entity_id_col, self.gt_uid_col]
        msg = f"aggregation_method '{self.aggregation_method}'"
        raise ValueError(msg)

    @abstractmethod
    def remove_blacklisted_names(self, df: Any, preprocessed_col: str) -> Any:
        raise NotImplementedError
