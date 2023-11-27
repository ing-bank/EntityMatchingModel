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

import fnmatch

import numpy as np
import pandas as pd


def negative_rerank_cossim(indexer_df, rank_col: str, rank_max, uid_col: str = "uid", correct_col: str = "correct"):
    """Reorder the rank column in negative dataset of cosine similarity indexer

    Create a negative name-pairs dataset from a positive name-pairs dataset after it has passed through the cosine
    similarity indexer. Effectively we create a negative names dataset where the maximum rank has been reduced by one
    unit compared with the positive names dataset. These are the steps taken:

    - Positive correct name-pairs are removed.
    - Rerank the remaining candidates of a name-to-match.
    - Remove any remaining candidates with the highest rank. This is needed in cases where no positive correct pair
      was present.

    Args:
        indexer_df: input positive names dataframe, which is the output a cosine similarity indexer,
            from which the negative names dataframe is created.
        rank_col: name of rank column to reorder.
        rank_max: only rank values lower than this value are kept, after reranking.
        uid_col: name of uid column. default is 'uid'.
        correct_col: name of correct-match column. default is 'correct'.

    Returns:
        the created negative names dataset
    """
    # remove all positive correct candidate pairs: keep only False matches
    indexer_df = indexer_df[~indexer_df[correct_col]].copy()
    # rerank the remaining candidates. note that rank starts at 1
    indexer_df = indexer_df.sort_values(by=[uid_col, rank_col])
    # groupby preserves the order of the rows in each group. See:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html (sort)
    gb = indexer_df.groupby(uid_col)
    indexer_df[rank_col] = gb[rank_col].transform(lambda x: range(1, len(x) + 1))
    # remove any remaining candidates with the highest rank
    # (possible in cases of no positive correct pair)
    return indexer_df[indexer_df[rank_col] < rank_max]


def negative_rerank_sni(indexer_df, rank_col, rank_max, uid_col="uid", correct_col="correct"):
    """Reorder the rank column in negative dataset of SNI indexer

    Create a negative name-pairs dataset from a positive name-pairs dataset after it has passed through the SNI indexer.
    Effectively we create a negative names dataset where the maximum rank has been reduced by one unit compared with the
    positive names dataset. These are the steps taken:

    - Positive correct name-pairs are removed.
    - Rerank the remaining, relevant SNI candidates of a name-to-match.
    - Remove any remaining candidates with the highest rank. This is needed in cases where no positive correct pair
      was present.

    Args:
        indexer_df: input positive names dataframe, which is the output a SNI indexer, from which the negative names dataframe is created.
        rank_col: name of rank column to reorder.
        rank_max: only (absolute) rank values lower than this value are kept, after reranking.
        uid_col: name of uid column. default is 'uid'.
        correct_col: name of correct-match column. default is 'correct'.

    Returns:
        the created negative names dataset
    """
    # create map with ranks of positive correct matches per uid
    uids = indexer_df[indexer_df[correct_col]][uid_col].values
    ranks = indexer_df[indexer_df[correct_col]][rank_col].values
    uid_2_pcrank = dict(zip(uids, ranks))

    # remove all positive correct candidate pairs: keep only the false matches
    indexer_df = indexer_df[~indexer_df[correct_col]].copy()

    # cast ranks to int
    indexer_df[rank_col] = indexer_df[rank_col].astype(int)

    # groupby uid and rerank per uid. then merge.
    if len(indexer_df) > 0:
        indexer_df = indexer_df.sort_values(by=[uid_col, rank_col])
        gb = list(indexer_df.groupby(uid_col))
        # use list not np.array, latter can unwantedly convert pd.series to np.array
        uid_dfs = [_rerank_sni(udf, uid_2_pcrank.get(uid, None), rank_col) for uid, udf in gb]
        # concat fails if dfs are empty
        indexer_df = pd.concat(uid_dfs)

    # remove any remaining candidates with the highest rank (in cases of no positive correct pair)
    return indexer_df[abs(indexer_df[rank_col]) < rank_max]


def _rerank_sni(udf, rank_poscor, rank_col="rank_2"):
    """Rerank the remaining, relevant SNI candidates of a name-to-match.

    Rerank the remaining, relevant SNI candidates of a name-to-match, after the positive correct name-pair has
    been removed.

    - There is no need to rerank the other candidates when the positive correct match is exact.
    - There is no need to rerank the other candidates when there are other candidates left with the same rank as the
      original positive correct match.
    - Else, shift all 'higher' ranks by one place closer to zero, where positive sni ranks go down, negative ones go up.

    Args:
        udf: input names-pairs dataframe of one name-to-match. all should have same uid value.
        rank_poscor: the SNI rank value of the positive correct candidate that was removed.
        rank_col: name of SNI rank column to reorder.

    Returns:
        dataframe with the reranked negative name-pairs
    """
    if rank_poscor == 0 or rank_poscor is None or np.isnan(rank_poscor):
        # no need to shift other ranks if poscor rank is exact match (the other ranks are not exact matches.)
        # when rank_poscor is None there was no pos correct match, so nothing to shift.
        return udf
    if len(udf[udf[rank_col] == rank_poscor]) > 0:
        # no need to shift other ranks if there are other candidate pairs left with same rank
        return udf
    # shift all higher ranks by one place closer to zero.
    # match = all name-pairs that need to be shifted.
    match = udf[rank_col] > rank_poscor if rank_poscor > 0 else udf[rank_col] < rank_poscor
    n_selected = np.sum(match)
    if n_selected > 0:
        # apply shift in rank by one unit
        # positive sni ranks go down, negative ones go up.
        udf.loc[match, rank_col] = udf[match][rank_col] + (-1 if rank_poscor > 0 else +1)
    return udf


def merge_indexers(df: pd.DataFrame, indexers: list, rank_cols: list):
    """Merging of indexer datasets after the reranking

    Args:
        df: input positive names dataframe, which is the output of cosine similarity and/or SNI indexers,
            from which the negative names dataframe is created.
        indexers: indexer datasets after the reranking, will overwrite original input dataset.
        rank_cols: list with rank columns to overwrite.

    Returns:
        merged dataset of indexer datasets after the reranking
    """
    # remove all name pairs that have been removed from original df
    u_indices = np.unique(np.concatenate([indexer_df.index.values for indexer_df in indexers]))
    df = df[df.index.isin(u_indices)].copy()

    for rank_col, indexer_df in zip(rank_cols, indexers):
        # reset existing ranks
        df[rank_col] = np.nan
        # set updated ranks to those of indexer
        indices = indexer_df.index.values
        df.loc[indices, rank_col] = indexer_df[rank_col]
    return df


def create_positive_negative_samples(
    df: pd.DataFrame,
    uid_col: str = "uid",
    correct_col: str = "correct",
    positive_set_col: str = "positive_set",
    pattern_rank_col: str = "rank_*",
):
    """Create negative and (consistent) positive datasets from a single positive names dataset

    Create a negative name-pairs dataset from a positive name-pairs dataset after it has passed through cosine
    similarity and/or SNI indexers. Effectively we create a negative names dataset from about half of the input data,
    where the maximum rank gets reduced by one unit compared with the input positive names dataset.
    The other half (the positive names) are also reduced in rank-window accordingly.

    These are the steps taken for the negative names:

    - Positive correct name-pairs are removed.
    - Rerank the remaining candidates of a name-to-match.
    - Remove any remaining candidates with the highest rank. This is needed in cases where no positive correct pair
      was present.

    Args:
        df: input positive names dataframe, which is the output of cosine similarity and/or SNI indexers,
        from which the negative names dataframe is created.
        uid_col: name of uid column. default is 'uid'.
        correct_col: name of correct-match column. default is 'correct'.
        positive_set_col: name of column that indicates which names-to-match go to the positive (and negative)
            name pair datasets. default is 'positive_set'.
        pattern_rank_col: pattern used to search for rank columns. Each rank column corresponds to an indexer.
            default is the pattern 'rank_*'.

    Returns:
        the created, merged negative plus positive name-pairs dataset
    """
    # basic checking
    for col in [uid_col, correct_col, positive_set_col]:
        if col not in df.columns:
            msg = f"Column {col} not present in input dataframe."
            raise AssertionError(msg)
    rank_cols = fnmatch.filter(df.columns, pattern_rank_col)
    if len(rank_cols) == 0:
        msg = f"No columns with pattern {pattern_rank_col} present in input dataframe."
        raise AssertionError(msg)

    positive_df = df[df[positive_set_col]].copy()
    negative_df = df[~df[positive_set_col]].copy()

    # positive and negative sample rewindowing
    # since a name-pair can pass through multiple indexers, need to do rewindowing per indexer.
    pos_indexers = []
    neg_indexers = []

    # loop over different indexers and process based on sni or cossim.
    for rank_col in rank_cols:
        # automatically deduce window size from the ranks (num_candidates)
        rank_min = df[rank_col].min()
        rank_max = max(df[rank_col].max(), abs(rank_min))

        # pick all data points for which the indexer is filled
        # do so by selecting all data row for which rank_col is not a nan
        neg_indexer_df = negative_df[~pd.isna(negative_df[rank_col])]
        pos_indexer_df = positive_df[~pd.isna(positive_df[rank_col])]

        # indexers are assumed to be cossim or sni based
        if rank_min < 0:
            # assume sni indexer when there are negative ranks
            neg_indexer_df = negative_rerank_sni(neg_indexer_df, rank_col, rank_max, uid_col, correct_col)
        else:
            # else assume cossim indexer
            neg_indexer_df = negative_rerank_cossim(neg_indexer_df, rank_col, rank_max, uid_col, correct_col)
        neg_indexers.append(neg_indexer_df)

        # remove any remaining positive candidates with the highest rank (in cases of no positive correct pair)
        pos_indexer_df = pos_indexer_df[abs(pos_indexer_df[rank_col]) < rank_max]
        pos_indexers.append(pos_indexer_df)

    # remerge truncated indexers
    negative_df = merge_indexers(negative_df, neg_indexers, rank_cols)
    positive_df = merge_indexers(positive_df, pos_indexers, rank_cols)

    # return merged dataset
    return pd.concat([positive_df, negative_df])
