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

from emm import resources
from emm.data.negative_data_creation import (
    create_positive_negative_samples,
    merge_indexers,
    negative_rerank_cossim,
    negative_rerank_sni,
)


@pytest.fixture()
def namepairs_df():
    return pd.read_csv(resources.data("unittest_sample_namepairs.csv.gz"))


def test_unittest_sample(namepairs_df):
    positive_df = namepairs_df[namepairs_df.positive_set]
    negative_df = namepairs_df[~namepairs_df.positive_set]

    neg_indexer0_df = negative_df[~pd.isna(negative_df["rank_0"])]
    pos_indexer0_df = positive_df[~pd.isna(positive_df["rank_0"])]
    neg_indexer1_df = negative_df[~pd.isna(negative_df["rank_1"])]
    pos_indexer1_df = positive_df[~pd.isna(positive_df["rank_1"])]
    neg_indexer2_df = negative_df[~pd.isna(negative_df["rank_2"])]
    pos_indexer2_df = positive_df[~pd.isna(positive_df["rank_2"])]

    # before negative sample creation
    np.testing.assert_equal(np.max(namepairs_df.rank_0), 11.0)
    np.testing.assert_equal(np.max(namepairs_df.rank_1), 11.0)
    np.testing.assert_equal(np.max(namepairs_df.rank_2), 2.0)

    np.testing.assert_equal(np.min(namepairs_df.rank_0), 1.0)
    np.testing.assert_equal(np.min(namepairs_df.rank_1), 1.0)
    np.testing.assert_equal(np.min(namepairs_df.rank_2), -2.0)

    np.testing.assert_equal(len(namepairs_df), 201)
    np.testing.assert_equal(len(positive_df), 123)
    np.testing.assert_equal(len(negative_df), 78)

    np.testing.assert_equal(np.sum(positive_df.correct), 6)
    np.testing.assert_equal(np.sum(negative_df.correct), 4)

    np.testing.assert_equal(len(pos_indexer0_df), 66)
    np.testing.assert_equal(len(neg_indexer0_df), 33)
    np.testing.assert_equal(len(pos_indexer1_df), 66)
    np.testing.assert_equal(len(neg_indexer1_df), 44)
    np.testing.assert_equal(len(pos_indexer2_df), 18)
    np.testing.assert_equal(len(neg_indexer2_df), 12)


def test_create_positive_negative_samples(namepairs_df):
    dataset = create_positive_negative_samples(namepairs_df)

    positive_df = dataset[dataset.positive_set]
    negative_df = dataset[~dataset.positive_set]

    neg_indexer0_df = negative_df[~pd.isna(negative_df["rank_0"])]
    pos_indexer0_df = positive_df[~pd.isna(positive_df["rank_0"])]
    neg_indexer1_df = negative_df[~pd.isna(negative_df["rank_1"])]
    pos_indexer1_df = positive_df[~pd.isna(positive_df["rank_1"])]
    neg_indexer2_df = negative_df[~pd.isna(negative_df["rank_2"])]
    pos_indexer2_df = positive_df[~pd.isna(positive_df["rank_2"])]

    # after negative sample creation
    np.testing.assert_equal(np.max(dataset.rank_0), 10.0)
    np.testing.assert_equal(np.max(dataset.rank_1), 10.0)
    np.testing.assert_equal(np.max(dataset.rank_2), 1.0)

    np.testing.assert_equal(np.min(dataset.rank_0), 1.0)
    np.testing.assert_equal(np.min(dataset.rank_1), 1.0)
    np.testing.assert_equal(np.min(dataset.rank_2), -1.0)

    np.testing.assert_equal(len(dataset), 177)
    np.testing.assert_equal(len(positive_df), 107)
    np.testing.assert_equal(len(negative_df), 70)

    np.testing.assert_equal(np.sum(positive_df.correct), 6)
    np.testing.assert_equal(np.sum(negative_df.correct), 0)

    np.testing.assert_equal(len(pos_indexer0_df), 60)
    np.testing.assert_equal(len(neg_indexer0_df), 29)
    np.testing.assert_equal(len(pos_indexer1_df), 60)
    np.testing.assert_equal(len(neg_indexer1_df), 40)
    np.testing.assert_equal(len(pos_indexer2_df), 12)
    np.testing.assert_equal(len(neg_indexer2_df), 4)


def test_negative_rerank_sni(namepairs_df):
    negative_df = namepairs_df[~namepairs_df.positive_set]
    neg_indexer2_df = negative_df[~pd.isna(negative_df["rank_2"])]
    neg_indexer_df = negative_rerank_sni(neg_indexer2_df, "rank_2", 2, "uid", "correct")

    np.testing.assert_equal(len(neg_indexer_df), 4)
    np.testing.assert_equal(np.sum(neg_indexer_df.correct), 0)
    np.testing.assert_equal(np.min(neg_indexer_df.rank_2), -1.0)
    np.testing.assert_equal(np.max(neg_indexer_df.rank_2), 1.0)


def test_negative_rerank_cossim_w(namepairs_df):
    negative_df = namepairs_df[~namepairs_df.positive_set]
    neg_indexer0_df = negative_df[~pd.isna(negative_df["rank_0"])]
    neg_indexer_df = negative_rerank_cossim(neg_indexer0_df, "rank_0", 10)

    np.testing.assert_equal(len(neg_indexer_df), 27)
    np.testing.assert_equal(np.sum(neg_indexer_df.correct), 0)
    np.testing.assert_equal(np.min(neg_indexer_df.rank_0), 1.0)
    np.testing.assert_equal(np.max(neg_indexer_df.rank_0), 9.0)


def test_negative_rerank_cossim_n(namepairs_df):
    negative_df = namepairs_df[~namepairs_df.positive_set]
    neg_indexer1_df = negative_df[~pd.isna(negative_df["rank_1"])]
    neg_indexer_df = negative_rerank_cossim(neg_indexer1_df, "rank_1", 10)

    np.testing.assert_equal(len(neg_indexer_df), 36)
    np.testing.assert_equal(np.sum(neg_indexer_df.correct), 0)
    np.testing.assert_equal(np.min(neg_indexer_df.rank_1), 1.0)
    np.testing.assert_equal(np.max(neg_indexer_df.rank_1), 9.0)


def test_negative_merge_indexers(namepairs_df):
    positive_df = namepairs_df[namepairs_df.positive_set]

    pos_indexer0_df = positive_df[~pd.isna(positive_df["rank_0"])]
    pos_indexer1_df = positive_df[~pd.isna(positive_df["rank_1"])]
    pos_indexer2_df = positive_df[~pd.isna(positive_df["rank_2"])]

    indexers = [pos_indexer0_df, pos_indexer1_df, pos_indexer2_df]
    rank_cols = ["rank_0", "rank_1", "rank_2"]

    merged_df = merge_indexers(positive_df, indexers, rank_cols)

    np.testing.assert_equal(len(merged_df), 123)
    np.testing.assert_equal(np.sum(merged_df.correct), 6)

    np.testing.assert_equal(np.max(merged_df.rank_0), 11.0)
    np.testing.assert_equal(np.max(merged_df.rank_1), 11.0)
    np.testing.assert_equal(np.max(merged_df.rank_2), 2.0)

    np.testing.assert_equal(np.min(merged_df.rank_0), 1.0)
    np.testing.assert_equal(np.min(merged_df.rank_1), 1.0)
    np.testing.assert_equal(np.min(merged_df.rank_2), -2.0)
