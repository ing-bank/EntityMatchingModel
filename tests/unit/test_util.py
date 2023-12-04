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
from scipy.sparse import csr_matrix

from emm.data.create_data import retrieve_kvk_test_sample
from emm.helper import spark_installed
from emm.helper.spark_utils import check_uid
from emm.helper.util import rename_columns
from emm.indexing.spark_indexing_utils import collect_matrix, down_casting_int, groupby
from tests.utils import create_test_data, get_n_top_sparse

if spark_installed:
    from emm.indexing.spark_cos_sim_matcher import add_blocking_col
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


def test_retrieve_kvk_test_sample():
    path, df = retrieve_kvk_test_sample()
    assert len(path.name) > 0
    assert len(df) == 6800


def test_get_csr_n_top():
    mat = csr_matrix(np.arange(0, 1.01, 0.1))
    res = get_n_top_sparse(mat, 3)
    assert [row_ix for row_ix, _ in res] == [10, 9, 8]
    res = get_n_top_sparse(mat, 6)
    assert [row_ix for row_ix, _ in res] == [10, 9, 8, 7, 6, 5]
    res = get_n_top_sparse(mat, 1)
    assert [row_ix for row_ix, _ in res] == [10]
    mat = csr_matrix(np.arange(1, -0.01, -0.1))
    res = get_n_top_sparse(mat, 4)
    assert [row_ix for row_ix, _ in res] == [0, 1, 2, 3]
    empty_matrix = csr_matrix([0] * 10)
    res = get_n_top_sparse(empty_matrix, 4)
    assert res is None
    mat = csr_matrix([0.5, 0.8, 1, 0.2])
    res = get_n_top_sparse(mat, 10)  # larger n than elements in list
    assert [row_ix for row_ix, _ in res] == [2, 1, 0, 3]


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_check_uid(spark_session):
    sdf = spark_session.createDataFrame([["a"], ["b"], ["c"]], ["name"])

    res = check_uid(sdf, "uid").toPandas()
    assert "uid" in res.columns
    assert res["uid"].nunique() == len(res)


def test_groupby():
    data = np.array(range(0, 100, 10))
    groups = ["a", "b"] * 5

    res1 = groupby(data, groups)
    assert set(res1.keys()) == {"a", "b"}
    np.testing.assert_array_equal(res1["a"], [0, 20, 40, 60, 80])
    np.testing.assert_array_equal(res1["b"], [10, 30, 50, 70, 90])

    res2 = groupby(data, groups, postprocess_func=sum)
    assert set(res2.keys()) == {"a", "b"}
    assert res2["a"] == sum([0, 20, 40, 60, 80])
    assert res2["b"] == sum([10, 30, 50, 70, 90])

    # Test only 1 element in data
    data = csr_matrix((1, 4), dtype=np.int8)
    groups = ["a"] * 1

    res1 = groupby(data, groups)
    assert set(res1.keys()) == {"a"}
    assert (res1["a"] != data).nnz == 0


def test_down_casting_int():
    a = np.array([8193, 8222222], dtype=np.int64)
    a1 = down_casting_int(a)
    assert a1.dtype == np.int32

    b = np.array([8193, 8222], dtype=np.int64)
    b1 = down_casting_int(b)
    assert b1.dtype == np.int16


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize("blocking_func", [None, lambda x: x.strip().lower()[0]])
def test_collect_matrix_type(spark_session, blocking_func):
    blocking_col = None if blocking_func is None else "block"

    # prepare data up to the point that the CosSimMatcher is used
    em = SparkEntityMatching(
        parameters={
            "preprocessor": "preprocess_merge_abbr",
            "indexers": [
                {"type": "cosine_similarity", "tokenizer": "characters", "ngram": 2, "blocking_func": blocking_func}
            ],
            "entity_id_col": "id",
            "uid_col": "uid",
            "name_col": "name",
            "name_only": True,
            "supervised_on": False,
            "keep_all_cols": True,
        }
    )
    ground_truth, _ = create_test_data(spark_session)

    # Turn off cossim
    stages = em.pipeline.getStages()
    stages[1].indexers[0].cossim = None

    em.fit(ground_truth)

    names_to_match = spark_session.createDataFrame(
        [["ABC", 1, 100], ["Eddie Eagle", 2, 101], ["Tzu Sun", 3, 102]], ["name", "id", "uid"]
    )

    names_to_match = em.transform(names_to_match)
    names_to_match = add_blocking_col(names_to_match, "preprocessed", blocking_col, blocking_func)

    gt_indices, gt_features = collect_matrix(names_to_match, "uid", "features", blocking_col=blocking_col)

    if blocking_func is None:
        gt_features_dtype = gt_features.dtype
        gt_indices_dtype = gt_indices.dtype
    else:
        gt_features_dtype = next(iter(gt_features.values())).dtype
        gt_indices_dtype = next(iter(gt_indices.values())).dtype

    assert gt_features_dtype == np.float32
    assert gt_indices_dtype == np.int8


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_rename_columns(spark_session):
    columns = ["a", "b", "c"]
    sdf = spark_session.createDataFrame([(1, 2, 3)], columns)
    df = pd.DataFrame(columns=columns)
    for mapping, expected in [
        ([("a", "aa")], {"aa", "b", "c"}),
        ([("a", "a1"), ("a", "a2")], {"a1", "a2", "b", "c"}),
        ([("a", "a"), ("a", "a1")], {"a", "a1", "b", "c"}),
    ]:
        new_df = rename_columns(df.copy(), mapping)
        assert set(new_df.columns) == expected
        new_sdf = rename_columns(sdf, mapping)
        assert set(new_sdf.columns) == expected
