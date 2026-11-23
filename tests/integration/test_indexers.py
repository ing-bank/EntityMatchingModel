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

import pandas as pd
import pytest

from emm.helper import spark_installed
from emm.indexing.pandas_cos_sim_matcher import PandasCosSimIndexer
from emm.indexing.pandas_sni import PandasSortedNeighbourhoodIndexer, SortedNeighbourhood
from emm.pipeline.pandas_entity_matching import PandasEntityMatching
from tests.utils import read_markdown

from .test_pandas_em import split_gt_and_names

if spark_installed:
    from emm.indexing.spark_cos_sim_matcher import SparkCosSimIndexer
    from emm.indexing.spark_sni import SparkSortedNeighbourhoodIndexer
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


def simplify_indexer_result(res, gt):
    return pd.concat(
        [gt.loc[res["gt_uid"]]["name"].reset_index(drop=True), res["rank"].reset_index(drop=True)], axis=1
    ).values.tolist()


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (1, [(40, 3), (40, 9), (10, 1), (20, 3), (20, 9)]),
        (3, [(10, 8), (40, 3), (40, 9), (10, 1), (20, 3), (20, 9), (40, 8), (20, 8), (50, 3), (50, 9)]),
        (
            5,
            [
                (10, 3),
                (10, 9),
                (10, 8),
                (40, 3),
                (40, 9),
                (10, 1),
                (20, 3),
                (20, 9),
                (40, 8),
                (20, 8),
                (50, 3),
                (50, 9),
                (40, 1),
                (20, 1),
                (50, 8),
            ],
        ),
        (
            9,
            [
                (10, 3),
                (10, 9),
                (10, 8),
                (40, 3),
                (40, 9),
                (10, 1),
                (20, 3),
                (20, 9),
                (40, 8),
                (20, 8),
                (50, 3),
                (50, 9),
                (40, 1),
                (20, 1),
                (50, 8),
                (50, 1),
            ],
        ),
    ],
)
def test_sorted_neighbourhood_candidates(window, expected):
    left = pd.DataFrame({"left_name": ["c", "a", None, "c", "d"]}, index=pd.Index([40, 10, 30, 20, 50], name="id"))
    right = pd.DataFrame({"right_name": ["b", "c", None, "a", "c"]}, index=pd.Index([8, 3, 5, 1, 9], name="id"))
    indexer = SortedNeighbourhood(left_on="left_name", right_on="right_name", window=window)
    result = indexer.index(left, right)
    assert sorted(result.tolist()) == sorted(expected)
    assert result.names == ["id_1", "id_2"]
    assert indexer.sorting_key_values.tolist() == ["a", "b", "c", "d"]


@pytest.mark.parametrize("empty_side", ["left", "right", "both"])
def test_sorted_neighbourhood_empty_inputs(empty_side):
    left = pd.DataFrame({"name": ["a"]}, index=pd.Index([1], name="id"))
    right = pd.DataFrame({"name": ["b"]}, index=pd.Index([2], name="id"))
    if empty_side in ("left", "both"):
        left = left.iloc[:0]
    if empty_side in ("right", "both"):
        right = right.iloc[:0]

    result = SortedNeighbourhood(left_on="name").index(left, right)
    assert result.empty
    assert result.names == ["id_1", "id_2"]


@pytest.mark.parametrize("window", [0, -1, 2, 1.5])
def test_sorted_neighbourhood_rejects_invalid_window(window):
    names = pd.DataFrame({"name": ["a"]})
    with pytest.raises(ValueError, match="window is not a positive and odd integer"):
        SortedNeighbourhood(left_on="name", window=window).index(names, names)


def test_sorted_neighbourhood_rejects_duplicate_indexes():
    names = pd.DataFrame({"name": ["a", "b"]}, index=[1, 1])
    with pytest.raises(ValueError, match="index of DataFrame is not unique"):
        SortedNeighbourhood(left_on="name").index(names, names)


def test_sni_indexer_with_empty_query():
    indexer = PandasSortedNeighbourhoodIndexer("name").fit(pd.DataFrame({"name": ["a"]}))
    result = indexer.transform(pd.DataFrame({"name": pd.Series([], dtype="str")}))
    assert result.empty
    assert result.columns.tolist() == ["uid", "gt_uid", "rank", "score"]


def test_sni_indexer(sample_gt):
    idx = PandasSortedNeighbourhoodIndexer("name", window_length=3)
    idx.fit(sample_gt)
    for name, expected_result in [
        ("a", [["a", 0], ["b", 1]]),
        ("c", [["b", -1], ["c", 0], ["d", 1]]),
        ("ca", [["c", -1], ["d", 1]]),
        ("e", [["d", -1], ["f", 1]]),
    ]:
        query = pd.DataFrame({"name": [name]})
        cand = idx.transform(query)
        assert isinstance(idx.sni, SortedNeighbourhood)
        actual_result = simplify_indexer_result(cand, sample_gt)
        assert expected_result == actual_result

        data_for_calc_score = pd.concat(
            [
                query.loc[cand["uid"]]["name"].rename("name1").reset_index(drop=True),
                sample_gt.loc[cand["gt_uid"]]["name"].rename("name2").reset_index(drop=True),
            ],
            axis=1,
        )
        scores = idx.calc_score(data_for_calc_score["name1"], data_for_calc_score["name2"])
        assert all(cand["score"] == scores["score"])
        assert all(cand["rank"] == scores["rank"])


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_sni_indexer_spark(sample_gt, spark_session):
    sample_gt["uid"] = range(len(sample_gt))
    sample_gt = sample_gt.rename({"id": "entity_id"}, axis=1)
    sample_gt = sample_gt.sample(frac=1)  # Shuffle rows
    sample_gt_sdf = spark_session.createDataFrame(sample_gt)
    sample_gt_sdf = sample_gt_sdf.repartition(10)

    idx = SparkSortedNeighbourhoodIndexer(window_length=3)._set(outputCol="candidates")._set(inputCol="name")

    model = idx.fit(sample_gt_sdf)

    # If you query 1 by 1:    'c' gives 'b,c,d'
    # If you query all names: 'c' gives 'c,b'
    # that is because of overlapping name between ground-truth and name-to-match
    names_pd = pd.DataFrame(["a", "c", "ca", "e"], columns=["name"])
    names_pd["uid"] = range(len(names_pd))
    names_pd = names_pd.sample(frac=1)  # Shuffle rows
    names = spark_session.createDataFrame(names_pd)
    names = names.repartition(10)

    cand = model.transform(names)
    cand_pd = cand.toPandas()
    cand_pd = cand_pd.merge(names_pd, on="uid").merge(sample_gt.add_prefix("gt_"), on="gt_uid")
    cand_pd = cand_pd.sort_values(["name", "gt_name"], ascending=[True, True])

    cand_result_pd = cand_pd[["name", "gt_name", "indexer_score", "indexer_rank"]].reset_index(drop=True)

    cand_excepted_pd = read_markdown(
        """
| name   | gt_name   |   indexer_score |   indexer_rank |
|:-------|:----------|----------------:|---------------:|
| a      | a         |             1   |              0 |
| a      | b         |             0.5 |              1 |
| c      | b         |             0.5 |             -1 |
| c      | c         |             1   |              0 |
| ca     | c         |             0.5 |             -1 |
| ca     | d         |             0.5 |              1 |
| e      | d         |             0.5 |             -1 |
| e      | f         |             0.5 |              1 |
"""
    )
    pd.testing.assert_frame_equal(cand_result_pd, cand_excepted_pd, check_dtype=False)

    assert cand_pd["indexer_rank"].between(-1, 1, inclusive="both").all()
    assert len(cand_pd.query("indexer_rank == 0")) == 2
    assert len(cand_pd.query("indexer_rank == 1")) == 3
    assert len(cand_pd.query("indexer_rank == -1")) == 3


def test_sni_indexer_with_mapping():
    gt = pd.DataFrame({"name": ["abc", "cba", "bbb", "ddd"], "id": range(4)})
    idx = PandasSortedNeighbourhoodIndexer("name", window_length=3, mapping_func=lambda x: x[::-1])
    idx.fit(gt)
    for name, expected_result in [
        ("xxc", [["abc", -1], ["ddd", 1]]),
        ("axx", [["ddd", -1]]),
        ("cba", [["cba", 0], ["bbb", 1]]),
    ]:
        query = pd.DataFrame({"name": [name]}, index=[123])
        cand = idx.transform(query)
        actual_result = simplify_indexer_result(cand, gt)
        assert expected_result == actual_result

        data_for_calc_score = pd.concat(
            [
                query.loc[cand["uid"]]["name"].rename("name1").reset_index(drop=True),
                gt.loc[cand["gt_uid"]]["name"].rename("name2").reset_index(drop=True),
            ],
            axis=1,
        )
        scores = idx.calc_score(data_for_calc_score["name1"], data_for_calc_score["name2"])
        assert all(cand["score"] == scores["score"])
        assert all(cand["rank"] == scores["rank"])


def test_sni_indexer_even_window(sample_gt):
    # expect odd integer as window_length
    with pytest.raises(ValueError, match="SNI window should be odd integer"):
        _ = PandasSortedNeighbourhoodIndexer("name", window_length=4)


def test_sni_indexer_within_em(sample_gt):
    em_params = {
        "preprocessor": "preprocess_name",
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 3}],
    }

    p = PandasEntityMatching(em_params)
    p = p.fit(sample_gt)
    candidates = p.transform(sample_gt)
    assert "score_0" in candidates.columns
    assert "rank_0" in candidates.columns


def test_sni_calc_score(sample_gt, sample_nm):
    em_params = {
        "preprocessor": "preprocess_name",
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "supervised_on": False,
        "indexers": [{"type": "sni", "window_length": 5}],
    }

    p = PandasEntityMatching(em_params)
    p = p.fit(sample_gt)
    candidates = p.transform(sample_nm).dropna(subset=["gt_uid"])
    indexer = p.pipeline.named_steps["candidate_selection"].indexers[0]
    scores = indexer.calc_score(candidates["preprocessed"], candidates["gt_preprocessed"])
    assert len(scores) == len(candidates)
    assert all(scores["score"] == candidates["score_0"])
    assert all(scores["rank"] == candidates["rank_0"])


@pytest.fixture()
def sample_gt():
    return pd.DataFrame({"name": ["a", "b", "c", "d", "f"], "id": range(5)})


@pytest.fixture()
def sample_nm():
    return pd.DataFrame({"name": ["a", "ba", "bc", "z"], "id": [10, 20, 30, 40]})


def test_indexer_objects_pandas(kvk_training_dataset):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:10]

    indexers = [
        PandasCosSimIndexer(
            input_col="preprocessed",
            tokenizer="words",
            ngram=1,
            num_candidates=10,
            cos_sim_lower_bound=0.0,
            binary_countvectorizer=True,
        ),
        PandasSortedNeighbourhoodIndexer(input_col="preprocessed", window_length=5),
    ]
    em_params = {"name_only": True, "entity_id_col": "id", "name_col": "name", "indexers": indexers}
    p = PandasEntityMatching(em_params)
    p.fit(gt)
    res = p.transform(names)

    assert len(res) == 117


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_indexer_objects_spark(kvk_training_dataset, spark_session):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:1000]
    names = names[:10]

    sgt = spark_session.createDataFrame(gt)
    snames = spark_session.createDataFrame(names)

    indexers = [
        SparkCosSimIndexer(
            tokenizer="words", ngram=1, num_candidates=10, binary_countvectorizer=True, cos_sim_lower_bound=0.0
        ),
        SparkSortedNeighbourhoodIndexer(window_length=5),
    ]
    em_params = {"name_only": True, "entity_id_col": "id", "name_col": "name", "indexers": indexers}
    p = SparkEntityMatching(em_params)
    p.fit(sgt)
    res = p.transform(snames)

    assert res.count() == 117


def test_naive_indexer_pandas(kvk_training_dataset):
    gt, names = split_gt_and_names(kvk_training_dataset)
    gt = gt[:10]
    names = names[:10]

    em_params = {"name_only": True, "entity_id_col": "id", "name_col": "name", "indexers": [{"type": "naive"}]}
    p = PandasEntityMatching(em_params)
    p.fit(gt)
    res = p.transform(names)

    assert len(res) == len(gt) * len(names)
