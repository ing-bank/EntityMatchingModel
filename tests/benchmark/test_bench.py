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

"""Benchmarking scripts (using pytest-benchmark).
By default, those tests are skipped, to run it use:

    pytest --benchmark-enable tests
"""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from emm.data.create_data import retrieve_kvk_test_sample
from emm.features.features_name import calc_name_features
from emm.features.features_vocabulary import compute_vocabulary_features
from emm.features.pandas_feature_extractor import PandasFeatureExtractor
from emm.helper import spark_installed
from emm.indexing.pandas_cos_sim_matcher import PandasCosSimIndexer
from emm.indexing.pandas_normalized_tfidf import PandasNormalizedTfidfVectorizer
from emm.indexing.pandas_sni import PandasSortedNeighbourhoodIndexer
from emm.pipeline.pandas_entity_matching import PandasEntityMatching
from emm.preprocessing.pandas_preprocessor import PandasPreprocessor

if spark_installed:
    from emm.indexing.spark_cos_sim_matcher import SparkCosSimIndexer
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


def num(x):
    if isinstance(x, str):
        x = int(x)
    if x >= 10**6:
        if x % 10**6 == 0:
            return f"{x // 10**6}m"
        return f"{x / 10**6:.1f}m"
    if x >= 10**4:
        if x % 10**3 == 0:
            return f"{x // 10**3}k"
        return f"{x / 10**3:.1f}k"
    return str(x)


@pytest.fixture()
def kvk_dataset():
    _, df = retrieve_kvk_test_sample()
    df = df.rename(columns={"Name": "name", "Index": "id"})
    df["id"] = range(len(df))
    df["is_gt"] = df["id"].map(lambda x: x % 2 == 0)
    return df


def increase_dataset(df, n):
    """Increases dataset by adding new names.
    New names are created by adding additional characters to each word in each batch.
    """
    original_names = df["name"]

    def fix_words(name, ii):
        return " ".join(f"{x}{ii}" for x in name.split(" "))

    names = [
        original_names.map(partial(fix_words, ii=chr(ord("a") + batch_num)))
        for batch_num in range(n // len(original_names))
    ]
    names = pd.concat([original_names, *names]).values[:n]
    return pd.DataFrame({"name": names, "id": range(len(names))})


def split_dataset(df, gt_n, names_n):
    assert len(df) >= gt_n + names_n
    gt = df.sample(n=gt_n, random_state=1)
    names = df[~df.index.isin(gt.index)].sample(n=names_n, random_state=2)
    assert len(gt) == gt_n
    assert len(names) == names_n
    return gt, names


@pytest.mark.parametrize("gt_size", [10**5, 5 * 10**5])
def test_bench_pandas_name_preprocessing(benchmark, gt_size, kvk_dataset):
    benchmark.extra_info["title"] = "Name preprocessing (pipeline=preprocess_merge_abbr)"
    benchmark.extra_info["label"] = f"n={num(gt_size)}"
    data = increase_dataset(kvk_dataset, gt_size)
    p = PandasPreprocessor(preprocess_pipeline="preprocess_merge_abbr")
    benchmark.pedantic(lambda: p.transform(data), rounds=1)


@pytest.mark.parametrize(
    ("stage", "size"), [("fit", 10**5), ("fit", 5 * 10**5), ("transform", 10**5), ("transform", 5 * 10**5)]
)
def test_bench_pandas_tfidf(benchmark, stage, size, kvk_dataset):
    benchmark.extra_info["title"] = f"TF-IDF ({stage})"
    benchmark.extra_info["label"] = f"n={num(size)}"
    names = increase_dataset(kvk_dataset, size)["name"]
    vec = PandasNormalizedTfidfVectorizer(analyzer="word")
    if stage == "fit":
        benchmark.pedantic(lambda: vec.fit(names), rounds=1)
    else:
        vec.fit(names)
        benchmark.pedantic(lambda: vec.transform(names), rounds=1)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("mode", "stage", "gt_size", "n_jobs"),
    [
        ("spark", "transform", 5 * 10**5, 1),
        ("pandas", "fit", 10**5, 1),
        ("pandas", "fit", 5 * 10**5, 1),
        ("pandas", "transform", 5 * 10**5, 1),
        ("pandas", "transform", 5 * 10**5, 8),
        ("pandas", "transform", 5 * 10**5, 12),
    ],
)
def test_bench_cossim_indexer(benchmark, spark_session, kvk_dataset, mode, stage, gt_size, n_jobs):
    n_size = 10**4
    benchmark.extra_info["title"] = f"{mode.capitalize()}CosSimIndexer ({stage})"
    benchmark.extra_info["label"] = f"gt_size={num(gt_size)}" + (f" n={num(n_size)}" if stage == "transform" else "")
    data = increase_dataset(kvk_dataset, gt_size + n_size)
    # preprocess name to be able to compare timing with test_bench_pandas_name_matching
    data = PandasPreprocessor(preprocess_pipeline="preprocess_merge_abbr").transform(data)
    gt, names = split_dataset(data, gt_size, n_size)

    if mode == "pandas":
        idx = PandasCosSimIndexer(
            input_col="preprocessed", tokenizer="words", cos_sim_lower_bound=0.1, num_candidates=10, n_jobs=n_jobs
        )
    else:
        gt["uid"] = range(len(gt))
        names["uid"] = range(len(names))
        gt = spark_session.createDataFrame(gt)
        names = spark_session.createDataFrame(names)
        idx = SparkCosSimIndexer(
            {
                "cos_sim_lower_bound": 0.1,
                "tokenizer": "words",
                "num_candidates": 10,
                "ngram": 1,
                "max_features": 2**20,
                "binary_countvectorizer": True,
                "streaming": False,
                "blocking_func": None,
                "indexer_id": 0,
                "keep_all_cols": False,
            }
        )
    if stage == "fit":
        benchmark.pedantic(lambda: idx.fit(gt), rounds=1)
    else:
        m = idx.fit(gt)
        if mode == "pandas":
            _ = benchmark.pedantic(lambda: m.transform(names), rounds=1)
        else:
            _ = benchmark.pedantic(lambda: m.transform(names).toPandas(), rounds=1)


@pytest.mark.parametrize("gt_size", [10**5, 5 * 10**5])
def test_bench_pandas_sni_indexer(benchmark, gt_size, kvk_dataset):
    n_size = 10**4
    benchmark.extra_info["title"] = "TF-IDF (transform)"
    benchmark.extra_info["label"] = f"gt_size={num(gt_size)} n={num(n_size)}"
    data = increase_dataset(kvk_dataset, gt_size + n_size)
    gt, names = split_dataset(data, gt_size, n_size)

    idx = PandasSortedNeighbourhoodIndexer(input_col="name", window_length=5)
    idx.fit(gt)
    benchmark.pedantic(lambda: idx.transform(names), rounds=1)


def gen_candidates(df, size, num_candidates_per_uid=10, seed=1):
    data = increase_dataset(df, size * 2)
    names1, names2 = split_dataset(data, size, size)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "name1": names1["name"].values,
            "name2": names2["name"].values,
            "uid": rng.integers(0, size // num_candidates_per_uid, size),
            "gt_uid": range(size),
            "score": rng.random(size),
        }
    )


@pytest.mark.parametrize("size", [10**4])
def test_bench_pandas_calc_features(benchmark, size, kvk_dataset):
    benchmark.extra_info["title"] = "Calc features"
    benchmark.extra_info["label"] = f"n={num(size)}"
    candidates = gen_candidates(kvk_dataset, size)

    obj = PandasFeatureExtractor(
        name1_col="name1", name2_col="name2", uid_col="uid", gt_uid_col="gt_uid", score_columns=["score"]
    )
    benchmark.pedantic(lambda: obj.transform(candidates), rounds=1)


@pytest.mark.parametrize("size", [10**4])
def test_bench_pandas_calc_name_features(benchmark, size, kvk_dataset):
    benchmark.extra_info["title"] = "Calc name features"
    benchmark.extra_info["label"] = f"n={num(size)}"
    pfe = PandasFeatureExtractor()
    candidates = gen_candidates(kvk_dataset, size)
    res = benchmark.pedantic(
        lambda: calc_name_features(candidates, funcs=pfe.name_features, name1="name1", name2="name2"), rounds=1
    )
    assert len(res) == len(candidates)


@pytest.mark.parametrize("size", [10**4])
def test_bench_pandas_calc_hits_features(benchmark, size, kvk_dataset):
    benchmark.extra_info["title"] = "Calc hits features"
    benchmark.extra_info["label"] = f"n={num(size)}"
    candidates = gen_candidates(kvk_dataset, size)
    res = benchmark.pedantic(lambda: compute_vocabulary_features(candidates, col1="name1", col2="name2"), rounds=1)
    assert len(res) == len(candidates)


@pytest.mark.parametrize(
    ("stage", "gt_size", "supervised_on"),
    [
        ("fit", 10**5, False),
        ("fit", 2 * 10**5, False),
        ("transform", 10**5, False),
        ("transform", 2 * 10**5, False),
        ("transform", 2 * 10**5, True),
    ],
)
def test_bench_pandas_name_matching(stage, benchmark, gt_size, supervised_on, kvk_dataset, supervised_model):
    n_size = 10**4
    benchmark.extra_info["title"] = f"Name matching ({stage})"
    if stage == "transform":
        benchmark.extra_info["title"] += " " + (
            "with supervised model" if supervised_on else "without supervised model"
        )

    benchmark.extra_info["label"] = f"gt_size={num(gt_size)}" + (f" n={num(n_size)}" if stage == "transform" else "")
    data = increase_dataset(kvk_dataset, gt_size + n_size)
    gt, names = split_dataset(data, gt_size, n_size)
    assert len(gt) == gt_size
    assert len(names) == n_size

    em = PandasEntityMatching(
        {
            "preprocessor": "preprocess_merge_abbr",
            "entity_id_col": "id",
            "aggregation_layer": False,
            "name_only": True,
            "supervised_on": supervised_on,
            "supervised_model_dir": supervised_model[2].parent,
            "supervised_model_filename": supervised_model[2].name if supervised_on else None,
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": "words",
                    "n_jobs": 8,
                    "num_candidates": 10,
                    "cos_sim_lower_bound": 0.1,
                }
            ],
        }
    )
    if stage == "fit":
        benchmark.pedantic(lambda: em.fit(gt), rounds=1)
    else:
        em.fit(gt)
        _ = benchmark.pedantic(lambda: em.transform(names), rounds=1)


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("mode", "stage", "gt_size"),
    [
        ("spark", "transform", 10**4),
        ("pandas", "transform", 10**4),
        ("pandas", "fit", 10**5),
        ("pandas", "fit", 2 * 10**5),
        ("pandas", "transform", 10**5),
        ("pandas", "transform", 2 * 10**5),
    ],
)
def test_bench_name_matching_with_3_indexers(benchmark, kvk_dataset, spark_session, mode, stage, gt_size):
    n_size = 10**3
    benchmark.extra_info["title"] = f"{mode.capitalize()} Name matching ({stage})"
    benchmark.extra_info["label"] = f"mode={mode} gt_size={num(gt_size)}" + (
        f" n={num(n_size)}" if stage == "transform" else ""
    )
    data = increase_dataset(kvk_dataset, gt_size + n_size)
    gt, names = split_dataset(data, gt_size, n_size)
    assert len(gt) == gt_size
    assert len(names) == n_size
    if mode == "spark":
        gt["uid"] = range(len(gt))
        names["uid"] = range(len(names))
        gt = spark_session.createDataFrame(gt)
        names = spark_session.createDataFrame(names)

    n_jobs = 8 if mode == "pandas" else 1
    em = ({"pandas": PandasEntityMatching, "spark": SparkEntityMatching}[mode])(
        {
            "preprocessor": "preprocess_merge_abbr",
            "entity_id_col": "id",
            "aggregation_layer": False,
            "name_only": True,
            "supervised_on": False,
            "supervised_model_dir": ".",
            "supervised_model_filename": None,
            "indexers": [
                {
                    "type": "cosine_similarity",
                    "tokenizer": "words",
                    "n_jobs": n_jobs,
                    "num_candidates": 10,
                    "cos_sim_lower_bound": 0.1,
                },
                {
                    "type": "cosine_similarity",
                    "tokenizer": "characters",
                    "ngram": 2,
                    "n_jobs": n_jobs,
                    "num_candidates": 10,
                    "cos_sim_lower_bound": 0.1,
                },
                {"type": "sni", "window_length": 5},
            ],
        }
    )
    if stage == "fit":
        benchmark.pedantic(lambda: em.fit(gt), rounds=1)
    else:
        m = em.fit(gt)
        if mode == "spark":
            _ = m.transform(names.limit(1)).toPandas()  # to force fit
            _ = benchmark.pedantic(lambda: m.transform(names).toPandas(), rounds=1)
        else:
            _ = benchmark.pedantic(lambda: m.transform(names), rounds=1)
