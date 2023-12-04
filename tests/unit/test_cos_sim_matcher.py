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
import pytest
from sklearn.preprocessing import normalize

from emm.helper import spark_installed
from tests.utils import create_test_data

if spark_installed:
    import pyspark.sql.functions as F
    from pyspark.ml.linalg import SparseVector

    from emm.indexing.spark_cos_sim_matcher import SparkCosSimMatcher
    from emm.pipeline.spark_entity_matching import SparkEntityMatching


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_cos_sim_matcher(spark_session):
    # prepare data up to the point that the CosSimMatcher is used
    nm = SparkEntityMatching(
        parameters={
            "preprocessor": "preprocess_name",
            "indexers": [{"type": "cosine_similarity", "tokenizer": "characters", "ngram": 3, "num_candidates": 1}],
            "entity_id_col": "id",
            "uid_col": "uid",
            "name_col": "name",
            "supervised_on": False,
            "name_only": True,
            "keep_all_cols": True,
        }
    )

    # Turn off cossim
    stages = nm.pipeline.getStages()
    stages[1].indexers[0].cossim = None

    ground_truth, _ = create_test_data(spark_session)
    names = nm.fit(ground_truth).transform(ground_truth)
    assert names.select("uid").distinct().count() == names.count()

    # Fit/Create CosSimMatcher
    csm = (
        SparkCosSimMatcher(num_candidates=3, cos_sim_lower_bound=0.2, index_col="id", uid_col="uid", name_col="name")
        ._set(inputCol="features")
        ._set(outputCol="candidates")
        .fit(names)
    )

    assert csm.gt_features_csr_bc is not None
    assert csm.gt_indices_bc is not None

    # Sanity check that ground truth is matched correctly back to ground truth
    matched = csm.transform(names).toPandas()

    assert matched["indexer_score"].fillna(0).between(0, 1 + 1e-6, inclusive="both").all()
    assert (matched["uid"] == matched["gt_uid"]).sum() == names.count()


###
# The following functions are testing only CosSimMatcher with simple data on both Dense and Sparse
def create_simple_data():
    indexes = [110, 120, 130, 140, 150, 160]  # Simulate Grid ID

    features = np.array(
        [
            [0, 0],  # 110
            [0, 1],  # 120  |
            [1, 0],  # 130   _
            [1, 1],  # 140   /
            [1, 2],  # 150   / closer to |
            [1, 3],  # 160   / even more closer to |
        ]
    )

    features = normalize(features, axis=1, norm="l2")
    indexes_features = np.column_stack((indexes, features))

    return indexes_features, indexes


def cos_sim_assert(spark_session, features_df, indexes, num_candidates, num_partitions):
    lower_bound = 0.1
    csm = (
        SparkCosSimMatcher(
            num_candidates=num_candidates,
            cos_sim_lower_bound=lower_bound,
            index_col="id",
            uid_col="id",
            name_col="id_str",
            streaming=False,
        )
        ._set(outputCol="candidates")
        ._set(inputCol="vector")
    )

    features_df = features_df.repartition(num_partitions)

    spark_session.sql(f"set spark.sql.shuffle.partitions={num_partitions}").collect()
    spark_session.sql(f"set spark.default.parallelism={num_partitions}").collect()

    csm_model = csm.fit(features_df)

    df = csm_model.transform(features_df)
    df_pd = df.toPandas()

    # Check that each vector is most similar with him self
    for i in indexes[1:]:  # except first vector which is null
        vect = df_pd.query("id == " + str(i)).iloc[0]
        assert vect["gt_uid"] == i

    # Check similar vector
    vect120 = df_pd.query("id == 120")
    assert vect120.iloc[1]["gt_uid"] == 160  # 1st closest vector after himself
    assert vect120.iloc[2]["gt_uid"] == 150  # 2nd closest vector after himself


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_cos_sim_matcher_sparse(spark_session):
    # Data
    indexes_features, indexes = create_simple_data()

    # Create Spark DataFrame with Sparse Vector
    dim = 2
    features_vector = (
        (int(x[0]), SparseVector(dim, {k: float(v) for k, v in enumerate(x[1:])})) for x in indexes_features
    )
    features_df = spark_session.createDataFrame(features_vector, schema=["id", "vector"])
    features_df = features_df.withColumn("id_str", F.col("id"))

    # Trying edge cases: num_partitions < and > than number of rows
    param_list = [
        {"num_candidates": 3, "num_partitions": 4},
        {"num_candidates": 10, "num_partitions": 1},
        {"num_candidates": 10, "num_partitions": 10},
        {"num_candidates": 3, "num_partitions": 10},
        {"num_candidates": 10, "num_partitions": 4},
    ]

    for param in param_list:
        cos_sim_assert(spark_session=spark_session, features_df=features_df, indexes=indexes, **param)
