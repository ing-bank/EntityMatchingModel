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

"""Helper function for name matching model save and load"""

from __future__ import annotations

import gc
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, vstack

from emm.helper import spark_installed
from emm.helper.util import groupby

if TYPE_CHECKING:
    from pyspark.ml.linalg import DenseVector, SparseVector

if spark_installed:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window


def stack_features(matrices, dense=False):
    """Combine multiple (>=1) feature matrices to a larger one"""
    if dense:
        # if matrices contains only 1 element, we still return a list of 1 matrix, for type consistency
        return np.vstack(matrices)
    return vstack(matrices) if len(matrices) > 1 else matrices[0]


def collect_matrix(dist_matrix, uid_col, feature_col, blocking_col=None):
    """Convert a distributed matrix (spark.sql.Column of pyspark.ml.linalg.SparseVector) to a
    local matrix (scipy.sparse.csr matrix), and keep ground truth indices along with matrix:
    - the returned indices is a 1d np.array containing the ground-truth uid
    - the return matrix has the same integer position index as the indices
    In the blocking case it returns dicts where the key is the block and the value is the same as describe above.
    """

    def spark_row_to_local(row, blocking_col=None):
        row = row.asDict()
        if blocking_col is None:
            return row[uid_col], as_matrix(row[feature_col], False), None
        return row[uid_col], as_matrix(row[feature_col], False), row[blocking_col]

    def vstack_worker(iterator, dense=False):
        rows = list(iterator)
        if rows:
            indices, matrices, blocks = zip(*((x[0], x[1], x[2]) for x in rows))
            yield indices, stack_features(matrices, dense), blocks

    # Select only the necessary columns to minimize serialization
    if blocking_col is None:
        dist_matrix = dist_matrix.select(uid_col, feature_col)
    else:
        dist_matrix = dist_matrix.select(uid_col, feature_col, blocking_col)

    # We aggregate and convert Spark vectors into numpy matrix in parallel for each partition, and then we collect all partitions.
    # Remark: partial() is used to set function parameter without a lambda, and without local variable.
    local_matrix_parts = (
        dist_matrix.rdd.map(partial(spark_row_to_local, blocking_col=blocking_col))
        .mapPartitions(lambda it: vstack_worker(it, False))
        .collect()
    )

    uids, matrices, blocks = zip(*local_matrix_parts)

    # we use numpy array because smaller in size than list and necessary for groupby
    indices = np.array(list(chain(*uids)))
    indices = down_casting_int(indices)
    blocks = list(chain(*blocks))
    matrix = stack_features(matrices, False)
    if blocking_col is None:
        gc.collect()  # Free some memory on the driver
        return indices, matrix.T

    # The data should be "sliceable/maskable" for groupby
    indicies = groupby(indices, blocks, postprocess_func=np.array)
    matrix = groupby(matrix, blocks, postprocess_func=lambda x: x.T)
    gc.collect()  # Free some memory on the driver
    return indicies, matrix


def curry(func, *args):
    """Curry a function so that only a single argument remains. This is required for rdd.mapPartitions()"""
    return lambda iterator: func(iterator, *args)


def flatten_df(nested_df, nested_cols, separator="_", keep_root_name=True):
    """Flatten all nested columns that are in nested_cols
    nested_cols: either one struct column or list of struct columns
    """
    if not isinstance(nested_cols, list):
        nested_cols = [nested_cols]
    flat_cols = [c for c in nested_df.columns if c not in nested_cols]

    def new_name(nc, c):
        if keep_root_name:
            return nc + separator + c
        return c

    return nested_df.select(
        flat_cols
        + [
            F.col(nc + "." + c).alias(new_name(nc, c))
            for nc in nested_cols
            for c in nested_df.select(nc + ".*").columns
        ]
    )


def explode_candidates(df, with_rank=True, separator="_"):
    """Change data structure from one row per names_to_match with a list candidates
    to one row per candidate
    """
    if with_rank:
        df = df.select("*", F.posexplode("candidates").alias("_pos", "candidate")).drop("candidates")
        # pos starts at 0
        return df.withColumn(f"candidate{separator}rank", F.expr("_pos +1")).drop("_pos")
    return df.select("*", F.explode("candidates").alias("candidate")).drop("candidates")


def down_casting_int(a: np.array):
    """Automatically downcast integer to the smallest int type
    according the minimum and maximum value of the array
    """
    a_min = a.min()
    a_max = a.max()

    types = [np.int8, np.int16, np.int32, np.int64]
    for t in types:
        info = np.iinfo(t)
        if info.min < a_min and info.max > a_max:
            return a.astype(t)

    return a


def take_topn_per_group(df, n, group, order_by=None, method="exactly", keep_col=True):
    """Take only top-n rows per group to remove data skewness.
    order_by should be a tuple like: (F.col('C'), )
    Method can have these values:
    'at_most' can in some situation remove accounts
    'at_least_n_different_order_values' can lead to some skewness still
    'at_least' can lead to some skewness still

    When to use "at_least_n_different_order_values" dense_rank() over "exactly" row_number():
    - if we have multiple names with same count_distinct at the limit, we have no information to pick one vs the other (but 'at_most' is better here)
    - if we have multiple rows that are linked together, like exploded candidates list
    - if you have within an account more than n different names with the same exact order value
    """
    if order_by is None:
        # orderBy is mandatory for Window.partitionBy()
        order_by = (F.rand(),)
    window = Window.partitionBy(group).orderBy(*order_by)

    if method == "at_least":
        f = F.rank()
    elif method == "at_least_n_different_order_values":
        f = F.dense_rank()
    elif method == "exactly":
        f = F.row_number()
    elif method == "at_most":
        f = F.count("*")
    else:
        msg = f"Unknown method '{method}'"
        raise ValueError(msg)

    col_name = f"{group}_rank"

    df = df.withColumn(col_name, f.over(window))
    df = df.filter(f"{col_name} <= {n}")

    if not keep_col:
        return df.drop(col_name)

    return df


def as_matrix(vec: DenseVector | SparseVector, dense: bool = False):
    """Convert a pyspark.ml.linalg.DenseVector to numpy matrix (only a single row)
    Convert a pyspark.ml.linalg.SparseVector to scipy.sparse.csr matrix (only a single row)

    Args:
        vec: vector
        dense: bool

    Returns:
        Numpy matrix / scipy csr matrix
    """
    if dense:
        return vec.toArray()

    return csr_matrix((vec.values, vec.indices, np.array([0, len(vec.values)])), (1, vec.size), dtype=np.float32)


def dot_product(vec1: SparseVector | DenseVector, vec2: SparseVector | DenseVector) -> float:
    """Dot product of two pyspark.ml.linalg.SparseVector for example
    It works for pyspark.ml*.linalg.*Vector.dot
    """
    return float(vec1.dot(vec2))
