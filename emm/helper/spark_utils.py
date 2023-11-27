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

from emm.helper import spark_installed
from emm.loggers.logger import logger

if spark_installed:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F


def logical_repartitioning(
    df: DataFrame, column: str, num_partitions: int | None = None, spark: SparkSession | None = None
) -> DataFrame:
    """Making sure we have all the candidates of a names_to_match (uid) in the same partition,
    we need this for computing rank feature in the pandas UDF.
    Repartition need to be after computation of missing feature, most probably because vectorizer is doing some repartitioning.
    This is needed for a logical reason and Spark data/execution parallelism reason.

    repartition(k, col) will create a dataframe with k partitions using a hash-based partitioner on col.
    repartition with the same number of partitions as before, so related to
    spark.sql.shuffle.partitions and spark.default.parallelism
    repartition in function of partition_size
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    num_partitions = df.rdd.getNumPartitions() if num_partitions is None else num_partitions

    logger.debug(
        f"SparkCandidateSelectionModel: repartitioning from {df.rdd.getNumPartitions()} to {num_partitions} partitions"
    )

    adaptive = spark.conf.get("spark.sql.adaptive.enabled")
    if adaptive != "false":
        logger.warning(
            f"Currently spark.sql.adaptive.enabled='{adaptive}', it MUST be disabled to keep small partitions. "
            "We are disabling it at runtime right now. Remark: Spark UI will not reflect this change."
        )
        spark.sql("SET spark.sql.adaptive.enabled=false").collect()

    return df.repartition(num_partitions, column)


def auto_repartitioning(sdf: DataFrame, partition_size: int | None, *cols):
    """Repartition Spark DataFrame so that it has 'partition_size' rows per partition
    If partition_size==None then no repartitioning is done.
    Returns repartitioned dataframe and size of dataset.
    """
    if partition_size is None:
        return sdf, -1

    logger.info(f"Estimating total dataset size for repartitioning. partition size = {partition_size} records")
    num_records = sdf.rdd.countApprox(timeout=20)
    num_partitions = max(1, num_records // partition_size)
    logger.debug(f"Repartitioning from {sdf.rdd.getNumPartitions()} to {num_partitions} partitions")
    logger.debug(f"Total number of records: {num_records}. Desired number of records per partition: {partition_size}")
    return sdf.repartition(num_partitions, *cols), num_records


def set_spark_job_group(*args, spark: SparkSession | None = None, **kwargs) -> None:
    """Label the spark job group

    Args:
        spark: spark session (optional)
        *args: args to pass to `setJobGroup`
        **kwargs: kwargs to pass to `setJobGroup`
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setJobGroup(*args, **kwargs)


def set_partitions(num_partitions: int, spark: SparkSession | None = None) -> None:
    logger.info(f"Updating to spark.sql.shuffle.partitions={num_partitions}")
    logger.info(f"Updating to spark.default.parallelism={num_partitions}")
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    spark.sql(f"SET spark.sql.shuffle.partitions={num_partitions}").collect()
    spark.sql(f"SET spark.default.parallelism={num_partitions}").collect()


def spark_checkpoint(sdf: DataFrame, spark: SparkSession | None = None) -> DataFrame:
    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    chk_dir = spark.sparkContext._jsc.sc().getCheckpointDir()
    if chk_dir.nonEmpty():
        return sdf.checkpoint(eager=True)

    logger.warning(
        "Spark checkpoint directory is empty, cannot do checkpointing; set it via spark.sparkContext.setCheckpointDir()"
    )
    return sdf


def add_uid_column(sdf, uid_col="uid"):
    """monotonically_increasing_id() is recalculated during transform and give different values for the same rows
    Therefore we need to save temporary the DataFrame with checkpointing for example.
    """
    if uid_col not in sdf.columns:
        logger.info(
            f"The unique-id column '{uid_col}' is not in your DataFrame. Adding it with monotonically_increasing_id() and trying to checkpoint."
        )
        sdf = sdf.withColumn(uid_col, F.monotonically_increasing_id())

        # double check that spark.sparkContext.setCheckpointDir has been used
        # we need to make uid column persistent (the value is non-deterministic if recalculated)
        sdf = spark_checkpoint(sdf)
    return sdf


def check_uid(sdf, uid_col):
    """Check if uid column is there and add it if missing"""
    if uid_col not in sdf.columns:
        sdf = add_uid_column(sdf, uid_col)
    else:
        # Column is there let's check if it is unique
        n_duplicate_id = sdf.groupby(uid_col).count().filter("count > 1").count()
        if n_duplicate_id > 0:
            msg = f"The unique-id column '{uid_col}' in is not a unique id in your DataFrame. There are {n_duplicate_id} duplicates."
            raise ValueError(msg)
    return sdf
