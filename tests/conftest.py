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

from emm.data.create_data import create_training_data, retrieve_kvk_test_sample
from emm.helper import spark_installed
from emm.helper.io import save_file
from emm.helper.util import string_columns_to_pyarrow
from emm.supervised_model.base_supervised_model import train_test_model

if spark_installed:
    from pyspark import SparkConf
    from pyspark.sql import SparkSession


APP_NAME = "pytest-pyspark-namematching-tests"


def pytest_configure(config):
    # by default disable benchmarking tests, it can be re-enabled using --benchmark-enable option
    if hasattr(config.option, "benchmark_enable") and not config.option.benchmark_enable:
        config.option.benchmark_skip = True


@pytest.fixture(scope="session")
def supervised_model(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("models")

    df, vocabulary = create_training_data()
    # overriding n_folds value due to small dataset size
    sem, dataset_scored = train_test_model(df, vocabulary, name_only=False, n_folds=4)
    save_file(str(tmp_path / "sem.pkl"), sem)
    dataset_scored.to_csv(tmp_path / "sem.csv")
    sem_nm, dataset_scored_nm = train_test_model(df, vocabulary, name_only=True, n_folds=4)
    save_file(str(tmp_path / "sem_nm.pkl"), sem_nm)
    dataset_scored_nm.to_csv(tmp_path / "sem_nm.csv")
    sem_nm_without_rank, dataset_scored_nm_without_rank = train_test_model(
        df, vocabulary, name_only=True, without_rank_features=True, n_folds=4
    )
    save_file(str(tmp_path / "sem_nm_without_rank.pkl"), sem_nm_without_rank)
    dataset_scored_nm_without_rank.to_csv(tmp_path / "sem_nm_without_rank.csv")
    return (
        tmp_path / "sem.pkl",
        tmp_path / "sem.csv",
        tmp_path / "sem_nm.pkl",
        tmp_path / "sem_nm.csv",
        tmp_path / "sem_nm_without_rank.pkl",
        tmp_path / "sem_nm_without_rank.csv",
    )


@pytest.fixture(scope="session")
def kvk_dataset():
    # read_csv with engine='pyarrow' not working (pyarrow 11.0.0)
    _, df = retrieve_kvk_test_sample()
    df = df.rename(columns={"Name": "name", "Index": "id"})
    df["id"] *= 10
    # converting string columns here instead
    return string_columns_to_pyarrow(df)


@pytest.fixture(scope="session")
def kvk_training_dataset():
    # read_csv with engine='pyarrow' not working (pyarrow 11.0.0)
    _, df = retrieve_kvk_test_sample()
    df = df.rename(columns={"Name": "name", "Index": "id"})
    df = df.sort_values(by=["name"])
    df["id"] = [i // 2 for i in range(len(df))]
    # converting string columns here instead
    return string_columns_to_pyarrow(df)


@pytest.fixture(scope="session")
def spark_session(tmp_path_factory):
    """Pytest fixture for get or creating the spark_session
    Creating a fixture enables it to reuse the spark contexts across all tests.
    """
    if not spark_installed:
        return None

    conf = {
        "spark.driver.maxResultSize": "1G",
        "spark.driver.memoryOverhead": "1G",
        "spark.executor.cores": "1",
        "spark.executor.memoryOverhead": "1G",
        "spark.python.worker.memory": "2G",
        "spark.driver.memory": "4G",
        "spark.executor.memory": "4G",
        # In Spark 3.2 it is enabled by default, very important to disable to keep full control over the partitions and their consistency:
        "spark.sql.adaptive.enabled": "false",
        "spark.ui.enabled": "false",
    }
    conf = list(conf.items())
    config = SparkConf().setAll(conf)

    spark_session = SparkSession.builder.appName("EMM Test").config(conf=config)
    spark = spark_session.getOrCreate()

    checkpoint_path = tmp_path_factory.mktemp("checkpoints")
    spark.sparkContext.setCheckpointDir(str(checkpoint_path))

    yield spark
    spark.stop()


# Global setting to display all the pandas dataframe for debugging
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 40)
