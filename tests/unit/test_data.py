import pytest

from emm.data.create_data import pandas_split_data
from emm.helper import spark_installed

if spark_installed:
    from emm.data.create_data import split_data


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_split_data(spark_session):
    ground_truth, negative_df = split_data(spark_session)

    assert negative_df.count() == 6800
    assert ground_truth.count() == 0


def test_pandas_split_data():
    ground_truth, negative_df = pandas_split_data()

    assert len(negative_df) == 6800
    assert len(ground_truth) == 0
