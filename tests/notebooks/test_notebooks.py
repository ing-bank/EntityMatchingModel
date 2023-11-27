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
from pathlib import Path

import pytest
from pytest_notebook.nb_regression import NBRegressionFixture

from emm.helper import spark_installed


@pytest.fixture(scope="module")
def root_directory():
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def nb_tester(root_directory):
    """Test notebooks using pytest-notebook"""
    exec_dir = root_directory / "notebooks"

    return NBRegressionFixture(
        diff_ignore=(
            "/metadata/language_info",
            "/cells/*/execution_count",
            "/cells/*/outputs/*",
        ),
        exec_timeout=1800,
        exec_cwd=str(exec_dir),
    )


def test_notebook_pandas(nb_tester, root_directory):
    file_path = (root_directory / "notebooks/01-entity-matching-pandas-version.ipynb").resolve()
    nb_tester.check(str(file_path))


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_notebook_spark(nb_tester, root_directory):
    file_path = (root_directory / "notebooks/02-entity-matching-spark-version.ipynb").resolve()
    nb_tester.check(str(file_path))


def test_notebook_fitter(nb_tester, root_directory):
    file_path = (root_directory / "notebooks/03-entity-matching-training-pandas-version.ipynb").resolve()
    nb_tester.check(str(file_path))


def test_notebook_aggregation(nb_tester, root_directory):
    file_path = (root_directory / "notebooks/04-entity-matching-aggregation-pandas-version.ipynb").resolve()
    nb_tester.check(str(file_path))
