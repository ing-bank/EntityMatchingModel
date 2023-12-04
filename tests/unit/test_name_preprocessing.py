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

import inspect
import os

import pandas as pd
import pytest

from emm.helper import spark_installed
from emm.pipeline import PandasEntityMatching
from emm.preprocessing.pandas_preprocessor import PandasPreprocessor

if spark_installed:
    from emm.pipeline import SparkEntityMatching
    from emm.preprocessing.spark_preprocessor import SparkPreprocessor


THIS_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.mark.skipif(not spark_installed, reason="spark not found")
@pytest.mark.parametrize(
    ("processor_name", "test_l", "expect_l"),
    [
        ("strip_hyphens", ["Tzu-Sun_BV.a;b,c_ä"], ["Tzu Sun BV.a;b,c ä"]),
        ("strip_punctuation", ["Tzu-Sun_BV:Chinese'Dutch.a;b,c_ä"], ["Tzu Sun BV Chinese Dutch a b c ä"]),
        (
            "insert_space_around_punctuation",
            ["Tzu-Sun_BV:Chinese'Dutch.a;b,c_ä"],
            ["Tzu - Sun _ BV : Chinese ' Dutch . a ; b , c _ ä"],
        ),
        ("handle_lower_trim", ["Tzu-Sun_BV.a;b,c_ä"], ["tzu-sun_bv.a;b,c_ä"]),
        (
            "strip_accents_unicode",
            ["Tzu-Sun_BV.a;b,c_ä", "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "Café"],
            ["Tzu-Sun_BV.a;b,c_a", "acelnoszzACELNOSZZ", "Cafe"],
        ),
        ("merge_&", ["xyz & abc C&D"], ["xyz & abc CD"]),
        (
            "preprocess_name",
            ["Tzu-Sun_BV.a;b,c_ä", "Tzu-Sun_BV  morethan1space"],
            ["tzu sun bv a b c a", "tzu sun bv  morethan1space"],
        ),
        ("preprocess_with_punctuation", ["Tzu-Sun_BV.a;b,c_ä"], ["tzu - sun _ bv . a ; b , c _ a"]),
        (
            "preprocess_merge_abbr",
            ["Tzu-Sun_B.V.a;b,c_ä", "Z. S. B. V.", "Z Sun B V", "Z. Sun B.V.", "Z Sun B.V"],
            ["tzu sun b v a b c a", "zsbv", "z sun bv", "z  sun bv", "z sun bv"],
        ),
        (
            "preprocess_merge_legal_abbr",
            [
                "Tzu-Sun B. V.",
                "Tzu-Sun B.V",
                "Tzu-Sun B V",
                "Tzu-Sun BV.",
                "J. Arnheim. N.V.",
                "J.A. N. V.",  # does not work for this one!
                "J.A. vof",
                "cris adamsky s.p.z.o.o.",
            ],
            [
                "tzu sun bv",
                "tzu sun bv",
                "tzu sun bv",
                "tzu sun bv",
                "j arnheim nv",
                "j a n v",
                "j a vof",
                "cris adamsky spzoo",
            ],
        ),
        (
            "remove_legal_form",
            ["Tzu-Sun Ltd", "Tzu-Sun GMBH", "Ltd Tzu-Sun", "Tzu Ltd Sun", "Tzu-Sun sp. z o.o.", "Tzu-Sun sp. z.o.o."],
            ["Tzu-Sun", "Tzu-Sun", "Tzu-Sun", "Tzu Sun", "Tzu-Sun", "Tzu-Sun"],
        ),
    ],
)
def test_preprocessor(spark_session, processor_name, test_l, expect_l):
    df_before = spark_session.createDataFrame(enumerate(test_l), ["id", "name"])
    if not processor_name.startswith("preprocess"):
        processor_name = [processor_name]
    spark_preprocessor = SparkPreprocessor(processor_name, input_col="name", output_col="name")
    pandas_preprocessor = PandasPreprocessor(processor_name, input_col="name", output_col="name")
    spark_name_after = spark_preprocessor._transform(df_before).select("name").toPandas()["name"].tolist()
    pandas_name_after = pandas_preprocessor.transform(df_before.toPandas())["name"].tolist()
    assert spark_name_after == expect_l
    assert pandas_name_after == expect_l


def add_extra(x):
    return f"{x} EXTRA"


def test_custom_function_in_pandas_preprocessor():
    pandas_preprocessor = PandasPreprocessor([add_extra], input_col="name", output_col="name")
    df = pd.DataFrame({"name": ["name1", "name2", "name3"]})
    res = pandas_preprocessor.transform(df)["name"].tolist()
    assert res == ["name1 EXTRA", "name2 EXTRA", "name3 EXTRA"]


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_custom_function_in_spark_preprocessor(spark_session):
    spark_preprocessor = SparkPreprocessor([add_extra], input_col="name", output_col="name")
    df = spark_session.createDataFrame(enumerate(["name1", "name2", "name3"]), ["id", "name"])
    res = spark_preprocessor._transform(df).select("name").toPandas()["name"].tolist()
    assert res == ["name1 EXTRA", "name2 EXTRA", "name3 EXTRA"]


@pytest.fixture()
def sample_gt():
    return pd.DataFrame({"id": [1, 2], "name": ["Some company! ltd", "OthEr s.a."]})


def test_preprocessor_object_pandas_in_em(sample_gt):
    pandas_preprocessor = PandasPreprocessor("preprocess_name")

    em_params = {
        "preprocessor": pandas_preprocessor,
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [{"type": "sni", "window_length": 3}],
        "supervised_on": False,
    }

    p = PandasEntityMatching(em_params)
    p = p.fit(sample_gt)
    candidates = p.transform(sample_gt)
    assert len(candidates) > 0
    assert "preprocessed" in candidates.columns
    assert "gt_preprocessed" in candidates.columns


@pytest.mark.skipif(not spark_installed, reason="spark not found")
def test_preprocessor_object_spark_in_em(spark_session, sample_gt):
    sample_gt_sdf = spark_session.createDataFrame(sample_gt)

    preprocessor = SparkPreprocessor("preprocess_name")

    em_params = {
        "preprocessor": preprocessor,
        "name_only": True,
        "entity_id_col": "id",
        "name_col": "name",
        "indexers": [{"type": "sni", "window_length": 3}],
        "supervised_on": False,
    }

    p = SparkEntityMatching(em_params)
    p = p.fit(sample_gt_sdf)
    candidates = p.transform(sample_gt_sdf)
    assert candidates.count() > 0
    assert "preprocessed" in candidates.columns
    assert "gt_preprocessed" in candidates.columns


def test_preprocessor_pandas_unusual_chars():
    pandas_preprocessor = PandasPreprocessor("preprocess_name")

    # test for $=“”\n
    df = pd.DataFrame(
        {
            "name": [
                "B=N=Consult B.V.",
                "Stichting Vrienden van Laurens “Pax Intrantibus”",
                "Nicren$ N.V.",
                "Scheepvaartbedrijf Absurdia \nInc",
                "æøå ÆØÅ inc",
                "ẞ ß german co",
            ]
        }
    )

    out = pandas_preprocessor.transform(df)["preprocessed"].tolist()
    expect = [
        "b n consult b v",
        "stichting vrienden van laurens  pax intrantibus",
        "nicren  n v",
        "scheepvaartbedrijf absurdia  inc",
        "aeoa aeoa inc",
        "ss ss german co",
    ]

    assert out == expect
