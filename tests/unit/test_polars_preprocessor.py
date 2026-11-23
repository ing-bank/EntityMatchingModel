# Copyright (c) 2023 ING Analytics Wholesale Banking
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from importlib import import_module

import pytest

pl = pytest.importorskip("polars")

PolarsPreprocessor = import_module("emm.preprocessing.polars_preprocessor").PolarsPreprocessor


@pytest.mark.parametrize(
    ("preprocess_pipeline", "names", "expected"),
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
        ("merge_&", ["xyz & abc C&D", "foobar S/A"], ["xyz & abc CD", "foobar SA"]),
        (
            "preprocess_name",
            ["Tzu-Sun_BV.a;b,c_ä", "Tzu-Sun_BV  morethan1space", None],
            ["tzu sun bv a b c a", "tzu sun bv morethan1space", ""],
        ),
        ("preprocess_with_punctuation", ["Tzu-Sun_BV.a;b,c_ä"], ["tzu - sun _ bv . a ; b , c _ a"]),
        (
            "preprocess_merge_abbr",
            ["Tzu-Sun_B.V.a;b,c_ä", "Z. S. B. V.", "Z Sun B V", "Z. Sun B.V.", "Z Sun B.V"],
            ["tzu sun b v a b c a", "zsbv", "z sun bv", "z sun bv", "z sun bv"],
        ),
        (
            "preprocess_merge_legal_abbr",
            [
                "Tzu-Sun B. V.",
                "Tzu-Sun B.V",
                "Tzu-Sun B V",
                "Tzu-Sun BV.",
                "J. Arnheim. N.V.",
                "J.A. N. V.",
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
def test_polars_preprocessor(preprocess_pipeline, names, expected):
    pipeline = preprocess_pipeline if preprocess_pipeline.startswith("preprocess") else [preprocess_pipeline]
    dataset = pl.DataFrame({"name": names})

    result = PolarsPreprocessor(pipeline, output_col="name").transform(dataset)

    assert result["name"].to_list() == expected
    assert dataset.columns == ["name"]
    assert dataset["name"].to_list() == names


def test_polars_preprocessor_supports_custom_functions_and_series():
    def add_extra(name):
        return f"{name} EXTRA"

    result = PolarsPreprocessor([add_extra], input_col="name", output_col="name").transform(
        pl.Series("name", ["name1", "name2"])
    )

    assert result["name"].to_list() == ["name1 EXTRA", "name2 EXTRA"]


def test_polars_preprocessor_unusual_chars():
    dataset = pl.DataFrame(
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

    result = PolarsPreprocessor("preprocess_name").transform(dataset)

    assert result["preprocessed"].to_list() == [
        "b n consult b v",
        "stichting vrienden van laurens pax intrantibus",
        "nicren n v",
        "scheepvaartbedrijf absurdia inc",
        "aeoa aeoa inc",
        "ss ss german co",
    ]
