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

from emm.preprocessing.functions import create_func_dict


def test_map_shorthands():
    map_shorthands = create_func_dict(use_spark=False)["map_shorthands"]

    data = pd.Series(
        [
            "stichting het museum",
            "willem barentszstraat",
            "vereniging van eigenaren gebouw",
            "ver v appartementseigenaars gebouw",
            "public limited co",
            "public lim co",
            "public ltd co",
            "public co ltd",
        ]
    )
    result = map_shorthands(data)
    expected = pd.Series(
        ["stg het museum", "willem barentszstr", "vve  gebouw", "vve appartements gebouw", "plc", "plc", "plc", "plc"]
    )
    assert (result == expected).all()
