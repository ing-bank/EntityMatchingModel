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

from functools import partial
from typing import Callable

from emm.helper import spark_installed

if spark_installed:
    from pyspark.sql import Column
    from pyspark.sql import functions as sf
    from pyspark.sql.types import StringType


def run_custom_function(fn: Callable) -> Callable[[str], Column]:
    return sf.udf(fn, StringType())


def regex_replace(pat: str, repl: str, simple: bool = False) -> Callable[[str], Column]:
    return partial(sf.regexp_replace, pattern=pat, replacement=repl)


def lower(x: str) -> Column:
    return sf.lower(sf.col(x))


def trim(x: str) -> Column:
    return sf.trim(sf.col(x))


def trim_lower(x: str) -> Column:
    return sf.trim(sf.lower(sf.col(x)))
