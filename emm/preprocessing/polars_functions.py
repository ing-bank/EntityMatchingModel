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

from __future__ import annotations

import re
from functools import partial
from typing import Callable

import polars as pl


def run_custom_function(fn: Callable) -> Callable[[pl.Series], pl.Series]:
    return partial(map_elements, fn)


def map_elements(fn: Callable, data: pl.Series) -> pl.Series:
    return data.map_elements(fn, return_dtype=pl.String, skip_nulls=False)


def regex_replace(pat: str, repl: str, simple: bool = False) -> Callable[[pl.Series], pl.Series]:
    # Polars uses Rust regex syntax and $1-style capture references.
    pat = pat.replace(r"\p{Punct}", r"\p{P}")
    repl = re.sub(r"\\([1-9][0-9]*)", r"$\1", repl)
    return lambda data: data.str.replace_all(pat, repl)


def lower(data: pl.Series) -> pl.Series:
    return data.str.to_lowercase()


def trim(data: pl.Series) -> pl.Series:
    return data.str.strip_chars()


def trim_lower(data: pl.Series) -> pl.Series:
    return data.str.to_lowercase().str.strip_chars()
