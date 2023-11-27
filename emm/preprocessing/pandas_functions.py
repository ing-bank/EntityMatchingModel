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

from typing import Callable

import pandas as pd
from regex import regex


def run_custom_function(fn) -> Callable[[pd.Series], pd.Series]:
    return lambda data: data.apply(fn)


# we need regex instead of re due to support of unicode groups (i.e. \p{Punct})
def regex_replace(pat: str, repl: str, simple: bool = False) -> Callable[[pd.Series], pd.Series]:
    if simple:
        return lambda data: data.str.replace(pat, repl, regex=True)
    return lambda data: data.apply(lambda value: regex.sub(pat, repl, value))


def lower(x: pd.Series) -> pd.Series:
    return x.str.lower()


def trim(x: pd.Series) -> pd.Series:
    return x.str.strip()


def trim_lower(x: pd.Series) -> pd.Series:
    return x.str.lower().str.strip()
