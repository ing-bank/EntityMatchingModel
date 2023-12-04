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

import re
from typing import Callable, Match

import numpy as np
import pandas as pd

# NOT_FULL_UPPER: at least there lower case chars exist
NOT_FULL_UPPER = re.compile(r".*[a-z].*[a-z].*[a-z].*", re.UNICODE)
# ABBR_FINDER_UPPER: word with capital letters with a length of at least 2
ABBR_FINDER_UPPER = re.compile(r"([A-Z]{2,})", re.UNICODE)
# ABBR_FINDER_CAMEL: CamelCase abbreviations like PetroBras
ABBR_FINDER_CAMEL = re.compile(r"(?:[A-Z][a-z]+){2,}", re.UNICODE)
# ABBR_FINDER_PUNC: one character with a separator followed by one or more one-char words with the same separator
# the character before the abbreviation should be ^ or \s so that we don't split words accidentally
ABBR_FINDER_PUNC = re.compile(r"(?:^|\s)((?:\w(\.\s|\s|\.))(?:\w\2)+)", re.UNICODE)
# RE_ABBR_SEPARATOR: abbreviation separators
RE_ABBR_SEPARATOR = re.compile(r"(\s|\.)", re.UNICODE)
# WORD SPLIT
WORD_SPLIT = re.compile(r"\W+", re.UNICODE)
# WORDS ABBR
WORDS_ABBR = re.compile(r"[A-Z][a-z]+", re.UNICODE)


def find_abbr_merged_initials(name: str) -> list[str]:
    """Finds abbreviations with merged initials
    examples: FC Barcelona => FC, ING BANK B.V. => BV
    """
    name += " "
    abbr = []
    if NOT_FULL_UPPER.match(name):
        abbr = ABBR_FINDER_UPPER.findall(name)
    all_abbreviations = [x[0] for x in ABBR_FINDER_PUNC.findall(name + " ")]
    for abbreviation in all_abbreviations:
        abbr += [RE_ABBR_SEPARATOR.sub("", abbreviation)]
    return abbr


def find_abbr_merged_word_pieces(name: str) -> list[str]:
    """Finds abbreviations with merged word pieces
    examples: PetroBras
    """
    return ABBR_FINDER_CAMEL.findall(name)


def extract_abbr_merged_initials(abbr: str, name: str) -> Match | None:
    """Extract possible open form of the given abbreviation if exists
    examples: (SK, Fenerbahce Spor Klubu) => Spor Klubu
    """
    regex = r"\b"
    for char in abbr.lower():
        regex += char + r"\w+\s?"
    return re.search(regex, name.lower(), re.UNICODE)


def extract_abbr_merged_word_pieces(abbr: str, name: str) -> Match | None:
    """Extract possible open form of the given abbreviation if exists
    examples: (PetroBras, Petroleo Brasileiro B.V.) => Petroleo Brasileiro
    """
    words = WORDS_ABBR.findall(abbr)
    regex = r""
    for word in words:
        regex += word.lower() + r"\w*\s?"
    return re.search(regex, name.lower(), re.UNICODE)


def original_abbr_match(str_with_abbr: str, str_with_open_form: str) -> bool:
    """Checks if the second string has an open form of an abbreviation from the first string"""
    abbr_list = find_abbr_merged_initials(str_with_abbr)
    for abbr in abbr_list:
        if extract_abbr_merged_initials(abbr, str_with_open_form) is not None:
            return True
    abbr_list = find_abbr_merged_word_pieces(str_with_abbr)
    return any(extract_abbr_merged_word_pieces(abbr, str_with_open_form) is not None for abbr in abbr_list)


def abbr_match(str_with_abbr: str, str_with_open_form: str) -> bool:
    """If `str_with_abbr` contains both upper & lower case characters, we use original method,
    otherwise we apply approximate check:
    all short words (with length from range 2..5) are tested for abbreviation.
    """
    if any(c.islower() for c in str_with_abbr) and any(c.isupper() for c in str_with_abbr):
        return original_abbr_match(str_with_abbr, str_with_open_form)

    # extract all words from str_with_abbr
    # if token is short try if it used as abbrv.
    return any(
        2 <= len(token) <= 5 and extract_abbr_merged_initials(token, str_with_open_form) is not None
        for token in WORD_SPLIT.split(str_with_abbr)
    )


def abs_len_diff(name1: str, name2: str) -> int:
    """Difference (in characters) in lengths of names"""
    return abs(len(name1) - len(name2))


def len_ratio(name1: str, name2: str) -> float:
    """Calculates the lengths' ratio (1 means the same lengths, 0.5 one name is two times longer)"""
    len_n1 = len(name1)
    len_n2 = len(name2)
    max_len = max(len_n1, len_n2)
    if max_len > 0:
        return float(min(len_n1, len_n2)) / max_len
    return 1.0


def name_cut(name1: str, name2: str) -> bool:
    """Tests if one name is a prefix of other"""
    return name1.startswith(name2) or name2.startswith(name1)


def calc_name_features(
    df: pd.DataFrame, funcs: dict[Callable, str], name1: str = "preprocessed", name2: str = "gt_preprocessed"
) -> pd.DataFrame:
    res = pd.DataFrame(index=df.index)

    df = df[[name1, name2]].fillna("")
    for column, (func, dtype) in funcs.items():
        res[column] = np.vectorize(func)(df[name1], df[name2]).astype(dtype) if len(df) != 0 else None
    return res
