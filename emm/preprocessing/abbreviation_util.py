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

import re

# NOT_FULL_UPPER: at least there lower case chars exist
NOT_FULL_UPPER = re.compile(r".*[a-z].*[a-z].*[a-z].*", re.UNICODE)
# ABBR_FINDER_UPPER: word with capital letters with a length of at least 2
ABBR_FINDER_UPPER = re.compile(r"([A-Z]{2,})", re.UNICODE)
# ABBR_FINDER_CAMEL: CamelCase abbreviations like PetroBras
ABBR_FINDER_CAMEL = re.compile(r"(?:[A-Z][a-z]+){2,}", re.UNICODE)
# ABBR_FINDER_PUNC: one character with a separator followed by one or more one-char words with the same separator
# the character before the abbreviation should be ^ or \s so that we don't split words accidentally
# the last group could miss dot, the regex does not capture the trailing space
ABBR_FINDER_PUNC = re.compile(
    r"(?:^|\s)("
    # without dots, i.e. A B C
    r"(?:(?:\w\s)+(?:\w(?=\s|$)))|"
    # with dots and spaces, i.e. A. B. C.
    r"(?:(?:\w\.\s)+(?:\w(?=\s|$)|\w\.))|"
    # with dots and no spaces, i.e. A.B.C.
    r"(?:(?:\w\.)+(?:\w(?=\s|$)|\w\.)))",
    re.UNICODE,
)
ABBR_FINDER_PUNC2 = re.compile(r"(?:^|\s)((?:\w(?:\.\s|$|\s|\.))+|(?:\w+(?:\.\s|$|\.))+)", re.UNICODE)

# RE_ABBR_SEPARATOR: abbreviation separators
RE_ABBR_SEPARATOR = re.compile(r"(\s|\.)", re.UNICODE)
RE_ABBR_SEPARATOR2 = re.compile(r"(\s|\.)+", re.UNICODE)


def find_abbr_merged_initials(name):
    """Finds abbreviations with merged initials
    examples: FC Barcelona => FC, ING BANK B.V. => BV
    """
    name += " "
    abbr = []
    if NOT_FULL_UPPER.match(name):
        abbr = ABBR_FINDER_UPPER.findall(name)
    all_abbreviations = list(ABBR_FINDER_PUNC.findall(name + " "))
    for abbreviation in all_abbreviations:
        abbr += [RE_ABBR_SEPARATOR.sub("", abbreviation)]
    return abbr


def find_abbr_merged_word_pieces(name):
    """Finds abbreviations with merged word pieces
    examples: PetroBras
    """
    return ABBR_FINDER_CAMEL.findall(name)


def extract_abbr_merged_initials(abbr, name):
    """Extract possible open form of the given abbreviation if exists
    examples: (SK, Fenerbahce Spor Klubu) => Spor Klubu
    """
    regex = r"\b"
    for char in abbr.lower():
        regex += char + r"\w+\s?"
    return re.search(regex, name.lower(), re.UNICODE)


def extract_abbr_merged_word_pieces(abbr, name):
    """Extract possible open form of the given abbreviation if exists
    examples: (PetroBras, Petroleo Brasileiro B.V.) => Petroleo Brasileiro
    """
    words = re.findall(r"[A-Z][a-z]+", abbr, re.UNICODE)
    regex = r""
    for word in words:
        regex += word.lower() + r"\w*\s?"
    return re.search(regex, name.lower(), re.UNICODE)


def abbreviations_to_words(name):
    """Maps all the abbreviations to the same format (B. V. = B.V. = B.V = B V = BV)"""
    name += " "
    all_abbreviations = list(ABBR_FINDER_PUNC.findall(name + " "))
    for abbreviation in all_abbreviations:
        new_form = RE_ABBR_SEPARATOR.sub("", abbreviation) + "<END_MARKER>"
        name = name.replace(abbreviation, new_form)
    # fix end markers
    name = re.sub("<END_MARKER> ?", " ", name)
    return name.strip()


def preprocess(name):
    if name is None:
        return ""
    return abbreviations_to_words(name).lower()


def legal_abbreviations_to_words(name):
    """Maps all the abbreviations to the same format (B. V.= B.V. = B V = BV)"""
    # a legal form list contains most important words
    legal_form_abbr_list = [
        "bv",
        "nv",
        "vof",  # netherlands
        "bvba",
        "vzw",
        "asbl",
        "vog",
        "snc",
        "scs",
        "sca",
        "sa",
        "sprl",
        "cvba",
        "scrl",  # Belgium
        "gmbh",
        "kgaa",
        "ag",
        "ohg",  # Germany
        "ska",
        "spzoo",  # Poland
        "plc",  # us
    ]
    all_abbreviations = ABBR_FINDER_PUNC2.findall(name)
    for abbreviation in all_abbreviations:
        new_form = RE_ABBR_SEPARATOR2.sub("", abbreviation)
        if new_form in legal_form_abbr_list:
            name = name.replace(abbreviation, new_form)
    return name


def abbr_match(str_with_abbr, str_with_open_form):
    """Checks if the second string has an open form of an abbreviation from the first string"""
    abbr_list = find_abbr_merged_initials(str_with_abbr)
    for abbr in abbr_list:
        if extract_abbr_merged_initials(abbr, str_with_open_form) is not None:
            return True
    abbr_list = find_abbr_merged_word_pieces(str_with_abbr)
    return any(extract_abbr_merged_word_pieces(abbr, str_with_open_form) is not None for abbr in abbr_list)
