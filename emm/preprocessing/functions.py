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

from functools import partial
from typing import Any, Callable

import cleanco
from unidecode import unidecode

from emm.preprocessing.abbreviation_util import abbreviations_to_words, legal_abbreviations_to_words


def create_func_dict(use_spark: bool = True) -> dict[str, Callable[[Any], Any] | Callable[[str], str]]:
    if use_spark:
        import emm.preprocessing.spark_functions as F
    else:
        import emm.preprocessing.pandas_functions as F

    def map_shorthands(name):
        for pat, shorthand in [
            (r"ver(?:eniging)? v(?:an)? (\w*)(?:eigenaren|eigenaars)", r"vve \1"),
            (r"stichting", r"stg"),
            (r"straat", r"str"),
            (
                r"pub(?:lic)? lim(?:ited)? co(?:mpany)?|pub(?:lic)? l(?:td)? co(?:mpany)?|pub(?:lic)? co(?:mpany)? lim(?:ited)?|pub(?:lic)? co(?:mpany)? l(?:td)?|pcl",
                r"plc",
            ),
            (r"limited", r"ltd"),
        ]:
            name = F.regex_replace(pat, shorthand, simple=True)(name)
        return name

    return {
        # Replace accented characters by their normalized representation, e.g. replace 'ä' with 'A\xa4'
        "strip_accents_unicode": F.run_custom_function(unidecode),
        # Replace all dash and underscore characters with a space characters
        "strip_hyphens": F.regex_replace(r"""[-_]""", " ", simple=True),
        # Replace all punctuation characters (e.g. '.', '-', '_', ''', ';') with spaces
        # in Pyspark \p{Punct} includes + and | and $, Python regex does not include them, so they are added manually
        "strip_punctuation": F.regex_replace(r"""[\p{Punct}+|$=“”¨]""", " "),
        # Insert space around all punctuation characters, e.g., H&M => H & M; H.M. => H . M .
        "insert_space_around_punctuation": F.regex_replace(
            r"""([\p{Punct}+|$=“”])""", r" $1 " if use_spark else r" \1 "
        ),
        # Convert all upper-case characters to lower case and remove leading and trailing whitespace
        "handle_lower_trim": F.trim_lower,
        # Convert all upper-case characters to lower case and remove leading and trailing whitespace
        "handle_lower": F.lower,
        # Remove leading and trailing whitespace
        "handle_trim": F.trim,
        # Map all the abbreviations to the same format (Z. S. = Z.S. = ZS)
        "merge_abbreviations": F.run_custom_function(abbreviations_to_words),
        # Map all the legal form abbreviations to the same format (B. V.= B.V. = B V = BV)
        "merge_legal_form_abbreviations": F.run_custom_function(legal_abbreviations_to_words),
        # Map all the legal form abbreviations to the same format (B. V.= B.V. = B V = BV)
        "remove_extra_space": F.regex_replace(r"""\s+""", " ", simple=True),
        # Map all the shorthands to the same format (stichting => stg)
        "map_shorthands": map_shorthands,
        # Merge & separated abbreviations by removing & and the spaces between them
        "merge_&": F.regex_replace(
            r"(\s|^)(\w)\s*&\s*(\w)(\s|$)", r"$1$2$3$4" if use_spark else r"\1\2\3\4", simple=True
        ),
        # remove legal form
        "remove_legal_form": F.run_custom_function(
            partial(
                cleanco.clean.custom_basename,
                # Warning! the default set is incomplete and misses a lot of popular legal forms
                terms=cleanco.prepare_default_terms(),
                prefix=True,
                middle=True,
                suffix=True,
            )
        ),
        # removed any newlines in string (this is a sign of a dq problem!).
        "remove_newline": F.regex_replace(r"\n|\r", " "),
        # replace atypical dashes
        "replace_punctuation": F.regex_replace("[\u2013\u2014\u2015]", "-"),
    }


def replace_none(name: str | None) -> str:
    if name is None:
        return ""
    return name
