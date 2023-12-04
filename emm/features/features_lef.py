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

from collections import defaultdict

import cleanco
import numpy as np
import pandas as pd
from cleanco.clean import normalize_terms, normalized, strip_punct, strip_tail
from cleanco.termdata import terms_by_type

LEGAL_TERMS = cleanco.clean.prepare_default_terms()
NO_LEF = "no_lef"
UNKNOWN_LEF = "unknown_lef"


def types_by_lef_dict(lefs_by_type=terms_by_type):
    """Business types by legal entity form

    Invert cleanco's dictionary `terms_by_type`.

    Args:
        lefs_by_type: cleanco's terms_by_type dictionary.

    Returns:
        types_by_lef dict
    """
    # convert to normalized terms, as used in custom_basename(). keep unique terms only.
    norm_tbt = {key: sorted(set(normalize_terms(lefs_by_type[key]))) for key in lefs_by_type}
    # inverse mapping -> types by normalized legal entity form
    types_by_lef = defaultdict(list)
    for business_type, lefs in norm_tbt.items():
        for lef in lefs:
            types_by_lef[lef].append(business_type)
    # add dummy empty-lef type
    types_by_lef[""].append(NO_LEF)
    return types_by_lef


TYPES_BY_LEF = types_by_lef_dict()


def custom_basename_and_lef(
    name: str,
    terms=LEGAL_TERMS,
    suffix: bool = True,
    prefix: bool = False,
    middle: bool = False,
    return_lef: bool = False,
):
    """Return cleaned base version of the business name and legal entity form

    Same as cleanco.clean.custom_basename(), but also return legal entity form(s).

    Args:
        name: business name to clean
        terms: legal entity forms to search for.
        suffix: remove legal entity forms from suffix of name. default is True.
        prefix: remove legal entity forms from prefix of name. default is False.
        middle: remove legal entity forms from middle of name. default is False.
        return_lef: default is False.

    Returns:
        basename and list with list with legal entity forms
    """
    name = strip_tail(name)
    nparts = name.split()
    nname = normalized(name)
    nnparts = list(map(strip_punct, nname.split()))
    nnsize = len(nnparts)

    if return_lef:
        suffix_lef = []
        prefix_lef = []
        middle_lef = []

    if suffix or prefix or middle:
        for termsize, termparts in terms:
            if suffix and nnparts[-termsize:] == termparts:
                del nnparts[-termsize:]
                del nparts[-termsize:]
                if return_lef:
                    suffix_lef.append(" ".join(termparts))
            if prefix and nnparts[:termsize] == termparts:
                del nnparts[:termsize]
                del nparts[:termsize]
                if return_lef:
                    prefix_lef.append(" ".join(termparts))
            if middle and termsize > 1:
                sizediff = nnsize - termsize
                if sizediff > 1:
                    for i in range(nnsize - termsize + 1):
                        if termparts == nnparts[i : i + termsize]:
                            del nnparts[i : i + termsize]
                            del nparts[i : i + termsize]
                            if return_lef:
                                middle_lef.append(" ".join(termparts))
            elif middle and termsize <= 1 and termparts[0] in nnparts[1:-1]:
                idx = nnparts[1:-1].index(termparts[0])
                del nnparts[idx + 1]
                del nparts[idx + 1]
                if return_lef:
                    middle_lef.append(" ".join(termparts))

    base = strip_tail(" ".join(nparts))
    if return_lef:
        lef = prefix_lef + middle_lef + suffix_lef[::-1]
        return base, lef
    return base


def extract_lef(name, terms=LEGAL_TERMS, suffix=True, prefix=False, middle=False, return_lef=True):
    """Extract legal entity form(s) from business name.

    Same as `custom_basename_and_lef()`, but returns no basename.

    Args:
        name: business name to clean
        terms: legal entity forms to search for.
        suffix: remove legal entity forms from suffix of name. default is True.
        prefix: remove legal entity forms from prefix of name. default is False.
        middle: remove legal entity forms from middle of name. default is False.
        return_lef: default is True.

    Returns:
        joined string of legal entity forms found
    """
    _, lef = custom_basename_and_lef(
        name, terms=terms, suffix=suffix, prefix=prefix, middle=middle, return_lef=return_lef
    )
    return ":".join(lef)


def get_business_type(joined_lef: str, types_by_lef=TYPES_BY_LEF):
    """Derive general business type from legal entity form

    Args:
        joined_lef: joined string of legal entity forms, from `extract_lef()`.
        types_by_lef: default is TYPES_BY_LEF classification from cleanco.

    Returns:
        joined string of general business types found.
    """
    lefs = joined_lef.split(":")
    entity_types = np.concatenate([types_by_lef.get(lef, [UNKNOWN_LEF]) for lef in lefs])
    # keep unique types only.
    indices = np.unique(entity_types, return_index=True)[1]
    entity_types = np.array([entity_types[index] for index in sorted(indices)])
    return ":".join(entity_types)


def matching_legal_terms(term1: str, term2: str):
    """Do two legal entity forms match

    Args:
        term1: legal entity form 1
        term2: legal entity form 2

    Returns:
        matching string.
    """
    if term1 in {NO_LEF, ""} and term2 in {NO_LEF, ""}:
        return "lef1_lef2_missing"
    if term1 in {NO_LEF, ""}:
        return "lef1_missing"
    if term2 in {NO_LEF, ""}:
        return "lef2_missing"
    if term1 == UNKNOWN_LEF and term2 == UNKNOWN_LEF:
        return "lef1_lef2_unknown"
    if term1 == UNKNOWN_LEF:
        return "lef1_unknown"
    if term2 == UNKNOWN_LEF:
        return "lef2_unknown"
    if term1 == term2:
        return "identical"

    bts1 = sorted(term1.split(":"))
    bts2 = sorted(term2.split(":"))

    if bts1 == bts2:
        return "identical"

    overlap = not set(bts1).isdisjoint(bts2)
    return "partial_match" if overlap else "no_match"


def make_combi(joined1: str, joined2: str):
    """Make combined string utility function"""
    if joined1 == "":
        joined1 = NO_LEF
    if joined2 == "":
        joined2 = NO_LEF
    return f"{joined1}__{joined2}"


def calc_lef_features(
    df: pd.DataFrame,
    name1: str = "preprocessed",
    name2: str = "gt_preprocessed",
    business_type: bool = False,
    detailed_match: bool = False,
) -> pd.DataFrame:
    """Determine legal entity form-based features of both names using cleanco

    Args:
        df: candidates dataframe.
        name1: column of name1, default is "preprocessed".
        name2: column of name1, default is "gt_preprocessed".
        business_type: if True, determine match of general international business type (from LEF).
        detailed_match: if True, store both legal entity forms (and possibly business types).
        n_jobs: desired number of parallel jobs. default is 1.

    Returns:
        dataframe with match of legal entity forms.
    """
    for name in [name1, name2]:
        if name not in df.columns:
            msg = f"column {name} not in dataframe"
            raise ValueError(msg)

    tmp = pd.DataFrame(index=df.index)
    res = pd.DataFrame(index=df.index)

    # legal entity forms
    tmp["lef1"] = df[name1].apply(extract_lef)
    tmp["lef2"] = df[name2].apply(extract_lef)
    # determine match
    res["match_legal_entity_form"] = tmp.apply(lambda x: matching_legal_terms(x["lef1"], x["lef2"]), axis=1).astype(
        "category"
    )

    # general (international) business type
    if business_type:
        # extract general international business type from LEF
        tmp["bt1"] = tmp["lef1"].apply(get_business_type)
        tmp["bt2"] = tmp["lef2"].apply(get_business_type)
        # determine match
        res["match_business_type"] = tmp.apply(lambda x: matching_legal_terms(x["bt1"], x["bt2"]), axis=1).astype(
            "category"
        )

    if detailed_match:
        res["legal_entity_forms"] = tmp.apply(lambda x: make_combi(x["lef1"], x["lef2"]), axis=1).astype("category")

        if business_type:
            res["business_types"] = tmp.apply(lambda x: make_combi(x["bt1"], x["bt2"]), axis=1).astype("category")

    return res
