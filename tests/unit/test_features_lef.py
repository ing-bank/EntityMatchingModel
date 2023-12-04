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

from emm.features.features_lef import extract_lef, get_business_type, make_combi, matching_legal_terms


def test_extract_lef():
    business_name1 = "Some Big Pharma B.V."
    business_name2 = "Some Big Pharma flobble."
    business_name3 = "Some Big Pharma NV"

    lef1 = extract_lef(business_name1)
    lef2 = extract_lef(business_name2)
    lef3 = extract_lef(business_name3)

    assert lef1 == "bv"
    assert lef2 == ""
    assert lef3 == "nv"


def test_get_business_type():
    lef1 = "bv"
    lef2 = ""
    lef3 = "nv"
    lef4 = "fdjafdja;fjdkls"

    bt1 = get_business_type(lef1)
    bt2 = get_business_type(lef2)
    bt3 = get_business_type(lef3)
    bt4 = get_business_type(lef4)

    assert bt1 == "Limited"
    assert bt2 == "no_lef"
    assert bt3 == "Corporation:Limited Liability Company"
    assert bt4 == "unknown_lef"


def test_combi():
    lef1 = "bv"
    lef2 = ""

    combi = make_combi(lef1, lef2)
    assert combi == "bv__no_lef"


def test_matching_legal_entity_forms():
    lef1 = "bv"
    lef2 = ""
    lef3 = "nv"
    lef4 = "fdjafdjafjdkls:bv"

    assert matching_legal_terms(lef1, lef1) == "identical"
    assert matching_legal_terms(lef1, lef2) == "lef2_missing"
    assert matching_legal_terms(lef1, lef3) == "no_match"
    assert matching_legal_terms(lef1, lef4) == "partial_match"


def test_matching_business_types():
    bt1 = "Limited"
    bt2 = "no_lef"
    bt3 = "Corporation:Limited Liability Company"
    bt4 = "unknown_lef:Limited"

    assert matching_legal_terms(bt1, bt1) == "identical"
    assert matching_legal_terms(bt1, bt2) == "lef2_missing"
    assert matching_legal_terms(bt1, bt3) == "no_match"
    assert matching_legal_terms(bt1, bt4) == "partial_match"
