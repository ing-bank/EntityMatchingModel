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

from emm.preprocessing import abbreviation_util as util


def test_find_abbr_initials():
    assert util.find_abbr_merged_initials("38th International Conference on Very Large Databases, Turkey 2012") == []
    assert util.find_abbr_merged_initials("VLDB 2012 Conf TR") == ["VLDB", "TR"]
    assert util.find_abbr_merged_initials("International V.L.D.B. Conference, 2013") == ["VLDB"]
    assert util.find_abbr_merged_initials("WarnerBros Entertainment") == []
    assert util.find_abbr_merged_initials("PetroBras B.V.") == ["BV"]
    assert util.find_abbr_merged_initials("Petroleo Brasileiro B.V.") == ["BV"]


def test_find_abbr_word_pieces():
    assert util.find_abbr_merged_word_pieces("38th International Conference on Very Large Databases, Turkey 2012") == []
    assert util.find_abbr_merged_word_pieces("VLDB 2012 Conf TR") == []
    assert util.find_abbr_merged_word_pieces("International V.L.D.B. Conference, 2013") == []
    assert util.find_abbr_merged_word_pieces("WarnerBros Entertainment") == ["WarnerBros"]
    assert util.find_abbr_merged_word_pieces("PetroBras B.V.") == ["PetroBras"]
    assert util.find_abbr_merged_word_pieces("Petroleo Brasileiro B.V.") == []


def test_extract_abbr_initials():
    assert (
        util.extract_abbr_merged_initials("VLDB", "38th International Conference on Very Large Databases, Turkey 2012")
        is not None
    )
    assert util.extract_abbr_merged_initials("VLDB", "Very Large People Meeting") is None
    assert util.extract_abbr_merged_initials("VLDB", "Verified Lames Database") is not None
    assert util.extract_abbr_merged_initials("AM", "Anmot Meder Investment") is not None


def test_extract_abbr_word_pieces():
    assert util.extract_abbr_merged_word_pieces("PetroBras", "Petroleo Brasileiro B.V.") is not None
    assert util.extract_abbr_merged_word_pieces("PetroBras", "Petrov Brothers") is None
    assert util.extract_abbr_merged_word_pieces("PetroBras", "Vladimir Petrov Bras B.V.") is not None
    assert util.extract_abbr_merged_word_pieces("TeknoPark", "Istanbul Teknoloji Parki") is not None


def test_abbreviations_to_words():
    assert util.abbreviations_to_words("Fenerbahce S. K.") == "Fenerbahce SK"
    assert util.abbreviations_to_words("Fenerbahce S.K.") == util.abbreviations_to_words("Fenerbahce S K")
    assert util.abbreviations_to_words("mcdonalds. j. lens") != "mcdonaldsj lens"  # NOT EQUAL!
    assert util.abbreviations_to_words("a.b.c. b.v.") == "abc bv"
    assert util.abbreviations_to_words("a b cde") == "ab cde"
    assert util.abbreviations_to_words("a. b. van den xyz b.v.") == "ab van den xyz bv"
    # edge case no space at the end of the group
    assert util.abbreviations_to_words("a.b.c.def") == "abc def"
    assert util.abbreviations_to_words("a.b.c. def") == "abc def"
    # multiple groups
    assert util.abbreviations_to_words("a b c.d.") == "ab cd"
    # cases with missing dot at the end of the group
    assert util.abbreviations_to_words("abc b.v") == "abc bv"
    assert util.abbreviations_to_words("abc b.b.v") == "abc bbv"
    assert util.abbreviations_to_words("abc b.b v.x") == "abc bb vx"
    assert util.abbreviations_to_words("abc b. b. v") == "abc bbv"
    assert util.abbreviations_to_words("abc b.v x") == "abc bv x"


def test_abbr_to_words_only_legal_form():
    # change because legal form
    assert util.legal_abbreviations_to_words("tzu sun b.v.") == "tzu sun bv"
    assert util.legal_abbreviations_to_words("Eddie Arnheim g.m.b.h.") == "Eddie Arnheim gmbh"
    assert util.legal_abbreviations_to_words("Kris sp. zoo.") == "Kris spzoo"

    # not change
    assert util.legal_abbreviations_to_words("z. s. chinese company") == "z. s. chinese company"


def test_abbr_match():
    assert (
        util.abbr_match("38th International Conference on Very Large Databases, Turkey 2012", "VLDB 2012 Conf TR")
        is False
    )
    assert (
        util.abbr_match("VLDB 2012 Conf TR", "38th International Conference on Very Large Databases, Turkey 2012")
        is True
    )
    assert util.abbr_match("PetroBras B.V.", "Petroleo Brasileiro B.V.") is True
    assert util.abbr_match("WarnerBros Entertainment", "Petroleo Brasileiro B.V.") is False
