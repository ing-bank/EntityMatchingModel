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

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from emm.loggers import Timer
from emm.loggers.logger import logger


@dataclass
class Vocabulary:
    very_common_words: set[str]
    common_words: set[str]


def create_vocabulary(
    df: pd.DataFrame,
    columns: list[str],
    very_common_words_min_df: float | int = 0.01,
    common_words_min_df: float | int = 0.0001,
) -> Vocabulary:
    """Get two sets of 'common' and 'very common' words

    Args:
        df: data to obtain the vocabulary from
        columns: columns to compute the vocabulary from
        very_common_words_min_df: minimal document frequency to be considered 'very common'
        common_words_min_df: minimum document frequency to be considered 'common'

    Examples:
        >>> vocabulary = create_vocabulary(
        >>>    df,
        >>>    columns=["preprocessed", "gt_preprocessed"],
        >>>    very_common_words_min_df=0.05,
        >>>    common_words_min_df=0.005,
        >>> )
        >>> print(vocabulary.very_common_words)
        {"hello", "world"}
        >>> print(vocabulary.common_words)
        {"the", "a", "in"}

    Returns:
        Vocabulary with common and very common words
    """
    if common_words_min_df >= very_common_words_min_df:
        msg = "`common_words_min_df` should be smaller than `very_common_words_min_df`"
        raise ValueError(msg)

    # df contains one row per candidate; it should be generated via the same pipeline as when scoring, to ensure the same preprocessing is applied.
    logger.debug("Creating vocabulary")

    # very_common_words and common_words should be extracted after Preprocessor, here we already have those columns
    preprocessed_names = [df[col] for col in columns]

    logger.debug("Concat preprocessed")
    all_preprocessed = pd.concat(preprocessed_names, axis=0, ignore_index=True, sort=True).drop_duplicates().dropna()

    with Timer("Very common words") as timer:
        try:
            cv = CountVectorizer(min_df=very_common_words_min_df)
            cv.fit(all_preprocessed)
            very_common_words = set(cv.vocabulary_.keys())
        except ValueError:
            very_common_words = set()

        timer.log_param("n_very_common", len(very_common_words))

    with Timer("Very common words") as timer:
        try:
            cv = CountVectorizer(min_df=common_words_min_df)
            cv.fit(all_preprocessed)
            common_words = set(cv.vocabulary_.keys()) - very_common_words
        except ValueError:
            common_words = set()

        timer.log_param("n_common", len(common_words))

    return Vocabulary(very_common_words=very_common_words, common_words=common_words)


def compute_vocabulary_features(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    very_common_words: set[str] | None = None,
    common_words: set[str] | None = None,
) -> pd.DataFrame:
    """Features on tokens

    Args:
        df: input DataFrame
        col1: name to compare
        col2: other name to compare
        common_words: pre-computed common words
        very_common_words: pre-computed very common words

    Returns:
        DataFrame with features, e.g.:
        - hits: words in both names
        - misses: words that is just one name (on either side)

    """
    assert common_words is None or isinstance(common_words, set)
    assert very_common_words is None or isinstance(very_common_words, set)
    name1 = df[col1]
    name2 = df[col2]
    word_set1 = name1.str.findall(r"\w\w+").map(set)
    word_set2 = name2.str.findall(r"\w\w+").map(set)
    hits = pd.Series(word_set1.values & word_set2.values, index=word_set1.index)
    total_wrds = pd.Series(word_set1.values | word_set2.values, index=word_set1.index)
    misses = total_wrds - hits

    common_words = common_words or set()
    very_common_words = very_common_words or set()
    vocab = common_words | very_common_words

    very_common_hits = hits.apply(lambda x: sum(1 for y in x if y in very_common_words))
    common_hits = hits.apply(lambda x: sum(1 for y in x if y in common_words))
    no_hits = hits.apply(lambda x: sum(1 for y in x if y not in vocab))

    very_common_miss = misses.apply(lambda x: sum(1 for y in x if y in very_common_words))
    common_miss = misses.apply(lambda x: sum(1 for y in x if y in common_words))
    no_miss = misses.apply(lambda x: sum(1 for y in x if y not in vocab))

    n_hits = hits.map(len)
    n_total = total_wrds.map(len)
    n_set1 = word_set1.map(len)
    n_set2 = word_set2.map(len)
    ratio_overlap = (n_hits / n_total).replace(np.inf, 0)
    return pd.DataFrame(
        {
            "very_common_hit": very_common_hits,
            "common_hit": common_hits,
            "rare_hit": no_hits,
            "very_common_miss": very_common_miss,
            "common_miss": common_miss,
            "rare_miss": no_miss,
            "n_overlap_words": n_hits,
            "ratio_overlap_words": ratio_overlap,
            "num_word_difference": (n_set1 - n_set2).abs(),
        },
        dtype="float32",
    )
