"""Unit tests for `create_vocabulary`"""

import pandas as pd
import pytest

from emm.features.features_vocabulary import create_vocabulary


def test_create_vocabulary():
    data = pd.DataFrame(
        {
            "preprocessed": ["hello", "hello world", "world", "world"],
            "gt_preprocessed": ["world", "foobar", "world", "world"],
        }
    )
    vocab = create_vocabulary(
        data, columns=["preprocessed", "gt_preprocessed"], very_common_words_min_df=2, common_words_min_df=1
    )
    assert vocab.very_common_words == {"world", "hello"}
    assert vocab.common_words == {"foobar"}


def test_create_vocabulary_preprocessed_col():
    data = pd.DataFrame({"preprocessed": ["hello"], "gt_preprocessed": ["world"], "extra_col": ["foobar"]})
    vocab = create_vocabulary(
        data,
        columns=["preprocessed", "gt_preprocessed", "extra_col"],
        very_common_words_min_df=0.1,
        common_words_min_df=0.05,
    )
    assert vocab.very_common_words == {"hello", "world", "foobar"}
    assert vocab.common_words == set()


def test_create_vocabulary_exception():
    with pytest.raises(ValueError, match="`common_words_min_df` should be smaller than `very_common_words_min_df`"):
        _, _ = create_vocabulary(
            pd.DataFrame(), columns=["col1", "col2"], very_common_words_min_df=0.01, common_words_min_df=0.1
        )
