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

"""This file provides several helper function for name preprocessing

As a user, you could use preprocess_name directly
"""

from __future__ import annotations

from typing import Any

from emm.base.module import Module
from emm.preprocessing.functions import create_func_dict

DEFINED_PIPELINE_DICT = {
    "preprocess_name": [
        "strip_accents_unicode",
        "replace_punctuation",
        "remove_newline",
        "strip_punctuation",  # normal way: remove punctuation, handle unicode, lower and trim
        "handle_lower_trim",
    ],
    "preprocess_with_punctuation": [
        "strip_accents_unicode",
        "replace_punctuation",
        "remove_newline",
        "insert_space_around_punctuation",  # punctuation will be kept. Must be work with
        "handle_lower_trim",
    ],
    "preprocess_merge_abbr": [
        "strip_accents_unicode",
        "replace_punctuation",
        "remove_newline",
        "merge_abbreviations",  # merge all abbreviation
        "merge_&",
        "strip_punctuation",
        "handle_lower_trim",
        "map_shorthands",
    ],
    "preprocess_merge_legal_abbr": [
        "strip_accents_unicode",
        "replace_punctuation",
        "remove_newline",
        "handle_lower",  # merge only legal form abbreviation
        "merge_legal_form_abbreviations",
        "strip_punctuation",
        "handle_trim",
        "remove_extra_space",
    ],
}


class AbstractPreprocessor(Module):
    """Base class of Name Preprocessor"""

    def __init__(
        self,
        preprocess_pipeline: Any = "preprocess_merge_abbr",
        input_col: str = "name",
        output_col: str = "preprocessed",
        spark_session: Any | None = None,
    ) -> None:
        """Base class of Name Preprocessor

        Cleaning and standardization of input names and their legal entity forms. Perform string cleaning, to-lower,
        remove punctuation and white spaces, convert legal entity forms to standard abbreviations.

        Four predefined options for "preprocess_pipeline":

        - "preprocess_name": normal cleaning, remove punctuation, handle unicode, lower and trim
        - "preprocess_with_punctuation": normal cleaning. punctuation will be kept, insert spaces around it.
        - "preprocess_merge_abbr": normal cleaning. merge all abbreviations. (default.)
        - "preprocess_merge_legal_abbr": normal cleaning. merge only legal form abbreviation.

        See `emm.preprocessing.base_name_preprocessor.DEFINED_PIPELINE_DICT` for details.

        Args:
            preprocess_pipeline: default is "preprocess_merge_abbr". Perform string cleaning, to-lower, remove
                                    punctuation and white spaces, convert legal entity forms to standard abbreviations.
            input_col: column name of input names. optional. default is "name".
            output_col: column name of output names. optional. default is "preprocessed".
            spark_session: spark session for processing. default processing is local. optional.
        """
        self.input_col = input_col
        self.output_col = output_col
        self.spark_session = spark_session
        if isinstance(preprocess_pipeline, list):  # custom pipeline
            self.preprocess_list = preprocess_pipeline
        elif isinstance(preprocess_pipeline, str):  # defined pipeline (type==str)
            self.preprocess_list = DEFINED_PIPELINE_DICT[preprocess_pipeline]
        else:
            msg = f"wrong type: {preprocess_pipeline!r}"
            raise TypeError(msg)
        super().__init__()

    def create_func_dict(self) -> dict[str, Any]:
        return create_func_dict()
