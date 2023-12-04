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
from collections import Counter

import numpy as np

AVAILABLE_NOISES = [
    "swap_words",
    "merge_words",
    "drop_word",
    "abbreviate",
    "insert_word",
    "cut_word",
    "split_word",
    "change_word",
]


def create_noiser(names, noise_level, noise_type, random_seed=None):
    """Creates a suitable Noiser class"""
    words = re.findall(r"\w{3,}", " ".join(names))
    # prepare vocabulary for insert_word noise
    insert_vocabulary = [x[0] for x in Counter(words).most_common(20)]
    return Noiser(insert_vocabulary, noise_level, noise_type, random_seed)


class Noiser:
    def __init__(
        self,
        insert_vocabulary: list[str] | None = None,
        noise_threshold: float = 0.3,
        noise_type: str = "all",
        seed: int = 1,
    ) -> None:
        self.insert_vocabulary = insert_vocabulary or ["randomWord"]
        self.noise_threshold = noise_threshold
        # only words longer than 3 chars are considered as words
        self.re_word = re.compile(r"\w{3,}", re.UNICODE)
        self.operations = [
            self.swap_words,
            self.merge_words,
            self.drop_word,
            self.abbreviate,
            self.insert_word,
            self.cut_word,
            self.split_word,
            self.change_word,
        ]
        self.rng = np.random.default_rng(seed)
        operation_dict = dict(zip(AVAILABLE_NOISES, self.operations))
        if noise_type != "all":
            self.operations = [operation_dict[noise_type]]

    def swap_words(self, name):
        words = self.re_word.findall(name)
        if len(words) < 3:
            return name
        words_to_swap = self.rng.choice(words, 2, replace=False)
        name = re.sub(words_to_swap[0], "__temp__ ", name)
        name = re.sub(words_to_swap[1], words_to_swap[0], name)
        return re.sub("__temp__ ", words_to_swap[1], name)

    def merge_words(self, name):
        words = self.re_word.findall(name)
        if len(words) < 3:
            return name
        index = self.rng.choice(len(words) - 1)
        return re.sub(r"" + words[index] + r"\W+" + words[index + 1], words[index] + words[index + 1].lower(), name)

    def drop_word(self, name):
        words = self.re_word.findall(name)
        if len(words) < 3:
            return name
        word_to_drop = self.rng.choice(words)
        return re.sub(r"" + word_to_drop + r"\W+", "", name)

    def abbreviate(self, name):
        abbr_limits = {"lower": 1, "upper": 4}
        words = self.re_word.findall(name)
        if len(words) < 3:
            return name
        abbr_len = self.rng.integers(abbr_limits["lower"], min(len(words), abbr_limits["upper"])) + 1
        max_start = len(words) - abbr_len
        start = self.rng.integers(0, max_start + 1)
        abbr = ""
        for word in words[start : start + abbr_len - 1]:
            abbr += word[0]
            name = re.sub(r"" + word + r"\W+", "", name)
        abbr += words[start + abbr_len - 1][0]
        return re.sub(words[start + abbr_len - 1], abbr, name)

    def insert_word(self, name):
        words = self.re_word.findall(name)
        if len(words) == 0:
            return name
        word_to_append = self.rng.choice(words)
        random_word = self.rng.choice(self.insert_vocabulary)
        return re.sub(word_to_append, word_to_append + " " + random_word, name)

    def cut_word(self, name):
        words = self.re_word.findall(name)
        words = [word for word in words if len(word) >= 8]
        if len(words) == 0:
            return name
        word_to_cut = self.rng.choice(words)
        cut_point = self.rng.choice([4, 5])
        return re.sub(word_to_cut, word_to_cut[:cut_point] + ".", name)

    def split_word(self, name):
        words = self.re_word.findall(name)
        words = [word for word in words if len(word) >= 8]
        if len(words) == 0:
            return name
        word_to_split = self.rng.choice(words)
        split_point = self.rng.choice([4, 5])
        return re.sub(word_to_split, word_to_split[:split_point] + " " + word_to_split[split_point:], name)

    def drop_letter(self, word):
        drop_point = self.rng.choice(len(word) - 1)
        return word[:drop_point] + word[drop_point + 1 :]

    def insert_letter(self, word):
        insert_point = self.rng.choice(len(word) - 1)
        random_char = chr(self.rng.choice(26) + 97)
        return word[:insert_point] + random_char + word[insert_point:]

    def swap_letter(self, word):
        swap_point = self.rng.choice(len(word) - 2)
        return word[:swap_point] + word[swap_point + 1] + word[swap_point] + word[swap_point + 2 :]

    def change_letter(self, word):
        change_point = self.rng.choice(len(word) - 1)
        random_char = chr(self.rng.choice(26) + 97)
        return word[:change_point] + random_char + word[change_point + 1 :]

    def change_word(self, name):
        words = self.re_word.findall(name)
        if len(words) == 0:
            return name
        word_to_change = self.rng.choice(words)
        mutate = self.rng.choice([self.drop_letter, self.insert_letter, self.swap_letter, self.change_letter])
        return re.sub(word_to_change, mutate(word_to_change), name)

    def noise(self, name):
        for operation in self.operations:
            if self.rng.random() < self.noise_threshold:
                name = operation(name)
        return name
