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

from emm.base.module import Module
from emm.version import __version__


class BaseIndexer(Module):
    """Base implementation of Indexer class"""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def version():
        return __version__

    def increase_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This should change the parameter settings of the fitted model.
        """

    def decrease_window_by_one_step(self):
        """Utility function for negative sample creation during training

        This should change the parameter settings of the fitted model.
        """


class CosSimBaseIndexer(BaseIndexer):
    """Base implementation of CosSimIndexer class"""

    def __init__(self, num_candidates: int) -> None:
        super().__init__()
        if num_candidates <= 0:
            msg = "Number of candidates should be a positive integer"
            raise ValueError(msg)
        self.num_candidates = num_candidates

    def increase_window_by_one_step(self) -> None:
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        self.num_candidates += 1

    def decrease_window_by_one_step(self) -> None:
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        self.num_candidates -= 1


class SNBaseIndexer(BaseIndexer):
    """Base implementation of SN Indexer class"""

    def __init__(self, window_length: int) -> None:
        super().__init__()
        if window_length % 2 == 0:
            msg = "SNI window should be odd integer"
            raise ValueError(msg)
        self.window_length = window_length

    def increase_window_by_one_step(self) -> None:
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        self.window_length += 2

    def decrease_window_by_one_step(self) -> None:
        """Utility function for negative sample creation during training

        This changes the parameter settings of the fitted model.
        """
        self.window_length -= 2
