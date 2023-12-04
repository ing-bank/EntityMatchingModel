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

import logging
from contextlib import ContextDecorator
from timeit import default_timer
from typing import Any

logger = logging.getLogger(__name__)


def format_values(values: dict[str, Any]) -> str:
    return ", ".join([f"{key}={value}" for key, value in values.items()])


class Timer(ContextDecorator):
    """Context manager that logs the timing of labelled blocks of code

    Example:
        >>> with Timer("label") as timer:
        >>>     timer.label("part 1")
        >>>     ...
        >>>
        >>>     timer.label("part 2")
        >>>     ...
    """

    def __init__(self, label) -> None:
        self._label = label
        self._start = None
        self._end = None
        self.measurements = {}
        self.values = {}

    def start(self):
        self._start = default_timer()

    def end(self):
        self._end = default_timer()

    def difference(self):
        return self._end - self._start

    def label(self, name: str) -> None:
        """Labelled checkpoint

        Args:
            name: label for block of code

        Raises:
            ValueError: if reserved or used name is provided
        """
        if name in {"start", "end"}:
            msg = f"Reserved name '{name}'"
            raise ValueError(msg)
        if name in self.measurements:
            msg = f"Name '{name}' already used"
            raise ValueError(msg)

        logger.debug("Task '%s' label '%s'", self._label, name)
        self.measurements[name] = default_timer()

    def log_param(self, key: str, value: Any):
        self.log_params({key: value})

    def log_params(self, value: dict):
        logger.debug("%s", format_values(value))
        self.values.update(value)

    def __enter__(self) -> Timer:
        logger.debug("+> Starting task '%s'", self._label)
        self.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.end()
        d = self.difference()
        if self.values:
            values_str = format_values(self.values)
            values_str = f" ({values_str})"
        else:
            values_str = ""

        if self.measurements:
            labels = ["setup", *list(self.measurements.keys())]

            times = list(self.measurements.values())
            times = [end - start for start, end in zip([self._start, *times], [*times, self._end])]

            measurement_str = ", ".join([f"{key}: {value:.3f}s" for key, value in zip(labels, times)])
            measurement_str = f" ({measurement_str})"
        else:
            measurement_str = ""
        logger.info("%s%s time: %.3fs%s", self._label, values_str, d, measurement_str)
        logger.debug("-> Finished task '%s' in: %.3fs", self._label, d)
