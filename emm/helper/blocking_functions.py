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

"""List of blocking functions.

Their names are used to name indexers.
Please don't modify the function names.
"""

from __future__ import annotations

from typing import Callable


def first(x: str) -> str:
    """First character blocking function."""
    return x.strip().lower()[:1]


def first2(x: str) -> str:
    """First two characters blocking function."""
    return x.strip().lower()[:2]


def first3(x: str) -> str:
    """First two characters blocking function."""
    return x.strip().lower()[:3]


BLOCKING_MAP = {"first": first, "first2": first2, "first3": first3}


def _parse_blocking_func(input: Callable[[str], str] | str | None = None) -> Callable[[str], str] | None:
    """Helper function to get blocking function

    Args:
        input: blocking function or name of existing blocking function

    Returns:
        blocking function or None
    """
    if input is None or callable(input):
        return input
    if isinstance(input, str):
        if input not in BLOCKING_MAP:
            msg = f"Input {input} is not a recognized blocking function."
            raise ValueError(msg)
        return BLOCKING_MAP[input]

    msg = "Input is not None, no string and not callable."
    raise TypeError(msg)
