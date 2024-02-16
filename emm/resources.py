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

# Resources lookup file
import pathlib
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent

# data files that are shipped with emm.
_DATA = {_.name: _ for _ in pathlib.Path(ROOT_DIRECTORY / "emm/data").glob("*.csv.gz")}
# Tutorial notebooks
_NOTEBOOK = {_.name: _ for _ in pathlib.Path(ROOT_DIRECTORY / "notebooks").glob("*.ipynb")}

# Resource types
_RESOURCES = {"data": _DATA, "notebook": _NOTEBOOK}


def _resource(resource_type, name: str) -> str:
    """Return the full path filename of a resource.

    Args:
        resource_type: The type of the resource.
        name: The name of the resource.

    Returns:
        The full path filename of the fixture data set.

    Raises:
        FileNotFoundError: If the resource cannot be found.
    """
    full_path = _RESOURCES[resource_type].get(name, None)

    if full_path and full_path.exists():
        return str(full_path)

    msg = f'Could not find {resource_type} "{name!s}"! Does it exist?'
    raise FileNotFoundError(msg)


def data(name: str) -> str:
    """Return the full path filename of a shipped data file.

    Args:
        name: The name of the data.

    Returns:
        The full path filename of the data.

    Raises:
        FileNotFoundError: If the data cannot be found.
    """
    return _resource("data", name)


def notebook(name: str) -> str:
    """Return the full path filename of a tutorial notebook.

    Args:
        name: The name of the notebook.

    Returns:
        The full path filename of the notebook.

    Raises:
        FileNotFoundError: If the notebook cannot be found.
    """
    return _resource("notebook", name)
