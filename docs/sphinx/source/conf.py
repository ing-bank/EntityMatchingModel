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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from importlib import import_module
from inspect import getsource

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx import addnodes

sys.path.insert(0, os.path.abspath("../../../"))


# -- Project information -----------------------------------------------------

project = "Entity Matching Model"
copyright = "2023, ING"
author = "ING"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "Entity Matching Model API"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_default_options = {"undoc-members": True, "exclude-members": "__weakref__,_abc_impl"}

# sphinx_autodoc_typehints settings
always_document_param_types = True

# show todos
todo_include_todos = True

# nbsphinx configuration
nbsphinx_execute = "always" if os.environ.get("NBSPHINX_FORCE_EXECUTE") is not None else "auto"
nbsphinx_allow_errors = True
here = os.path.dirname(__file__)
repo = os.path.join(here, "..", "..", "..")
nbsphinx_link_target_root = repo


class PrettyPrintIterable(Directive):
    required_arguments = 1

    def run(self):
        def _get_iter_source(src, varname):
            # 1. identifies target iterable by variable name, (cannot be spaced)
            # 2. determines iter source code start & end by tracking brackets
            # 3. returns source code between found start & end
            start = end = None
            open_brackets = closed_brackets = 0
            for i, line in enumerate(src):
                if line.startswith(varname) and start is None:
                    start = i
                if start is not None:
                    open_brackets += sum(line.count(b) for b in "([{")
                    closed_brackets += sum(line.count(b) for b in ")]}")

                if open_brackets > 0 and (open_brackets - closed_brackets == 0):
                    end = i + 1
                    break
            return "\n".join(src[start:end])

        module_path, member_name = self.arguments[0].rsplit(".", 1)
        src = getsource(import_module(module_path)).split("\n")
        code = _get_iter_source(src, member_name)

        literal = nodes.literal_block(code, code)
        literal["language"] = "python"

        return [addnodes.desc_name(text=member_name), addnodes.desc_content("", literal)]


def setup(app):
    app.add_directive("pprint", PrettyPrintIterable)
