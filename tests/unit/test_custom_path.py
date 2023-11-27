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

from emm.helper.custom_path import CustomPath


def test_custom_path1():
    # path with //, keep the (first) //. Otherwise acts like path
    p1 = CustomPath("s3://foo/bar")
    assert p1.is_local is False
    assert str(p1) == "s3://foo/bar"
    assert str(p1 / "bla") == "s3://foo/bar/bla"
    assert p1.as_uri() == "s3://foo/bar"


def test_custom_path2():
    # for local paths, CustomPath acts like normal pathlib.Path
    p2 = CustomPath("/foo/bar")
    assert p2.is_local is True
    assert str(p2) == "/foo/bar"
    assert str(p2 / "bla") == "/foo/bar/bla"
    assert p2.as_uri() == "file:///foo/bar"
