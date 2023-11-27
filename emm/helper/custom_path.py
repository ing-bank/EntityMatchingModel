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

from pathlib import Path


class CustomPath(type(Path())):
    """Custom wrapper for Path class to keep any first double slash

    By default Path supports file paths. However, in practice there are URI schemes that are used to refer to paths
    and for which PurePath manipulations are desirable (in this case S3).
    To accommodate this functionality, the Path class is extended to detect, capture and store the scheme as attribute.
    The remainder of the URI is treated as local path.

    This approach obviously has its limitations, which is the responsibility of the user.
    If your use case requires host, port and/or credential information, you should use proper URI parsing.

    In practice CustomPath acts just like the normal Path class, for any local files.
    However, it prevents the replacement of the first-encountered // by / which happens in Path.
    This makes it possible to also use Path for eg. hdfs or s3 path, not just local ones.

    Example: 's3://foo/bar/bla' => 's3://foo/bar/bla' (and not 's3:/foo/bar/bla')

    This makes is possible to reuse basic Path string manipulation eg of subdirectories for files on s3.
    In particular one can do correctly: new_path = path / 'foo', and str(path).

    For more complex functions, check if CustomPath works, and else use the flag CustomPath.is_local
    and write an alternative.

    Suggestions taken from:
    https://stackoverflow.com/questions/61689391/error-with-simple-subclassing-of-pathlib-path-no-flavour-attribute
    https://stackoverflow.com/questions/49078156/use-pathlib-for-s3-paths

    Other Resources:
    https://en.wikipedia.org/wiki/List_of_URI_schemes
    https://en.wikipedia.org/wiki/File_URI_scheme
    https://docs.aws.amazon.com/cli/latest/reference/s3/

    Args:
        Same as Path.
    """

    def __new__(cls, *args, **kwargs) -> "CustomPath":
        path = str(args[0]) if len(args) > 0 else "."
        path = path if len(path) > 0 else "."
        # store location of '//' for later replacement
        pos = path.find("//")
        cls.schema = path[0:pos] if pos > -1 else None
        return super().__new__(cls, *args, **kwargs)

    @property
    def is_local(self):
        return self.schema is None

    def __str__(self) -> str:
        current_path = super().__str__()
        if self.is_local:
            return current_path
        # replace the single '/' with '//'
        return self.schema + "/" + current_path[len(self.schema) :]

    def as_uri(self):
        """Return the path as a 'file' URI."""
        if self.is_local:
            return super().as_uri()
        return self.__str__()
