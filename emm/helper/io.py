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

import pickle
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING

import joblib

from emm.helper import spark_installed
from emm.helper.custom_path import CustomPath

if TYPE_CHECKING:
    from pathlib import Path

if spark_installed:
    from pyspark.sql import SparkSession


def _spark_load_from_s3(path: str) -> object:
    spark = SparkSession.builder.getOrCreate()
    file = spark.sparkContext.binaryFiles(path)
    return file.collect()[0][1]


def load_joblib(file_path: str | Path, directory: str | Path | None = None) -> object:
    """Load object from (possibly compressed) joblib file

    Args:
        file_path: full file path, or optionally only file name using the additional directory argument.
        directory: directory corresponding to file name, is then joined with file name (optional)
    """
    # for bkw compatibility, join directory and file name
    path = CustomPath(directory) / file_path if directory else CustomPath(file_path)

    if not path.is_local and spark_installed:
        # try loading with spark, eg. from s3
        data = _spark_load_from_s3(str(path))
        buf = BytesIO(data)
        return joblib.load(buf)
    return joblib.load(path)


def load_pickle(file_path: str | Path, directory: str | Path | None = None) -> object:
    """Load object from pickle file

    Args:
        file_path: full file path, or optionally only file name using the additional directory argument.
        directory: directory corresponding to file name, is then joined with file name (optional)
    """
    # for bkw compatibility, join directory and file name
    path = CustomPath(directory) / file_path if directory else CustomPath(file_path)

    if not path.is_local and spark_installed:
        # try loading with spark, eg. from s3
        data = _spark_load_from_s3(str(path))
        buf = BytesIO(data)
        return pickle.load(buf)
    with open(path, "rb") as f_in:
        return pickle.load(f_in)


class IOFunc:
    """Reader and writer functions used inside SparkCustomWriter/Reader classes

    Container with reading and writing function. Used for reading and storage of non-spark objects.
    By default these are set to joblib's load and dump functions.

    Note: reader and writer are global attributes, so they get picked up by all classes that use IOFunc,
    and only need to be set once.

    Examples:
        >>> io = IOFunc()
        >>> io.writer = pickle.dump
        >>> io.reader = pickle.load
    """

    # reader function (for local object and non-local objects with spark)
    _reader = load_joblib
    # writer function (for local objects only)
    _writer = partial(joblib.dump, compress=True)

    @property
    def writer(self):
        return IOFunc._writer

    @writer.setter
    def writer(self, func):
        if callable(func):
            IOFunc._writer = func
        else:
            msg = "Provided function is not callable."
            raise TypeError(msg)

    @property
    def reader(self):
        return IOFunc._reader

    @reader.setter
    def reader(self, func):
        if callable(func):
            IOFunc._reader = func
        else:
            msg = "Provided function is not callable."
            raise TypeError(msg)

    def set_reader(self, func, call_inside_joblib_load=False, call_inside_pickle_load=False):
        """Set the reader function

        Args:
            func: input reader function
            call_inside_joblib_load: if true, set the reader function as: joblib.load(func(path)).
            call_inside_pickle_load: if true, set the reader function as: pickle.load(func(path)).
        """
        if not callable(func):
            msg = "Provided function is not callable."
            raise TypeError(msg)
        if call_inside_joblib_load:

            def load_data(path, func):
                return joblib.load(func(path))

            IOFunc._reader = partial(load_data, func=func)
        elif call_inside_pickle_load:

            def load_data(path, func):
                return pickle.load(func(path))

            IOFunc._reader = partial(load_data, func=func)
        else:
            IOFunc._reader = func


def save_file(file_path, obj, dump_func=pickle.dump, **kwargs):
    with open(file_path, "wb") as f_out:
        dump_func(obj, f_out, **kwargs)
