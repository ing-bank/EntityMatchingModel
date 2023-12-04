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

import json
import time
from typing import Any

from pyspark.ml.util import DefaultParamsReader, DefaultParamsWritable, MLReader, MLWritable, MLWriter
from pyspark.sql import DataFrame, SparkSession

# CustomPath acts just like the normal Path class for local paths,
# but captures the scheme for URIs that can be treated as paths (to accommodate simple s3 paths)
# Ie. it does not replace first-encountered '//' by '/' (eg. for s3)
from emm.helper.custom_path import CustomPath
from emm.helper.io import IOFunc


class SparkWriteable:
    """Mixin for Spark serialization

    SERIALIZE_ATTRIBUTES are the class attributes that will be serialized
    SPARK_SESSION_KW is the keyword name to pass the spark session to
    """

    SERIALIZE_ATTRIBUTES: list[str]
    SPARK_SESSION_KW: str | None = None

    def write(self):
        """Returns a SparkCustomWriter instance for this class."""
        # split kwargs into regular parameters and spark objects. these are stored (and retrieved) separately.
        # stored as a compressed binary object

        data = {k: getattr(self, k) for k in self.SERIALIZE_ATTRIBUTES}

        # filter empty values
        data = {k: v for k, v in data.items() if v is not None}
        return SparkCustomWriter(instance=self, data=data, spark_session_kw=self.SPARK_SESSION_KW)


class SparkReadable:
    """Mixin for Spark serialization"""

    @classmethod
    def read(cls):
        """Returns a SparkCustomReader instance for this class.

        Assumes that object can be instantiated with obj(**kwargs)
        """
        return SparkCustomReader(cls)


class SparkCustomWriter(MLWriter):
    """Spark custom writer class"""

    def __init__(
        self,
        instance,
        data: dict[str, Any] | None = None,
        spark_session_kw=None,
        file_format: str = "parquet",
        **kwargs,
    ) -> None:
        """Spark custom writer class

        optional: special object writer function, e.g. function for writing a single file to s3.
        if set, use this dumper function to non-local files (eg to s3) for all non-spark objects.
        need to set this as global property as spark does not pass-thru a writer function to sub-objects when calling save()

        set this externally through: IOFunc().writer = func

        Args:
            instance: Instance of spark object to store. Inherits from DefaultParamsReadable, DefaultParamsWritable.
            data: key-word args for object to be stored, needed to reinitialize the object.
            spark_session_kw: key-word of spark session needed to reinitialize the object, if any.
            file_format: storage format of spark dataframes, default is parquet.
            kwargs: storage kw-args, passed on to: sdf.write.save(path, format=self.file_format, **self.kwargs)
        """
        super().__init__()
        self.instance = instance
        self.data = data or {}
        self.spark_session_kw = spark_session_kw
        self.file_format = file_format
        self.store_kws = kwargs

        self._composition = {}
        self._spark_objects = {}
        self._spark_dfs = {}
        self._other_objects = {}

        # function used for storage of non-spark objects. default is joblib.dump()
        self.writer_func = IOFunc().writer

    def saveImpl(self, path: str):
        """Saves metadata + Params to: path + "/metadata"
        - class
        - timestamp
        - sparkVersion
        - uid
        - sparkObjKeys
        - sparkDFNames
        - sparkKW
        - file_format
        - store_kws

        Args:
            path: storage path
        """
        path = CustomPath(path)
        if path.is_local:
            path.mkdir(parents=True, exist_ok=True)

        # extend composition
        composition_data = {}
        for key, obj in self.data.items():
            if isinstance(obj, list):
                self._composition[key] = []
                try:
                    for idx, value in enumerate(obj):
                        ckey = f"{key}{idx}"
                        composition_data[ckey] = value
                        self._composition[key].append(ckey)
                except AttributeError as e:
                    msg = f"{self.__class__.__name__} misses {key}"
                    raise AttributeError(msg) from e

        self.data.update(composition_data)
        for key in self._composition:
            del self.data[key]

        # split data to store into spark dfs, spark objects, and other.
        for key, obj in self.data.items():
            if isinstance(obj, (DefaultParamsWritable, MLWritable)):
                self._spark_objects[key] = obj
            elif isinstance(obj, DataFrame):
                self._spark_dfs[key] = obj
            else:
                self._other_objects[key] = obj

        # store class metadata (json dump)
        # needed for DefaultParamsReader to reconstruct spark pipelines
        metadata_json = self._get_metadata_to_save()
        metadata_path = path / "metadata"
        self.sc.parallelize([metadata_json], 1).saveAsTextFile(str(metadata_path))

        # flexibility to use joblib.dump, e.g. works for numpy arrays
        if len(self._other_objects) > 0:
            data_path = path / "data_joblib.gz"
            self.writer_func(self._other_objects, str(data_path))

        # store spark objects (that don't work with json dump) by calling write().save()
        for key, spark_obj in self._spark_objects.items():
            if callable(getattr(spark_obj, "write", None)):
                obj_path = path / key
                if self.shouldOverwrite:
                    spark_obj.write().overwrite().save(str(obj_path))
                else:
                    spark_obj.write().save(str(obj_path))

        # store spark dfs as files in `file_format`
        for key, sdf in self._spark_dfs.items():
            sdf_path = path / key
            sdf.write.save(str(sdf_path), format=self.file_format, **self.store_kws)

    def _get_metadata_to_save(self):
        """Helper for :py:meth:`DefaultParamsWriter.saveMetadata` which extracts the JSON to save.
        This is useful for ensemble models which need to save metadata for many sub-models.

        .. note:: :py:meth:`DefaultParamsWriter.saveMetadata` for details on what this includes.
        """
        uid = self.instance.uid if hasattr(self.instance, "uid") else 0
        cls = self.instance.__module__ + "." + self.instance.__class__.__name__

        # store spark object keys
        spark_obj_keys = {}
        for key, spark_obj in self._spark_objects.items():
            if callable(getattr(spark_obj, "write", None)):
                obj_cls = spark_obj.__module__ + "." + spark_obj.__class__.__name__
            else:
                obj_cls = ""
            spark_obj_keys[key] = obj_cls

        # store spar kdf keys
        spark_df_names = list(self._spark_dfs.keys())

        metadata = {
            "class": cls,
            "timestamp": int(round(time.time() * 1000)),
            "sparkVersion": self.sc.version,
            "uid": uid,
            "sparkObjKeys": spark_obj_keys,
            "sparkDFNames": spark_df_names,
            "sparkKW": self.spark_session_kw,
            "composition": self._composition,
            "file_format": self.file_format,
            "store_kws": self.store_kws,
        }
        return json.dumps(metadata, separators=[",", ":"])


class SparkCustomReader(MLReader):
    """Spark Custom class reader"""

    def __init__(self, cls) -> None:
        """Spark custom reader class

        optional setting: special object reader function, e.g. function for reading a single file to s3.
        if set, use this reader function of non-local files (eg from s3) for all non-spark objects.
        need to set this as class property as spark does not pass-thru a read function to sub-objects when calling load()

        set this externally through: IOFunc().reader = func
        """
        super().__init__()
        self.cls = cls
        # function used for loading of non-spark objects. default is joblib.load()
        self.reader_func = IOFunc().reader

    def _get_class_module(self, class_str):
        """Loads Python class from its name.

        Cannot call DefaultParamsReader__get_class from here. So copied here for now.
        https://spark.apache.org/docs/2.3.0/api/python/_modules/pyspark/ml/util.html#DefaultParamsReader
        """
        parts = class_str.split(".")
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m

    def _load_spark_objects(self, path: CustomPath, keys) -> dict:
        return {
            key.rstrip("_"): self._get_class_module(obj_cls).load(str(path / key))
            for key, obj_cls in keys.items()
            if len(obj_cls) > 0
        }

    def load(self, path: str):
        """Load and instantiate object from spark directory path

        Args:
            path: directory path

        Returns:
            instantiated object
        """
        path = CustomPath(path)

        spark = SparkSession.builder.getOrCreate()

        # 1. retrieve metadata
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)

        # 2. retrieve possible data objects
        data_path = path / "data_joblib.gz"
        data = self.reader_func(str(data_path))
        data = {key.lstrip("_"): value for key, value in data.items()}

        # 3. retrieve possible spark objects
        spark_objs = self._load_spark_objects(path, metadata.get("sparkObjKeys", {}))

        # 4. retrieve possible spark dataframes
        df_names = metadata.get("sparkDFNames", [])
        file_format = metadata.get("file_format", "parquet")
        spark_dfs = {key.lstrip("_"): spark.read.load(path=str(path / key), format=file_format) for key in df_names}

        # combine all loaded objects
        kwargs = {**data, **spark_objs, **spark_dfs}

        # 5. do recomposition here of lists or dicts of objects from retrieved objects
        composition = metadata.get("composition", {})
        restructured = {}
        for key, comp in composition.items():
            # lists / tuples
            if isinstance(comp, list):
                if not all(obj_name in kwargs for obj_name in comp):
                    msg = f"Not all items in {comp} found in loaded objects."
                    raise KeyError(msg)
                comp_type = type(comp)
                restructured[key] = comp_type([kwargs.pop(obj_name) for obj_name in comp])
            else:
                msg = f"Type {type(comp)} not recognized for restructuring."
                raise TypeError(msg)

        # 6. does object need a spark session at initialization? If so add kw arg.
        spark_session = metadata.get("sparkKW", False)
        spark_kw = {spark_session: spark} if isinstance(spark_session, str) and len(spark_session) > 0 else {}

        # 7. object initialization
        kwargs = {**kwargs, **restructured, **spark_kw}
        py_type = self._get_class_module(metadata["class"])
        instance = py_type(**kwargs)
        if hasattr(instance, "_resetUid"):
            instance._resetUid(metadata.get("uid", 0))

        return instance
