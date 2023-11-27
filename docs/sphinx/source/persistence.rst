Persistence
===========

Here's how to save and load entity matching models.

Store and load pandas-based model

.. code-block:: python

  p.save("pandas_entity_matching_model.pkl")
  from emm import PandasEntityMatching
  p2 = PandasEntityMatching.load("pandas_entity_matching_model.pkl")

Apply the pandas model as usual:

.. code-block:: python

  p2.transform(names_pandas)

Store and reopen a spark-based model in the same way, but in a directory

.. code-block:: python

  s.save("spark_entity_matching_model")
  from emm import SparkEntityMatching
  s2 = SparkEntityMatching.load("spark_entity_matching_model")


For both pandas and spark, by default we use the ``joblib`` library with compression
to store and load all non-spark objects.

The load and dump functions used can be changed to different functions:

.. code-block:: python

  io = emm.helper.io.IOFunc()
  io.writer = pickle.dump
  io.reader = pickle.load

Note that ``reader`` and ``writer`` are global attributes, so they get picked up by all
classes that use ``IOFunc``, and only need to be set once.

For example, one will need to change these functions for writing and reading to ``s3``.
