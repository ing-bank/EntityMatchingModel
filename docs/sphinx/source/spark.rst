Spark settings
==============

The ``SparkEntityMatching`` tool is great for matching large sets of names.
Here are recommended spark settings for the driver and executors that
in our experience work well for matching large datasets
(10M names x 30M names, on a cluster with ~1000 nodes).

.. code-block:: python

  SPARK_CONFIG_EXAMPLE = {
      "spark.driver.memory": "25G",
      # default overhead = driverMemory * 0.10, with minimum of 384, in MiB unless otherwise specified
      "spark.driver.memoryOverhead": "10G",  # try "32G" if you face memory issues
      # 'spark.driver.cores': '1',  # default: 1
      # Amount of memory that can be occupied by the objects created via the Py4J bridge during a Spark operation,
      # above it spills over to the disk.
      "spark.python.worker.memory": "4G",  # default: 512m
      "spark.executor.memory": "30G",  # default 1G, 30G necessary for scoring
      # unlimited size object accepted by driver in collect() from workers (default 1G).
      # needed to collect large tfidf matrices between workers and driver.
      "spark.driver.maxResultSize": 0,
      "spark.rpc.message.maxSize": 1024,  # 1024mb message transfer size
      # In Spark 3.2+ adaptive shuffling/partitioning is enabled by default.
      # it is important to disable this to keep full control over the partitions and their consistency
      "spark.sql.adaptive.enabled": "false",
      # checkpoint directory are not cleaned up by default, and that leads to waste of HDFS space:
      "spark.cleaner.referenceTracking.cleanCheckpoints": "true",
  }

You can pick up this configuration dictionary with:

.. code-block:: python

  from emm.parameters import SPARK_CONFIG_EXAMPLE
