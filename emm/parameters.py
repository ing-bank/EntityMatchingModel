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

"""Default parameters for Entity Matching."""

from __future__ import annotations

from pathlib import Path

from emm.helper import blocking_functions, util

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent

# default model parameters picked up in PandasEntityMatching and SparkEntityMatching
MODEL_PARAMS = {
    # type of name preprocessor defined in name_preprocessing.py
    "preprocessor": "preprocess_merge_abbr",
    "indexers": [
        {"type": "cosine_similarity", "tokenizer": "words", "ngram": 1, "num_candidates": 10},
        {
            "type": "cosine_similarity",
            "tokenizer": "characters",
            "ngram": 2,
            "num_candidates": 10,
            "blocking_func": blocking_functions.first,
        },
        {
            "type": "sni",  # Sorted Neighbourhood Indexing,
            "window_length": 3,
        },
    ],
    "partition_size": 5000,  # Number of names in ground_truth and names_to_match per Spark partition: across-worker division. (Set to None for no automatic repartitioning)
    # input columns:
    "entity_id_col": "id",  # This is the id column, only to deal with alternative names and in EM group by account. default is 'id'.
    "name_col": "name",
    "country_col": "country",  # country information that a name belongs to. optional info, picked up in comparison when name_only is False.
    "uid_col": "uid",  # This column is a unique id that need to be in ground_truth and in names_to_match. (Set to None for automatic generation)
    "account_col": "account",  # Needed for aggregation: aggregation of name-matching scores of names that belong together. For example, all names used to address an external bank account.
    "freq_col": "counterparty_account_count_distinct",  # Needed for aggregation: frequency of how often a name is used in a cluster of names that belong together.
    "keep_all_cols": False,  # This is used if you want to keep all the pipeline temporary columns, like the vectorized names columns
    "streaming": False,
    "supervised_on": False,  # To activate the supervised layer
    "name_only": True,  # False: we use the country feature in the supervised model. (Before this param was switching from NM to EM, now we have aggregation_layer)
    "supervised_model_object": None,  # use in-memory supervised model
    "supervised_model_dir": Path("./"),  # can be used to set default location of trained sklearn models
    "aggregation_layer": False,  # The aggregation on account level
    "aggregation_method": "max_frequency_nm_score",  # 'max_frequency_nm_score', 'mean_score'. Needs 'account_col' and 'freq_col'.
    "aggregation_blacklist": [],  # list of names to blacklist in clustering. see data/cluster_blacklist.py
    "return_sm_features": False,  # if True returns supervised model features
    "without_rank_features": False,  # calcfeatures and supervised model without rank features
    "with_legal_entity_forms_match": False,  # if True, add match of legal entity forms feature
    "n_threads": 1,  # desired number of parallel threads in spark candidate selection. default 1.
    "force_execution": False,  # force spark execution (count) in spark candidate selection. default is false (lazy execution).
    "unpersist_broadcast": False,  # after spark indexer transform, free up memory that has been broadcast.
    "with_no_matches": False,  # if true, for each name with no match add an artificial name-pair candidate row.
    "carry_on_cols": [],  # list of column names that should always be copied to the dataframe with candidates if present. GT columns get prefix 'gt_'.
}

# default indexer settings. These are picked up when corresponding settings are missing in MODEL_PARAMS["indexers"]
DEFAULT_INDEXER_PARAMS = {
    "cosine_similarity": {
        "tokenizer": "words",  # "words" or "characters"
        "ngram": 1,  # number of token per n-gram
        "cos_sim_lower_bound": 0.0,
        "num_candidates": 10,  # Number of candidates returned by indexer.
        "binary_countvectorizer": True,  # use binary countVectorizer or not
        # the same value as is used in Spark pipeline in CountVectorizer(vocabSize) 2**25=33554432, 2**24=16777216
        "max_features": 2**25,
        # Python function to be used in blocking ground_truth & names_to_match (only pairs within the same block will be considered in cosine similarity)
        # - None   # No Blocking
        # - blocking_functions.first()  # block using first character
        "blocking_func": None,
    },
    "sni": {
        "window_length": 3,  # window size for SNI
        "mapping_func": None,  # custom mapping function applied in SNI step
    },
    "naive": {},
}

# list of column names that should always be copied to the dataframe with candidates if present
DEFAULT_CARRY_ON_COLS = ["name", "preprocessed", "country", "account", "counterparty_account_count_distinct"]

# update indexer settings with default values in case missing in MODEL_PARAMS["indexers"]
MODEL_PARAMS["indexers"] = util.indexers_set_values(DEFAULT_INDEXER_PARAMS, MODEL_PARAMS["indexers"])
MODEL_PARAMS["carry_on_cols"] = list(set(DEFAULT_CARRY_ON_COLS + MODEL_PARAMS["carry_on_cols"]))

# Example settings for spark driver and executors that work well for large datasets (10M names x 30M names)
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
