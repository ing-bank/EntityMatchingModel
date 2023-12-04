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

"""Helper function for name matching model save and load"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
from pandas.api.types import infer_dtype

from emm.loggers.logger import logger


def rename_columns(df, mapping):
    """Rename columns of Pandas or Spark DataFrame according to the mapping"""
    final_mapping = []
    columns_to_rename = set()
    for it_is, it_should_be in mapping:
        if it_is in df.columns:
            # if the same column has appeared before in the mapping
            if it_is in columns_to_rename:
                tmp = f"{it_is}_copy"
                if isinstance(df, pd.DataFrame):
                    df[tmp] = df[it_is]
                else:
                    df = df.withColumn(tmp, df[it_is])
                it_is = tmp

            columns_to_rename.add(it_is)
            if it_is != it_should_be:
                final_mapping.append((it_is, it_should_be))

    for it_is, it_should_be in final_mapping:
        assert it_is in df.columns, f"Column cannot be renamed because '{it_is}' doesn't exist"
        assert it_should_be not in df.columns, f"Column cannot be renamed because '{it_should_be}' already exist"
        if isinstance(df, pd.DataFrame):
            df = df.rename(columns={it_is: it_should_be})
        else:
            df = df.withColumnRenamed(it_is, it_should_be)
    return df


def string_columns_to_pyarrow(df: pd.DataFrame, columns: list | None = None) -> pd.DataFrame:
    """Convert string columns to pyarrow string datatype

    pyarrow string datatype is much more memory efficient. important for large lists of names (1M+).

    Args:
        df: input pandas dataframe to convert
        columns: columns to convert to pyarrow string type. if None, pick known relevant columns.

    Returns:
        converted dataframe
    """
    if columns is None:
        columns = df.columns
    columns = [col for col in columns if col in df.columns and infer_dtype(df[col]) == "string"]

    logger.debug(f"Converting string column(s) {columns} to pyarrow datatype.")
    for col in columns:
        df[col] = df[col].astype("string[pyarrow]")
    return df


def groupby(data: Iterable, groups: Iterable, postprocess_func: Callable | None = None) -> Mapping:
    """Aggregates `data` using grouping values from `groups`. Returns dictionary with
    keys from `groups` and lists of matching values from `data`. If postprocessing functions is defined
    all dictionary values are processed with this function.
    """
    res = defaultdict(list)
    for i, group in zip(range(data.shape[0]), groups):
        res[group].append(i)
    if postprocess_func is None:
        return {k: data[v] for k, v in res.items()}
    return {k: postprocess_func(data[v]) for k, v in res.items()}


def indexers_set_values(
    default_indexer_params: list[Mapping[str, Any]], indexers: list[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Helper function to update indexer settings

    Update indexer settings with default values where values are missing.
    Used when initializing indexers and in parameters.py.

    Args:
        default_indexer_params: dict with default indexer settings
        indexers: dict with indexer settings that should be updated
    """
    for i in range(len(indexers)):
        if not isinstance(indexers[i], dict):
            continue
        t = indexers[i]["type"]
        indexers[i] = {**default_indexer_params[t], **indexers[i]}
    return indexers


def get_model_title(params: dict) -> str:
    """Construct model title from parameters settings

    Extract model title based on model's indexer settings

    Args:
        params: model parameters
    """
    indexers = params["indexers"]
    title = "__".join([_indexer_to_str(p) for p in indexers])

    if params.get("supervised_on", False):
        title += "__sm"
    if params.get("aggregation_layer", False):
        title += "__agg"

    return title


def _indexer_to_str(params: dict) -> str:
    """Helper function to construct model title from indexer settings

    Args:
        params: indexer parameters
    """
    if params["type"] == "cosine_similarity":
        blocking = "_" + params["blocking_func"].__name__ if params["blocking_func"] is not None else ""
        cos_sim_lower_bound = str(params["cos_sim_lower_bound"]).replace(".", "")
        s = f"{params['tokenizer'][0]}{params['ngram']}_top{params['num_candidates']}_{cos_sim_lower_bound}_{params['max_features']}{blocking}"
    elif params["type"] == "sni":
        mapping = "_mapping" if params["mapping_func"] is not None else ""
        s = f"sni{params['window_length']}{mapping}"
    elif params["type"] == "naive":
        s = "naive"
    else:
        msg = "Unknown indexer"
        raise ValueError(msg)

    # The indexer abbreviation should be a valid HIVE table name:
    # character, number and underscore
    if not re.match("^[A-Za-z0-9_]*$", s):
        msg = "Invalid characters:"
        raise ValueError(msg, s)
    return s
