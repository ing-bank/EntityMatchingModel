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

import random
import tempfile
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from emm.data.noiser import create_noiser
from emm.features.features_vocabulary import Vocabulary
from emm.helper import spark_installed
from emm.loggers.logger import logger
from emm.preprocessing.pandas_preprocessor import PandasPreprocessor
from emm.resources import _RESOURCES

if spark_installed:
    from pyspark.sql.types import BooleanType, FloatType, IntegerType, StringType, StructField, StructType

# location of Dutch chamber of commerce (kvk) example dataset
KVK_URL = "https://web.archive.org/web/20140225151639if_/http://www.kvk.nl/download/LEI_Full_tcm109-377398.csv"


def _get_data_from_url(url: str) -> bytes:
    """Get data from URL with retries"""
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])

    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))

    return s.get(url, headers={"User-Agent": "Mozilla/5.0"}).content


def _retrieve_complete_kvk_data(
    url: str = KVK_URL,
    store_local: bool = True,
    ignore_local: bool = False,
    use_columns: list = ["registeredName", "legalEntityIdentifier"],
):
    """Download a Dutch chamber of commerce dataset.

    Download to a local file. Try to open local copy first, else download it. (38 mb)

    Args:
        url: url to retrieve
        store_local: store downloaded kvk file locally, default is true.
        ignore_local: ignore local file, default is false.
        use_columns: subset of columns to use

    Returns:
        tuple of path and dataframe
    """
    # download to data location
    local_path = Path(tempfile.gettempdir()) / url.split("/")[-1]

    # download url to local path. will not overwrite local copy
    if store_local and not local_path.is_file():
        logger.info(f"Downloading: {url}")
        with local_path.open("wb") as f:
            data = _get_data_from_url(url)
            f.write(data)
            # to known data resources
            _RESOURCES["data"][local_path.name] = local_path

    # pick up local file if it exists. Or ignore it if requested.
    path = local_path if local_path.is_file() else url
    path = url if ignore_local else path

    # note that read_csv can open url directly as well, but that will not store a local copy.
    df = pd.read_csv(path, sep=";", usecols=use_columns, encoding="ISO-8859-1")

    df.rename(columns={"registeredName": "Name"}, inplace=True)
    df["Index"] = df.index

    return path, df


def retrieve_kvk_test_sample(
    url: str = KVK_URL,
    n: int = 6800,
    random_state: int = 42,
    store_local: bool = True,
    ignore_local: bool = False,
    use_columns: list = ["registeredName", "legalEntityIdentifier"],
):
    """Get sample of the complete kvk data for unit testing

    For testing and demoing we only need a small subset of the complete kvk dataset. (470kb)

    Args:
        url: location to download the data from
        n: number of data records from complete kvk dataset, up to maximum of 6800. default is 6800.
        random_state: seed to use
        store_local: store downloaded kvk file locally, default is true.
        ignore_local: ignore local file, default is false.
        use_columns: subset of columns to use

    Returns:
        tuple of path and sample kvk dataframe
    """
    # construct local file path
    local_path = Path(tempfile.gettempdir()) / url.split("/")[-1]
    local_path = local_path.with_name(f"{local_path.stem}_r{random_state}_s{n}{local_path.suffix}")

    if not ignore_local and local_path.is_file():
        return local_path, pd.read_csv(local_path)

    # sample from COMPLETE kvk dataset. this needs to be downloaded
    _, df = _retrieve_complete_kvk_data(url=url, store_local=False, ignore_local=ignore_local, use_columns=use_columns)

    # random data points to select from df
    # truncate n to max of complete dataset
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=random_state, replace=False)
    sample.reset_index(drop=True, inplace=True)

    if store_local and not local_path.is_file():
        sample.to_csv(local_path, index=False)
        # add file to known data resources
        _RESOURCES["data"][local_path.name] = local_path

    return local_path, sample


def pandas_split_data(data_path=None, name_col="Name", index_col="Index"):
    """Split pandas dataset based on duplicate company ids

    Args:
        data_path: path of input csv file
        name_col: name column in csv file
        index_col: name-id column in csv file (optional)

    Returns:
        ground_truth and negative pandas dataframes
    """
    if data_path is None:
        # location of local sample of kvk unit test dataset; downloads the dataset in case not present.
        data_path, _ = retrieve_kvk_test_sample()

    # Prepare the ground truth names from public dataset
    companies_pd = pd.read_csv(data_path)
    if name_col not in companies_pd.columns:
        msg = f'Name column "{name_col}" not in data columns: {companies_pd.columns}'
        raise RuntimeError(msg)
    cols = [name_col] if index_col not in companies_pd.columns else [name_col, index_col]
    companies_pd = companies_pd.loc[:, cols].drop_duplicates()
    if index_col not in companies_pd.columns:
        companies_pd[index_col] = companies_pd.index.astype("int")
    # convert string based index column to unique integers. Nan/None -> -1
    codes, _ = companies_pd[index_col].factorize()
    companies_pd[index_col] = codes

    # switch to default naming from now on
    companies_pd = companies_pd.rename(columns={name_col: "Name", index_col: "Index"})
    # dummy amount variable
    companies_pd["amount"] = companies_pd["amount"].astype("float") if "amount" in companies_pd.columns else 1.0

    # ground truth are duplicate ids, but ignore Nan/None (-1)
    duplicate_ids = (companies_pd["Index"].duplicated(keep=False)) & (companies_pd["Index"] != -1)
    ground_truth = companies_pd[duplicate_ids].copy()
    negative_pd = companies_pd[~duplicate_ids].copy()

    if len(ground_truth) > 0:
        ground_truth["country"] = "NL"
        ground_truth["account"] = ground_truth["Index"].apply(lambda x: "NL" + str(x + 1000))
    else:
        cols = [*ground_truth.columns.tolist(), "country", "account"]
        ground_truth = pd.DataFrame(columns=cols)
    if len(negative_pd) > 0:
        negative_pd["country"] = "NL"
        negative_pd["account"] = negative_pd["Index"].apply(lambda x: "NL" + str(x + 1000))

    return ground_truth, negative_pd


def split_data(spark, data_path=None, name_col="Name", index_col="Index"):
    """Split dataset into ground truth and negative set based on duplicate company ids

    Args:
        spark: the spark session
        data_path: path of input csv file
        name_col: name column in csv file
        index_col: name-id column in csv file (optional)

    Returns:
        ground_truth and negative spark dataframes
    """
    if data_path is None:
        # location of local sample of kvk unit test dataset; downloads the dataset in case not present.
        data_path, _ = retrieve_kvk_test_sample()

    ground_truth_pd, negative_pd = pandas_split_data(data_path, name_col, index_col)

    # Sparkify dataframes
    schema = StructType(
        [
            StructField("Name", StringType(), True),
            StructField("Index", IntegerType(), nullable=True),
            StructField("amount", FloatType(), True),
            StructField("country", StringType(), True),
            StructField("account", StringType(), True),
        ]
    )

    ground_truth = spark.createDataFrame(ground_truth_pd, schema)
    negative = spark.createDataFrame(negative_pd, schema)

    return ground_truth, negative


def create_example_noised_names(noise_level=0.3, noise_type="all", random_seed=1):
    """Create example noised dataset based on company names from kvk.

    The kvk.csv dataset is sample from an open dataset from the Dutch chamber of commerce.
    open source: https://www.kvk.nl/download/LEI_Full_tcm109-377398.csv
    the relevant column 'registeredName' is already extracted and saved as kvk.csv)

    Args:
        noise_level: float with probability (0.0 < x < 1.0) of adding noise to a name
        noise_type: noise type, default is "all"
        random_seed: seed to use

    Returns:
        ground_truth and noised names, both pandas dataframes
    """
    ground_truth, _, positive_noised_pd, _ = pandas_create_noised_data(
        noise_level=noise_level, noise_type=noise_type, random_seed=random_seed, split_pos_neg=False
    )
    return ground_truth, positive_noised_pd


def pandas_create_noised_data(
    noise_level=0.3,
    noise_type="all",
    noise_count=1,
    split_pos_neg=True,
    data_path=None,
    name_col="Name",
    index_col="Index",
    random_seed=None,
    positive_set_col="positive_set",
):
    """Create pandas noised dataset based on company names from kvk.

    source: https://www.kvk.nl/download/LEI_Full_tcm109-377398.csv
    the relevant column 'registeredName' is already extracted and saved as kvk.csv)

    Args:
        noise_level: float with probability (0.0 < x < 1.0) of adding noise to a name
        noise_type: noise type, default is "all"
        noise_count: integer number of noised names to create per original name. default is 1.
        split_pos_neg: randomly split the dataset into positive and negative set
        data_path: path of input csv file
        name_col: name column in csv file
        index_col: name-id column in csv file (optional)
        random_seed: seed to use
        positive_set_col: name of positive set column in csv file, default is "positive_set".

    Returns:
        ground_truth and companies_noised_pd pandas dataframes
    """
    if data_path is None:
        # location of local sample of kvk unit test dataset; downloads the dataset in case not present.
        data_path, _ = retrieve_kvk_test_sample()

    if not isinstance(noise_count, int) or noise_count < 1:
        msg = "noise_count should be a positive integer."
        raise AssertionError(msg)

    if random_seed is not None:
        # Fix seed for shuffle
        random.seed(random_seed)

    # Prepare the ground truth names from public dataset
    companies_pd = pd.read_csv(data_path)
    if name_col not in companies_pd.columns:
        msg = f'Name column "{name_col}" not in data columns: {companies_pd.columns}'
        raise RuntimeError(msg)
    cols = [name_col] if index_col not in companies_pd.columns else [name_col, index_col]
    companies_pd = companies_pd.loc[:, cols].drop_duplicates()
    if index_col not in companies_pd.columns:
        companies_pd[index_col] = companies_pd.index.astype("int")
    # convert string based index column to unique integers. Nan/None -> -1
    codes, _ = companies_pd[index_col].factorize()
    companies_pd[index_col] = codes

    # switch to default naming from now on
    companies_pd = companies_pd.rename(columns={name_col: "Name", index_col: "Index"})

    # dummy variables:
    companies_pd["amount"] = companies_pd["amount"].astype("float") if "amount" in companies_pd.columns else 1.0
    companies_pd["counterparty_account_count_distinct"] = (
        companies_pd["counterparty_account_count_distinct"].astype("int")
        if "counterparty_account_count_distinct" in companies_pd.columns
        else 1
    )

    companies_pd["uid"] = companies_pd.reset_index().index

    # create noised dataset
    noiser = create_noiser(
        companies_pd["Name"], noise_level=noise_level, noise_type=noise_type, random_seed=random_seed
    )
    companies_noised_pd_list = []

    # Create positive and negative set
    # split based on index so there is no signal leakage
    shuffled_ids = companies_pd["Index"].unique()
    # remove nans (idx==-1)
    shuffled_ids = shuffled_ids[shuffled_ids != -1]
    random.shuffle(shuffled_ids)
    pos = shuffled_ids[: len(shuffled_ids) // 2]
    # ground truth only contains companies in positive set
    is_in_pos = companies_pd["Index"].isin(pos)
    companies_pd[positive_set_col] = is_in_pos

    if split_pos_neg:
        ground_truth = companies_pd[is_in_pos].copy()
        # forget links for negative set. Also affects companies_pd
        negative_pd = companies_pd[~is_in_pos].copy(deep=False)
        negative_pd["Index"] = int(-1)
        # will *not* add noise to the negative set. no need to add extra distortion.
        companies_noised_pd_list.append(negative_pd)
    else:
        # ground truth is full dataset.
        ground_truth = companies_pd.copy(deep=False)
        cols = [*companies_pd.columns.tolist(), "country", "account"]
        negative_pd = pd.DataFrame(columns=cols)

    # Add noise to the positive set (ground truth)
    positive_noised_pd = ground_truth.copy()
    positive_noised_pd["Name"] = positive_noised_pd["Name"].apply(noiser.noise)
    companies_noised_pd_list.append(positive_noised_pd)

    # Add extra copies if so requested.
    # In that case noise is added to both positive and negative sets.
    for _ in range(noise_count - 1):
        companies_noised_pd = companies_pd.copy()
        companies_noised_pd["Name"] = companies_noised_pd["Name"].apply(noiser.noise)
        companies_noised_pd_list.append(companies_noised_pd)

    # Concatenate
    companies_noised_pd = pd.concat(companies_noised_pd_list)

    # Add dummy entity and account features
    ground_truth["country"] = "NL"
    ground_truth["account"] = ground_truth["Index"].apply(lambda x: "NL" + str(x + 1000))
    positive_noised_pd["country"] = "NL"
    positive_noised_pd["account"] = positive_noised_pd["Index"].apply(lambda x: "NL" + str(x + 1000))
    companies_noised_pd["country"] = "NL"
    companies_noised_pd["account"] = companies_noised_pd["Index"].apply(lambda x: "NL" + str(x + 1000))

    if len(negative_pd.index) > 0:
        negative_pd["country"] = "NL"
        negative_pd["account"] = negative_pd["Index"].apply(lambda x: "NL" + str(x + 1000))

    return ground_truth, companies_noised_pd, positive_noised_pd, negative_pd


def create_noised_data(
    spark,
    noise_level=0.3,
    noise_type="all",
    noise_count=1,
    split_pos_neg=True,
    data_path=None,
    name_col="Name",
    index_col="Index",
    ret_posneg=False,
    random_seed=None,
    positive_set_col="positive_set",
):
    """Create spark noised dataset based on company names from kvk.

    source: https://www.kvk.nl/download/LEI_Full_tcm109-377398.csv
    the relevant column 'registeredName' is already extracted and saved as kvk.csv)

    Args:
        spark: the spark session
        noise_level: float with probability (0.0 < x < 1.0) of adding noise to a name
        noise_type: noise type, default is "all"
        noise_count: integer number of noised names to create per original name. default is 0.
        split_pos_neg: randomly split the dataset into positive and negative set
        data_path: path of input csv file
        name_col: name column in csv file
        index_col: name-id column in csv file (optional)
        ret_posneg: if true also return original positive and negative spark true datasets
        random_seed: seed to use
        positive_set_col: name of positive set column in csv file, default is "positive_set".

    Returns:
        ground_truth and companies_noised_pd spark dataframes
    """
    if data_path is None:
        # location of local sample of kvk unit test dataset; downloads the dataset in case not present.
        data_path, _ = retrieve_kvk_test_sample()

    # name_col and index_col get renamed to Name and Index
    (ground_truth_pd, companies_noised_pd, positive_noised_pd, negative_pd) = pandas_create_noised_data(
        noise_level,
        noise_type,
        noise_count,
        split_pos_neg,
        data_path,
        name_col,
        index_col,
        random_seed,
        positive_set_col,
    )

    # Sparkify dataframes
    schema = StructType(
        [
            StructField("Name", StringType(), True),
            StructField("Index", IntegerType(), nullable=True),
            StructField("amount", FloatType(), True),
            StructField("counterparty_account_count_distinct", IntegerType(), nullable=True),
            StructField("uid", IntegerType(), nullable=True),
            StructField(positive_set_col, BooleanType(), True),
            StructField("country", StringType(), nullable=True),
            StructField("account", StringType(), True),
        ]
    )
    ground_truth = spark.createDataFrame(ground_truth_pd, schema)
    companies_noised = spark.createDataFrame(companies_noised_pd, schema)
    positive_noised = spark.createDataFrame(positive_noised_pd, schema)
    negative = spark.createDataFrame(negative_pd, schema)

    if ret_posneg:
        return ground_truth, companies_noised, positive_noised, negative
    return ground_truth, companies_noised


def create_training_data() -> tuple[pd.DataFrame, Vocabulary]:
    rows = [
        (0, 0.9, "Ahmet Erdem A.S.", "Ahmet Erdem N.V.", "TR", "NL", True, True, False),
        (1, 0.5, "ING Bank BV", "ASD Bank B.V.", "NL", "NL", False, True, False),
        (2, 1.0, "ING Bank BV", "ING Bank B.V.", "NL", "NL", True, True, False),
        (3, 0.7, "ASD Investment Holding BV", "ASD Bank B.V.", None, "NL", True, True, False),
        (4, 0.4, "ASD Investment Holding", "Investment Holding BV", "EN", "NL", False, True, False),
        (5, 0.2, "Ahmet Erdem A.S.", "Erdem Holding Inc.", "TR", "EN", False, True, False),
        (6, None, "Missing score, no candidates", "Erdem Holding Inc.", "TR", "EN", False, True, False),
        (7, 0.9, "Negative names", "Name one in the GT", "TR", "EN", False, False, False),
        (7, 0.8, "Negative names", "Name two in the GT", "TR", "EN", False, False, False),
        (8, 0.02, "Negative name no candidate", "", "TR", None, False, False, True),
        (9, 0.02, "Positive name no candidate", "", "TR", None, False, True, True),
        (10, 1.0, "Exact match", "Exact match", "NL", "NL", True, True, False),
        (11, 0.8, "Exact match", "Perfect match", "NL", "NL", False, True, False),
        (12, 0.95, "Speling mistake", "Spelling mistake", "NL", "NL", True, True, False),
        (13, 0.96, "Data Quality mistake", "Completly wrong", "NL", "NL", True, True, False),
    ]

    df_small = pd.DataFrame(
        rows,
        columns=[
            "tmp_id",
            "score_0",
            "name",
            "gt_name",
            "country",
            "gt_country",
            "correct",
            "positive_set",
            "no_candidate",
        ],
    )

    # Preprocess both name columns: 'name'->'preprocessed' and 'gt_name'->'gt_preprocessed'
    p1 = PandasPreprocessor(
        preprocess_pipeline="preprocess_name"
    )  # The default value for input_col and output_col are for 'name'
    p2 = PandasPreprocessor(preprocess_pipeline="preprocess_name", input_col="gt_name", output_col="gt_preprocessed")
    df_small = p1.transform(df_small)
    df_small = p2.transform(df_small)

    # multiply data
    df_list = []
    for i in range(5):
        df_small["uid"] = df_small["tmp_id"] + i
        df_list.append(df_small)
    df = pd.concat(df_list)

    df = df.reset_index(drop=True)

    # add unique row identifiers
    df["uid"] = df.index
    df["account"] = df.index
    df["gt_uid"] = df["gt_name"].rank(method="dense").map(int)
    df["gt_uid"] = df.apply(
        lambda r: None if r["no_candidate"] else r["gt_uid"], axis=1
    )  # By convention, gt_uid is null in the no_candidate case

    vocabulary = Vocabulary(very_common_words={"bv", "nv"}, common_words={"bank", "holding"})
    return df, vocabulary
