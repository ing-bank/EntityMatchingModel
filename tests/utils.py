import logging
from io import StringIO

import numpy as np
import pandas as pd

from emm.helper import spark_installed

if spark_installed:
    from pyspark.sql import functions as F
    from pyspark.sql.types import FloatType

logger = logging.getLogger(__name__)


def read_markdown(input_str):
    return (
        pd
        # Read a markdown file
        # Remark: we don't use parameter 'skiprows=[1]' because it allows us to have, if we want, formatting blank lines in 'intput_str'
        # because 'skip_blank_lines' happens after 'skiprows', instead we use 'iloc' + 'reset_index' see below.
        .read_csv(StringIO(input_str), sep=r"\s*[|]+\s*", engine="python")  # skipinitialspace=True not necessary
        # Drop the left-most and right-most null columns
        .dropna(axis=1, how="all")
        # Drop the header underline row
        .iloc[1:]
        # Reset index since we dropped the first row
        .reset_index(drop=True)
        # Infer types
        .apply(pd.to_numeric, errors="ignore")
    )


def add_features_vector_col(em_vec, df, index_col=None, name_col=None):
    """This function is adding to df the column 'features' containing the vectorized version of the column '{name_col}'
    It is only calling em_vec.transform(df) but with some column renaming.
    If {name_col} is not specified it will fall back on the name_col of the {em_vec} EM parameters.
    {em_vec} should be a vectorizer only (i.e. with no indexer, cosim or supervised model)
    """
    em_index_col = em_vec.parameters["entity_id_col"]
    em_name_col = em_vec.parameters["name_col"]

    if index_col is not None and index_col not in df.columns:
        msg = f"Column '{index_col}' is not there"
        raise ValueError(msg)
    if name_col is not None and name_col not in df.columns:
        msg = f"Column '{name_col}' is not there"
        raise ValueError(msg)

    if (em_index_col not in df.columns) & (em_name_col not in df.columns):
        if index_col is not None and name_col is not None:
            # Rename the columns because em_vec is vectorizing only the column name from its parameters
            df = df.withColumnRenamed(index_col, em_index_col)
            df = df.withColumnRenamed(name_col, em_name_col)
        else:
            msg = "Columns are missing, and there no renaming parameters"
            raise ValueError(msg)
    else:
        logger.info("Columns are already there.")
        if index_col is not None or name_col is not None:
            msg = f"cannot rename columns! index_col='{index_col}' name_col='{name_col}'"
            raise ValueError(msg)

    df = em_vec.transform(df)

    if index_col is not None and name_col is not None:
        # Rename back the columns to their original name
        df = df.withColumnRenamed(em_index_col, index_col)
        df = df.withColumnRenamed(em_name_col, name_col)

    # Add only the features columns. i.e. drop the other columns tf, idf, etc
    for col in ["tf", "idf", "tokens", "ngram_tokens"]:
        if col in df.columns:
            df = df.drop(col)

    return df


def get_n_top_sparse(mat, n_top):
    """Get list of (index, value) of the n largest elements in a 1-dimensional sparse matrix"""
    length = mat.getnnz()
    if length == 0:
        return None
    if length <= n_top:
        result = zip(mat.indices, mat.data)
    else:
        arg_idx = np.argpartition(mat.data, -n_top)[-n_top:]
        result = zip(mat.indices[arg_idx], mat.data[arg_idx])
    return sorted(result, key=lambda x: -x[1])


def create_test_data(spark):
    # Mock ground truth
    grd_list = [
        ("Tzu Sun", 1, "NL", "G0001", True),
        ("Tzu General Chinese Sun", 1, "NL", "G0001", True),
        ("Tzu General Dutch Sun", 1, "NL", "G0001", True),
        ("Eddie Arnheim", 2, "NL", "G0002", True),
        ("Eddie Eagle", 2, "NL", "G0002", True),
        ("John Mokker", 3, "NL", "G0003", True),
        ("John little princess", 3, "NL", "G0003", True),
        ("Fokko X", 4, "NL", "G0004", True),
        ("Daniel Y", 5, "NL", "G0005", True),
        ("Delphine Douchy", 6, "NL", "G0006", True),
        ("Blizzard Entertainment B.V.", 7, "NL", "G0007", True),
        ("Sony Entertainment", 8, "NL", "G0008", True),
        ("Anmot Meder Investment", 9, "NL", "G0009", True),
        ("H&M BV", 10, "NL", "G0010", True),
        ("H.M. BV", 11, "NL", "G0011", True),
        ("Vereniging van Vrienden van het Allard Pierson Museum", 12, "NL", "G0012", True),
        ("TANK & TRUCK CLEANING WOERD TTC WOERD", 13, "NL", "G0013", True),
        ("Stephane false match", 14, "NL", "G0013", True),
        ("Wendely Nothing found", 15, "NL", "G0015", False),
        ("Also no match here", 16, "NL", "G0016", False),
        ("Negative", 17, "NL", "G0017", False),
        ("Coca Limited by Shares", 18, "NL", "G0018", True),
        ("Pepsi Limited by Shares", 19, "NL", "G0019", True),
        ("Best match incorrect", 20, "NL", "G0020", True),
        ("Best match not correct", 21, "NL", "G0021", True),
        ("Close match but incorrect for negative", 22, "NL", "G0022", True),
        ("Stephane Gullit", 23, "NL", "G0023", True),
        ("Xam Boko", 24, "NL", "G0024", True),
        ("Tomesk Wolen", 25, "NL", "G0025", True),
        ("Lorrainy D Almoeba", 26, "NL", "G0026", True),
    ]

    grd_pd = pd.DataFrame(grd_list, columns=["name", "id", "country", "account", "positive_set"])
    grd_pd["uid"] = grd_pd.index + 1000
    grd_df = spark.createDataFrame(grd_pd)
    grd_df = grd_df.withColumn("amount", F.lit(1.0).cast(FloatType()))

    # Mock names to match.
    # Negative names
    test_list = [
        ("Tzu Chines Sun", 1, "NL", "0001", True),
        ("Tzu Chines Sun a", 1, "NL", "0001", True),
        ("Tzu Chinese General", 1, "NL", "0001", True),
        ("Eddie Germen Arnheim", 2, "NL", "0002", True),
        ("John Dutch little princess", 3, "NL", "0002", True),
        ("Blizzard Entteretainment BV", 7, "NL", "0004", True),
        ("AE Investment", 9, "NL", "0005", True),
        ("H.M. BV", 10, "NL", "0007", True),
        ("H & M BV", 11, "NL", "0007", True),
        ("VER VAN VRIENDEN VAN HET ALLARD PIERSON MUSEUM", 12, "NL", "0008", True),
        ("Tank & Truck Cleaning Woerd T.T.C. Woerd", 13, "NL", "0009", True),
        ("Eddie Arnheim noise", 14, "NL", "0009", True),
        ("Tzu Sun noise", 14, "NL", "0009", True),
        ("Anmot Meder noise", 14, "NL", "0009", True),
        ("Wendely Nothing found", 15, "NL", "0015", False),  # to drop correct
        ("Also no match here", 16, "NL", "0016", False),  # to drop correct
        ("Negative 3", 17, "NL", "0083", False),
        ("Negative 4", 17, "NL", "0084", False),
        ("Negative 5", 17, "NL", "0085", False),
        ("Negative 6", 17, "NL", "0086", False),
        ("Negative 7", 17, "NL", "0087", False),
        ("Negative 8", 17, "NL", "0088", False),
        ("Negative 9", 17, "NL", "0089", False),
        ("Negative 10", 17, "NL", "0090", False),
        ("Positive no candidate 1", 1, "NL", "1001", True),
        ("Positive no candidate 2", 1, "NL", "1001", True),
        ("Positive no candidate 3", 1, "NL", "1001", True),
        ("Positive no candidate 4", 1, "NL", "1001", True),
        ("Coca Limited by Shares", 18, "NL", "0051", True),
        ("Coca Limited", 18, "NL", "0051", True),
        ("Coca", 18, "NL", "0051", True),
        ("Pepsi Limited by Shares", 19, "NL", "0022", True),
        ("Pepsi Limited", 19, "NL", "0022", True),
        ("Pepsi", 19, "NL", "0022", True),
        ("Best match incorrect different", 21, "NL", "0021", True),
        ("Best match not correct rare one", 20, "NL", "0020", True),
        ("Best match not correct rare two", 20, "NL", "0020", True),
        ("Best match not correct rare three", 20, "NL", "0020", True),
        ("Best match not correct rare four", 20, "NL", "0020", True),
        ("Best match not correct rare five", 20, "NL", "0020", True),
        ("Best match not correct rare six", 20, "NL", "0020", True),
        ("Best match not correct rare seven", 20, "NL", "0020", True),
        ("Best match not correct rare eight", 20, "NL", "0020", True),
        ("Best match not correct rare nine", 20, "NL", "0020", True),
        ("Close match but incorrect for negative", 17, "NL", "0022", False),
        ("Close match but not correct for negative", 17, "NL", "0023", False),
        ("Close match and not correct for negative", 17, "NL", "0024", False),
        ("Stephane Gullit", 23, "NL", "0023", True),
        ("Gullit Stephan", 23, "FR", "0023", True),
        ("Stephane Col.", 23, "NL", "0023", True),
        ("Xam Boko", 24, "NL", "0024", True),
        ("Xam Bok", 24, "NL", "0024", True),
        ("Tomesk Wol len", 25, "NL", "0025", True),
        ("Lorrainy D Almoeba", 26, "NL", "0026", True),
        ("Lorrainy Lorrainy", 26, "NL", "0026", True),
    ]

    base_pd = pd.DataFrame(test_list, columns=["name", "id", "country", "account", "positive_set"])
    # For metric and threshold_decision we need a bit of data to cover all cases in train and valid folds, so let's duplicate:
    test_pd = base_pd.copy()
    for prefix in ["A_", "B_", "C_", "D_"]:
        test2_pd = base_pd.copy()
        test2_pd["name"] = prefix + test2_pd["name"]
        test2_pd["account"] = prefix + test2_pd["account"]
        test_pd = pd.concat([test_pd, test2_pd], ignore_index=True)
    test_pd["uid"] = test_pd.index + 100
    test_df = spark.createDataFrame(test_pd)

    return grd_df, test_df
