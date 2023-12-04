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

import pandas as pd

from emm import PandasEntityMatching
from emm.data.create_data import create_example_noised_names
from emm.helper import spark_installed

if spark_installed:
    from emm import SparkEntityMatching


def example():
    """Simple entity matching example using PandasEntityMatching"""
    # This is the example shown in the readme.
    # if you update this example, please update the readme and vice versa!

    # generate example ground truth names and matching noised names, with typos and missing words
    ground_truth, noised_names = create_example_noised_names(random_seed=43)
    train_names, test_names = noised_names[:5000], noised_names[5000:]

    # two example name-pair candidate generators: character-based cosine similarity and sorted neighbouring indexing
    indexers = [
        {
            "type": "cosine_similarity",
            "tokenizer": "characters",  # character-based cosine similarity
            "ngram": 2,  # 2-gram tokens only
            "num_candidates": 5,  # max 5 candidates per name-to-match
            "cos_sim_lower_bound": 0.2,  # lower bound on cosine similarity
        },
        {"type": "sni", "window_length": 3},  # sorted neighbouring indexing window of size 3.
    ]
    em_params = {
        "name_only": True,  # only consider name information for matching
        "entity_id_col": "Index",  # important to set both index and name columns
        "name_col": "Name",
        "indexers": indexers,
        "supervised_on": False,  # no initial supervised model to select best candidates right now
        "with_legal_entity_forms_match": True,  # add feature that indicates match of legal entity forms (eg. ltd != co)
    }
    # initialize the entity matcher
    p = PandasEntityMatching(em_params)
    # prepare the indexers based on the ground truth names: e.g. fit the tfidf matrix of the first indexer.
    p.fit(ground_truth)

    # pandas dataframe with name-pair candidates, made by the indexers. all names have been preprocessed.
    candidates_pd = p.transform(test_names)
    candidates_pd.head()

    # create and fit a supervised model for the PandasEntityMatching object to pick the best match (this takes a while)
    # input is "positive" names column 'Name' that are all supposed to match to the ground truth,
    # and an id column 'Index' to check with candidate name-pairs are matching and which not.
    # A fraction of these names may be turned into negative names (no match to the ground truth).
    # (internally candidate name-pairs are automatically generated, which are input for the classification)
    p.fit_classifier(train_positive_names_to_match=train_names, create_negative_sample_fraction=0.5)

    # generated name-pair candidates, now with classifier-based probability of match.
    # Input is the names' column 'Name'. In the output candidates df, see extra column 'nm_score'.
    candidates_scored_pd = p.transform(test_names)
    candidates_scored_pd.head()

    # for each name-to-match, select the best ground-truth candidate
    best_candidates = candidates_scored_pd[candidates_scored_pd.best_match].copy()

    # print some performance statistics (which is possible in this example as we know the correct match).
    best_candidates["correct"] = best_candidates["gt_entity_id"] == best_candidates["entity_id"]
    print(f"Number of names-to-match: {len(test_names)}")
    print(f"Number of best candidates: {len(best_candidates)}")
    print(f"Number of correct matches: {len(best_candidates[best_candidates.correct])}")
    print(f"Number of incorrect matches: {len(best_candidates[~best_candidates.correct])}")

    # return these numbers for unit-testing
    n_ground_truth = len(ground_truth)
    n_noised_names = len(noised_names)
    n_names_to_match = len(test_names)
    n_best_match = len(best_candidates)
    n_correct = len(best_candidates[best_candidates.correct])
    n_incorrect = len(best_candidates[~best_candidates.correct])

    return (n_ground_truth, n_noised_names, n_names_to_match, n_best_match, n_correct, n_incorrect)


def example_pandas():
    """Simple pandas entity matching example using PandasEntityMatching"""
    # Another example, but this time in pandas with dummy ground truth and names-to-match.
    # (Otherwise same settings as the pandas example above.)

    ground_truth = pd.DataFrame(
        {"name": ["Apple", "Microsoft", "Google", "Amazon", "Netflix", "Spotify"], "id": [1, 2, 3, 4, 5, 6]}
    )
    train_names = pd.DataFrame(
        {"name": ["MicorSoft", "Gugle", "Netfliks", "Spot-on", "Spot-off"], "id": [2, 3, 5, 6, 6]}
    )
    test_names = pd.DataFrame(
        {"name": ["Apl", "Aplle", "Microbloft", "Netflfli", "amz", "googol"], "id": [1, 1, 2, 5, 4, 3]}
    )

    # two example name-pair candidate generators: character-based cosine similarity and sorted neighbouring indexing
    indexers = [
        {
            "type": "cosine_similarity",
            "tokenizer": "characters",  # character-based cosine similarity
            "ngram": 2,  # 2-gram tokens only
            "num_candidates": 5,  # max 5 candidates per name-to-match
            "cos_sim_lower_bound": 0.2,  # lower bound on cosine similarity
        },
        {"type": "sni", "window_length": 3},  # sorted neighbouring indexing window of size 3.
    ]
    emm_config = {
        "name_only": True,  # only consider name information for matching
        "entity_id_col": "id",  # important to set both index and name columns
        "name_col": "name",
        "indexers": indexers,
        "supervised_on": False,  # no initial supervised model to select best candidates right now
    }

    # fitting of first the ground truth, then the training names to match.
    model = PandasEntityMatching(emm_config)
    model.fit(ground_truth)
    model.fit_classifier(train_names, create_negative_sample_fraction=0.5)

    candidates_scored = model.transform(test_names)

    best_candidates = candidates_scored[candidates_scored.score_0 > 0][["name", "gt_name", "gt_entity_id"]]

    best_candidates.head()
    """
    +----------+---------+------------+
    |      name|  gt_name|gt_entity_id|
    +----------+---------+------------+
    |       Apl|    Apple|           1|
    |     Aplle|    Apple|           1|
    |Microbloft|Microsoft|           2|
    |  Netflfli|  Netflix|           5|
    |       amz|   Amazon|           4|
    |    googol|   Google|           3|
    +----------+---------+------------+
    """
    # return dataframe for unit-testing
    return best_candidates


def example_spark(spark):
    """Simple spark entity matching example using SparkEntityMatching"""
    # Another example, but this time in spark, with dummy ground truth and names-to-match.
    # (Otherwise same settings as the pandas example above.)

    ground_truth = spark.createDataFrame(
        [("Apple", 1), ("Microsoft", 2), ("Google", 3), ("Amazon", 4), ("Netflix", 5), ("Spotify", 6)], ["name", "id"]
    )
    train_names = spark.createDataFrame(
        [("MicorSoft", 2), ("Gugle", 3), ("Netfliks", 5), ("Spot-on", 6), ("Spot-off", 6)], ["name", "id"]
    )
    test_names = spark.createDataFrame(
        [("Apl", 1), ("Aplle", 1), ("Microbloft", 2), ("Netflfli", 5), ("amz", 4), ("googol", 3)], ["name", "id"]
    )

    # two example name-pair candidate generators: character-based cosine similarity and sorted neighbouring indexing
    indexers = [
        {
            "type": "cosine_similarity",
            "tokenizer": "characters",  # character-based cosine similarity
            "ngram": 2,  # 2-gram tokens only
            "num_candidates": 5,  # max 5 candidates per name-to-match
            "cos_sim_lower_bound": 0.2,  # lower bound on cosine similarity
        },
        {"type": "sni", "window_length": 3},  # sorted neighbouring indexing window of size 3.
    ]
    emm_config = {
        "name_only": True,  # only consider name information for matching
        "entity_id_col": "id",  # important to set both index and name columns
        "name_col": "name",
        "indexers": indexers,
        "supervised_on": False,  # no initial supervised model to select best candidates right now
    }

    # fitting of first the ground truth, then the training names to match.
    model = SparkEntityMatching(emm_config)
    model.fit(ground_truth)
    model.fit_classifier(train_names, create_negative_sample_fraction=0.5)

    candidates_scored = model.transform(test_names)

    best_candidates = candidates_scored.where(candidates_scored.score_0 > 0).select("name", "gt_name", "gt_entity_id")

    best_candidates.show()
    """
    +----------+---------+------------+
    |      name|  gt_name|gt_entity_id|
    +----------+---------+------------+
    |       Apl|    Apple|           1|
    |     Aplle|    Apple|           1|
    |Microbloft|Microsoft|           2|
    |  Netflfli|  Netflix|           5|
    |       amz|   Amazon|           4|
    |    googol|   Google|           3|
    +----------+---------+------------+
    """
    # return dataframe for unit-testing
    return best_candidates.toPandas()
