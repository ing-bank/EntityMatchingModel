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

import numpy as np
import pandas as pd

from emm.data.negative_data_creation import create_positive_negative_samples
from emm.loggers.logger import logger


def prepare_name_pairs(candidates, **kwargs):
    logger.info("Converting candidates from Spark to Pandas")
    return prepare_name_pairs_pd(candidates.toPandas(), **kwargs)


def prepare_name_pairs_pd(
    candidates_pd,
    drop_duplicate_candidates=False,
    drop_samename_nomatch=False,
    create_negative_sample_fraction=0,
    entity_id_col="entity_id",
    gt_entity_id_col="gt_entity_id",
    positive_set_col="positive_set",
    uid_col="uid",
    random_seed=42,
):
    """Prepare dataset of name-pair candidates for training of supervised model.

    This function is used inside em_model.create_training_name_pairs().

    The input are name-pair candidates that are created there, in particular that function creates name-pairs for
    training from positive names that match to the ground truth.

    Positive names are names that are supposed to match to the ground truth. A fraction of the positive names can be
    converted to negative names, which are not supposed to match to the ground truth.

    The creation of negative names drops the negative correct candidates and reranks the remaining negative candidates.

    Args:
        candidates_pd: input positive name-pair candidates created at em_model.create_training_name_pairs().
        drop_duplicate_candidates: if True, drop any duplicate training candidates and keep just one,
                        if available keep the correct match. Recommended for string-similarity models, eg. with
                        without_rank_features=True. default is False.
        drop_samename_nomatch: if True, drop any candidates name-pairs where the two names are equal but which
                        are not match. default is False.
        create_negative_sample_fraction: fraction of name-pairs converted to negative name-pairs. A negative name
                                            has guaranteed no match to any name in the ground truth. default is 0:
                                            no negative names are created.
        entity_id_col: entity id column of names to match, default is "entity_id".
                        For matching name-pairs entity_id == gt_entity_id.
        gt_entity_id_col: entity id column of ground-truth names, default is "gt_entity_id".
                        For matching name-pairs entity_id == gt_entity_id.
        positive_set_col: column that specifies which candidates remain positive and which become negative,
                        default is "positive_set".
        uid_col: uid column for names to match, default is "uid".
        random_seed: random seed for selection of negative names, default is 42.
    """
    """We can have the following dataset.columns, or much more like 'count', 'counterparty_account_count_distinct', 'type1_sum':
    ['uid', 'name', 'preprocessed', 'entity_id', 'country', 'account', 'positive_set',
    'amount', 'gt_uid', 'score_0', 'rank_0', 'gt_entity_id', 'gt_name', 'gt_preprocessed', 'gt_country']
    """
    # Important: the model need to be trained on Preprocessed names, meaning with columns preprocessed and gt_preprocessed
    logger.info("Creating pandas training set from name-pair candidates.")

    # assign label
    assert entity_id_col in candidates_pd.columns
    assert gt_entity_id_col in candidates_pd.columns

    candidates_pd["correct"] = candidates_pd[entity_id_col] == candidates_pd[gt_entity_id_col]

    # negative sample creation?
    # if so, add positive_set_col column for negative sample creation
    rng = np.random.default_rng(random_seed)
    create_negative_sample_fraction = min(create_negative_sample_fraction, 1)
    create_negative_sample = create_negative_sample_fraction > 0
    ids = sorted(candidates_pd[entity_id_col].unique())
    if create_negative_sample and positive_set_col not in candidates_pd.columns:
        logger.info(f"Setting fraction of {create_negative_sample_fraction} of negative ids in training set.")
        n_positive = int(len(ids) * (1.0 - create_negative_sample_fraction))
        pos_ids = list(rng.choice(ids, n_positive, replace=False))
        candidates_pd[positive_set_col] = candidates_pd[entity_id_col].isin(pos_ids)
    elif create_negative_sample and positive_set_col in candidates_pd.columns:
        logger.info(
            f"create_negative_sample_fraction is set, but {positive_set_col} already defined; using the latter."
        )

    # We remove duplicates ground-truth name candidates in the pure string similarity model (i.e. when WITHOUT_RANK_FEATURES==True)
    # because we noticed that when we don't do this, the model learns that perfect match are worst than non-perfect match (like different legal form)
    # meaning that the model will prefer to pick a different candidate than the perfect match.
    # To drop duplicates, when the duplicate ground-truth names:
    # - happens with incorrect/negative case, we just pick one candidate in those duplicate
    # - happens with one correct/positive case, we just pick the correct one
    if drop_duplicate_candidates:
        candidates_pd = candidates_pd.sort_values(
            ["uid", "gt_preprocessed", "correct"], ascending=False
        ).drop_duplicates(subset=["uid", "gt_preprocessed"], keep="first")
    # Similar, for a training set remove all equal names that are not considered a match.
    # This can happen a lot in actual data, e.g. with franchises that are independent but have the same name.
    # It's a true effect in data, but this screws up our intuitive notion that identical names should be related.
    if drop_samename_nomatch:
        samename_nomatch = (candidates_pd["preprocessed"] == candidates_pd["gt_preprocessed"]) & ~candidates_pd[
            "correct"
        ]
        candidates_pd = candidates_pd[~samename_nomatch]

    # Get automatically list of columns that are unique for each uid, i.e. all the names-to-match properties
    cols_max_nunique = candidates_pd.groupby(uid_col).nunique().max()
    names_to_match_cols = [uid_col, *cols_max_nunique[cols_max_nunique == 1].index.tolist()]

    # Get list of unique names-to-match
    names_to_match_before = candidates_pd[names_to_match_cols].drop_duplicates()

    if create_negative_sample:
        # candidates_pd at this point (before being fed into create_positive_negative_samples()
        # is referred to in: resources/data/howto_create_unittest_sample_namepairs.txt
        # create negative sample and rerank negative candidates
        # this drops, in part, the negative correct candidates
        candidates_pd = create_positive_negative_samples(candidates_pd)

    # It could be that we dropped all candidates, so we need to re-introduce the no-candidate rows
    names_to_match_after = candidates_pd[names_to_match_cols].drop_duplicates()
    names_to_match_missing = names_to_match_before.merge(
        names_to_match_after, on=names_to_match_cols, how="left", indicator=True
    )
    names_to_match_missing = names_to_match_missing[names_to_match_missing["_merge"] == "left_only"]
    names_to_match_missing = names_to_match_missing.drop(columns=["_merge"])
    names_to_match_missing["correct"] = False
    # Since this column is used to calculate benchmark metrics
    names_to_match_missing["score_0_rank"] = 1

    candidates_pd = pd.concat([candidates_pd, names_to_match_missing], ignore_index=True)
    candidates_pd["gt_preprocessed"] = candidates_pd["gt_preprocessed"].fillna("")
    candidates_pd["no_candidate"] = candidates_pd["gt_uid"].isnull()

    return candidates_pd
