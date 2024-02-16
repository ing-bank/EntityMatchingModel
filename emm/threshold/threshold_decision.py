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

from sklearn import metrics

from emm.aggregation.pandas_entity_aggregation import PandasEntityAggregation


def _get_threshold_confusion_matrices(y_true, y_prob):
    """Compute confusion matrices

    Args:
        y_true: true labels
        y_prob: scores

    Returns:
        thresholds, TN, FP, FN, TP
    """
    # Let's compute everything based on roc_curve since it has a default optimization to make the curve lighter,
    # unlike precision_recall_curve.
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    # Let's drop the first values because thresholds[0] represents no instances being predicted
    # and is arbitrarily set to "max(y_score) + 1" by roc_curve()
    thresholds = thresholds[1:]
    fpr = fpr[1:]
    tpr = tpr[1:]
    tnr = 1.0 - fpr
    fnr = 1.0 - tpr

    negatives = sum(~y_true)
    positives = sum(y_true)

    # Same order as sklearn
    tn = tnr * negatives  # True Negative Rate * #Negative = TN/N * N = # True Negative
    fp = fpr * negatives  # False Positive Rate * #Negative = FP/N * N = # False Positive = 'wrong_matches'
    fn = fnr * positives  # False Negative Rate * #Positive = FN/P * P = # False Negative
    tp = tpr * positives  # True Positive Rate * #Positive = TP/P * P = # True Positive = 'correct_matches'

    return thresholds, tn, fp, fn, tp


def _get_threshold_agg_name(aggregation_layer: bool = False, aggregation_method: str = "name_clustering"):
    """Helper function for setting aggregation method name"""
    if aggregation_layer:
        if aggregation_method is None:
            msg = "aggregation_method cannot be None with aggregation_layer enable"
            raise ValueError(msg)
        return aggregation_method
    return "non_aggregated"


def get_threshold_curves_parameters(
    best_candidate_df,
    score_col: str = "nm_score",
    aggregation_layer: bool = False,
    aggregation_method: str = "name_clustering",
    positive_set_col: str = "positive_set",
) -> dict:
    """Get threshold decision curves

    Args:
        best_candidate_df: dataframe with the best candidates
        score_col: which score column to use, default is 'nm_score'. For aggregation use 'agg_score'.
        aggregation_layer: use aggregation layer? default is False.
        aggregation_method: which aggregation method is used? 'name_clustering' or 'mean_score'.
        positive_set_col: name of positive set column in best candidates df. default is 'positive_set'

    Returns:
        dictionary with threshold decision curves
    """
    if positive_set_col not in best_candidate_df.columns:
        msg = f"positive set column {positive_set_col} not in best_candidates df."
        raise ValueError(msg)
    best_positive_df = best_candidate_df[best_candidate_df[positive_set_col]]
    best_negative_df = best_candidate_df[~best_candidate_df[positive_set_col]]
    n_positive_names_to_match = len(best_positive_df)
    name_sets = {"all": best_candidate_df, "positive": best_positive_df, "negative": best_negative_df}

    agg_name = _get_threshold_agg_name(aggregation_layer, aggregation_method)
    name_set_params = {}

    for name_set, df in name_sets.items():
        thresholds, tn, fp, fn, tp = _get_threshold_confusion_matrices(df["correct"], df[score_col])

        name_set_params[name_set] = {
            "thresholds": thresholds,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "n_positive_names_to_match": n_positive_names_to_match,
        }

    return {"threshold_curves": {agg_name: name_set_params}}


def decide_threshold(dataset_scored, aggregation_layer: bool = False):
    """Get threshold decision curves

    Args:
        dataset_scored: dataset from train_test_model(), with valid column.
        aggregation_layer: use aggregation layer? default is False.

    Returns:
        dictionary with threshold decision curves
    """
    if aggregation_layer:
        aggregation_method = "name_clustering"
        aggregator = PandasEntityAggregation(
            score_col="nm_score",
            account_col="account",
            uid_col="uid",
            gt_uid_col="gt_uid",
            name_col="name",
            freq_col="counterparty_account_count_distinct",
            aggregation_method=aggregation_method,
        )
        dataset_scored = aggregator.transform(dataset_scored)
        score_col = "agg_score"
        dataset_scored[score_col] = dataset_scored[score_col].fillna(0)
        dataset_scored[f"{score_col}_rank"] = 1
    else:
        aggregation_method = None
        score_col = "nm_score"

    # Metrics on the best candidate only
    valid_df = dataset_scored[dataset_scored.fold == "valid"]
    valid_best_candidate_df = valid_df[valid_df[f"{score_col}_rank"] == 1]

    # Get threshold curve for emm object
    return get_threshold_curves_parameters(valid_best_candidate_df, score_col, aggregation_layer, aggregation_method)
