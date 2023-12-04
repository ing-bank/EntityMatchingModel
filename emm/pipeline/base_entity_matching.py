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

from abc import ABC
from typing import Any, Mapping

import numpy as np

from emm.base.pipeline import Pipeline
from emm.helper.io import IOFunc
from emm.helper.util import get_model_title, indexers_set_values, rename_columns
from emm.indexing.base_indexer import BaseIndexer
from emm.loggers.logger import logger
from emm.parameters import DEFAULT_INDEXER_PARAMS, MODEL_PARAMS
from emm.supervised_model.base_supervised_model import create_new_model_pipeline
from emm.version import __version__


class BaseEntityMatching(Pipeline, ABC):
    """Base implementation of EntityMatching"""

    def __init__(self, parameters: dict | None = None, supervised_models: dict[str, Any] | None = None) -> None:
        """Base implementation of EntityMatching

        Args:
            parameters: a dictionary with algorithm parameters, missing values would be filled in by default values from `emm.config`
            supervised_models: optional dictionary of pretrained models.
        """
        self.parameters = MODEL_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)

        self.supervised_models = supervised_models
        self.ground_truth_df = None
        self.n_ground_truth = -1

        # Check that each indexer is dict with indexer settings or is of type BaseIndexer.
        self._check_indexer_types()
        super().__init__()
        logger.debug(f"Parameters used by entity-matching: {self.parameters}")

    @staticmethod
    def version():
        return __version__

    def _check_indexer_types(self):
        """Each indexer should be a dict with indexer settings or be of type BaseIndexer"""
        indexers_definition = self.parameters.get("indexers", [])
        assert isinstance(indexers_definition, (list, tuple))
        for indexer_def in indexers_definition:
            if not isinstance(indexer_def, (dict, BaseIndexer)):
                msg = "Each indexer should be a dict with indexer settings or be of type BaseIndexer."
                raise TypeError(msg)

    def _initialize_supervised_models(self):
        params = self.parameters
        if params["supervised_on"] is False:
            return

        assert (params["supervised_on"] is True) or (isinstance(params["supervised_on"], list))

        if (self.supervised_models is not None) and len(self.supervised_models) > 0:
            # enable all supervised models that have been added
            # 'X' is reserved for untrained models, which should not be enabled.
            if params["supervised_on"] is True:
                for model_col, model_dict in self.supervised_models.items():
                    if model_col == "X":
                        continue
                    model_dict["enable"] = True
            else:
                for model_col, model_dict in self.supervised_models.items():
                    model_dict["enable"] = bool(model_col in params["supervised_on"] and model_col != "X")
        else:
            # try adding a supervised model
            self._add_supervised_model()

    def _disable_multiprocessing_all_models(self):
        # Disable multiprocessing in Spark, because it is using pandarallel and it is copying the memory for each process.
        # Remark: multithreading will suffer from the Python GIL, so let's use Spark for the distribution.
        for model_col, model_dict in self.supervised_models.items():
            model = model_dict["model"]
            for step_name, step in model.steps:
                if hasattr(step, "n_jobs"):
                    step.n_jobs = 1  # disable multiprocessing in spark
                    logger.debug(f"Disable multiprocessing in Spark for {model_col}/{step_name}/n_jobs=1")

    def _add_supervised_model(self, model_key="nm_score", overwrite=True):
        params = self.parameters
        if params["supervised_on"] is False:
            return

        # basic key checks
        if model_key == "X":
            msg = 'Model key "X" reserved for untrained models. Please provide a different model name.'
            raise KeyError(msg)
        if isinstance(self.supervised_models, dict) and model_key in self.supervised_models:
            if not overwrite:
                msg = f'Model key "{model_key}" already in use. Provide a different model name.'
                raise KeyError(msg)
            logger.info(f'Model key "{model_key}" already in use, will be overwritten.')

        if self.parameters.get("supervised_model_object") is not None:
            if self.supervised_models is None:
                self.supervised_models = {}
            self.supervised_models[model_key] = {
                "description": "model from supervised_model_object",
                "model": self.parameters["supervised_model_object"],
                "enable": True,
            }
        elif self.parameters.get("supervised_model_filename") is not None:
            if self.supervised_models is None:
                self.supervised_models = {}
            load_func = IOFunc().reader
            model = load_func(self.parameters["supervised_model_filename"], self.parameters["supervised_model_dir"])
            self.supervised_models[model_key] = {
                "description": f"model loaded from {self.parameters['supervised_model_dir']}/{self.parameters['supervised_model_filename']}",
                "model": model,
                "enable": True,
            }
        elif params.get("return_sm_features", False):
            # untrained sm pipeline. Only used for feature generation.
            # 'X' is reserved for untrained models, which should not be enabled.
            if self.supervised_models is None:
                self.supervised_models = {}
            if "X" in self.supervised_models and not overwrite:
                logger.warning('Model key "X" already in use (untrained supervised model). Not overwriting.')
            else:
                if "X" in self.supervised_models and overwrite:
                    logger.info('Model key "X" already in use, will be overwritten.')
                self.supervised_models["X"] = {
                    "description": "calculate sm features only",
                    "model": create_new_model_pipeline(),
                    "enable": False,  # Note: full model is not enabled, only for calc features
                }

    def _normalize_column_names(self, df):
        return rename_columns(
            df,
            [
                (self.parameters["entity_id_col"], "entity_id"),
                (self.parameters["uid_col"], "uid"),
                (self.parameters["name_col"], "name"),
                (self.parameters["country_col"], "country"),
                (self.parameters["account_col"], "account"),
                (self.parameters["freq_col"], "counterparty_account_count_distinct"),
            ],
        )

    def _check_relevant_columns_present(self, df, ground_truth=False):
        """Check all required columns are present given emm setup

        Given the current parameter settings. Works for both pandas and spark.

        Args:
            df: input dataframe
            ground_truth: set true if input df is the ground truth
        """
        columns = [self.parameters["name_col"]]
        normalized_columns = ["name"]
        if ground_truth:
            columns += [self.parameters["entity_id_col"]]
            normalized_columns += ["entity_id"]
        if not self.parameters["name_only"]:
            columns += [self.parameters["country_col"]]
            normalized_columns += ["country"]
        if self.parameters["aggregation_layer"] and not ground_truth:
            columns += [self.parameters["account_col"], self.parameters["freq_col"]]
            normalized_columns += ["account", "counterparty_account_count_distinct"]

        for col, norm_col in zip(columns, normalized_columns):
            if all(c not in df.columns for c in [col, norm_col]):
                msg = f'Column "{col}" (and internal column "{norm_col}") not present in input dataframe.'
                raise ValueError(msg)

    @staticmethod
    def get_threshold_agg_name(aggregation_layer=False, aggregation_method="name_clustering"):
        """Helper function for getting/setting aggregation method name

        Args:
            aggregation_layer: use aggregation layer? default is False.
            aggregation_method: which aggregation method is used? 'name_clustering' or 'mean_score'.

        Returns:
            'non_aggregated' if aggregation_layer is False else aggregation_method.
        """
        if aggregation_layer:
            if aggregation_method is None:
                msg = "aggregation_method cannot be None with aggregation_layer enable"
                raise ValueError(msg)
            return aggregation_method
        return "non_aggregated"

    def calc_threshold(self, agg_name, type_name, metric_name, min_value, threshold_parameters=None):
        """Calculate threshold score for given metric with minimum metric value

        Args:
            agg_name: name of aggregation method, see get_threshold_agg_name().
            type_name: "positive" or "negative" names or "all" (positive and negative).
            metric_name: name of metric, eg. "precision", "TNR", "TPR", "fullrecall", "predicted_matches_rate".
            min_value: minimum value for the metric.
            threshold_parameters: dict with threshold curves. use threshold.get_threshold_curves_parameters()
                                     if not provided, try to get this from self.parameters.

        Returns:
            threshold score
        """
        if threshold_parameters is None:
            threshold_parameters = self.parameters
        if "threshold_curves" not in threshold_parameters:
            msg = 'Key "threshold_curves" not found in provided parameters.'
            raise KeyError(msg)
        base = threshold_parameters["threshold_curves"][agg_name][type_name]

        thresholds = base["thresholds"]
        if metric_name in base:
            values = base[metric_name]
        elif metric_name == "precision":
            values = base["TP"] / (base["TP"] + base["FP"])
        elif metric_name == "TNR":
            values = base["TN"] / (base["TN"] + base["FP"])
        elif metric_name == "TPR":
            values = base["TP"] / (base["TP"] + base["FN"])
        elif metric_name == "fullrecall":
            values = base["TP"] / base["n_positive_names_to_match"]
        elif metric_name == "predicted_matches_rate":
            values = (base["FP"] + base["TP"]) / (base["TN"] + base["FP"] + base["FN"] + base["TP"])
        else:
            msg = f"Unknown metric: {metric_name}"
            raise ValueError(msg)

        indexes_below_threshold = np.argwhere(values >= min_value).flatten()

        if len(indexes_below_threshold) > 0:
            threshold = thresholds[indexes_below_threshold[-1]]
            value = values[indexes_below_threshold[-1]]
        else:
            logger.warning(
                f"threshold: {agg_name}.{type_name}.{metric_name} >= {min_value} ==> WARNING there is no such threshold, we fall back on the maximum"
            )
            # Let's query threshold in the same way, but this time for the maximum value
            min_value = max(values)
            indexes_below_threshold = np.argwhere(values >= min_value).flatten()
            threshold = thresholds[indexes_below_threshold[-1]]
            value = values[indexes_below_threshold[-1]]

        logger.info(
            f"threshold: {agg_name}.{type_name}.{metric_name} >= {min_value} ==> t > {threshold}  ({type_name}.{metric_name} = {value})"
        )

        return threshold

    def set_threshold(self, type_name, metric_name, min_value, agg_name=None, threshold_parameters=None):
        """Calculate and set threshold score for given metric with minimum metric value

        Args:
            type_name: "positive" names or "all" (positive and negative).
            metric_name: name of metric, eg. "precision", "TNR", "TPR", "fullrecall", "predicted_matches_rate".
            min_value: minimum value for the metric.
            agg_name: name of aggregation method, if None take from self.get_threshold_agg_name().
            threshold_parameters: dict with threshold curves. use threshold.get_threshold_curves_parameters()
                                     if not provided, try to get this from self.parameters.
        """
        # If agg_name is not given, let's use the current EM paramters to get the agg_name key
        if agg_name is None:
            agg_name = self.get_threshold_agg_name(
                self.parameters.get("aggregation_layer", False), self.parameters.get("aggregation_method")
            )

        threshold = self.calc_threshold(agg_name, type_name, metric_name, min_value, threshold_parameters)
        self.parameters["threshold"] = threshold

    def _indexers_set_default_values(self, indexers: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return indexers_set_values(DEFAULT_INDEXER_PARAMS, indexers)

    def get_model_title(self):
        """Construct model title from parameters settings

        Extract experimental title of model based on model's settings: indexer, sm, aggregation.
        E.g. can be used for storage.
        """
        return get_model_title(self.parameters)
