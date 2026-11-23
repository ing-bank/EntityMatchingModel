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

from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from emm.aggregation.base_entity_aggregation import BaseEntityAggregation
from emm.aggregation.polars_entity_aggregation import PolarsEntityAggregation
from emm.data.polars_prepare_name_pairs import prepare_name_pairs_polars
from emm.helper.io import IOFunc
from emm.helper.sklearn_pipeline import SklearnPipelineWrapper
from emm.indexing.base_indexer import BaseIndexer
from emm.indexing.polars_candidate_selection import PolarsCandidateSelectionTransformer
from emm.indexing.polars_cos_sim_matcher import PolarsCosSimIndexer
from emm.indexing.polars_naive_indexer import PolarsNaiveIndexer
from emm.indexing.polars_sni import PolarsSortedNeighbourhoodIndexer
from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.parameters import DEFAULT_CARRY_ON_COLS, MODEL_PARAMS
from emm.pipeline.base_entity_matching import BaseEntityMatching
from emm.preprocessing.base_name_preprocessor import AbstractPreprocessor
from emm.preprocessing.polars_preprocessor import PolarsPreprocessor
from emm.supervised_model.polars_supervised_model import PolarsSupervisedLayerTransformer
from emm.supervised_model.polars_train_model import train_model_polars

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


class PolarsEntityMatching(BaseEntityMatching):
    """Entity matching pipeline using Polars DataFrames throughout."""

    def __init__(  # noqa: PLR0917
        self,
        parameters: dict[str, Any] | None = None,
        supervised_models: Mapping[str, Any] | None = None,
        name_col: str | None = None,
        entity_id_col: str | None = None,
        name_only: bool | None = None,
        preprocessor: str | None = None,
        indexers: list | None = None,
        supervised_on: bool | None = None,
        without_rank_features: bool | None = None,
        with_legal_entity_forms_match: bool | None = None,
        return_sm_features: bool | None = None,
        supervised_model_object: Pipeline | None = None,
        aggregation_layer: bool | None = None,
        aggregation_method: Literal["mean_score", "max_frequency_nm_score"] | None = None,
        carry_on_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        function_locals = locals()
        model_parameters = {
            key: function_locals.get(key) for key in MODEL_PARAMS if function_locals.get(key) is not None
        }
        if parameters is None:
            parameters = {}
        parameters.update({**model_parameters, **kwargs})
        super().__init__(parameters=parameters, supervised_models=supervised_models)
        self.model: SklearnPipelineWrapper | None = None
        self.initialize()

    def initialize(self) -> None:
        """Create a pipeline from the current parameters."""
        self.pipeline = self._create_pipeline()

    def _normalize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        mapping = [
            (self.parameters["entity_id_col"], "entity_id"),
            (self.parameters["uid_col"], "uid"),
            (self.parameters["name_col"], "name"),
            (self.parameters["country_col"], "country"),
            (self.parameters["account_col"], "account"),
            (self.parameters["freq_col"], "counterparty_account_count_distinct"),
        ]
        renames = []
        seen = set()
        for source, target in mapping:
            if source not in df.columns:
                continue
            if source in seen:
                copy_name = f"{source}_copy"
                df = df.with_columns(pl.col(source).alias(copy_name))
                source = copy_name
            seen.add(source)
            if source != target:
                renames.append((source, target))
        for source, target in renames:
            assert source in df.columns, f"Column cannot be renamed because '{source}' doesn't exist"
            assert target not in df.columns, f"Column cannot be renamed because '{target}' already exist"
            df = df.rename({source: target})
        return df

    def _create_preprocessor(self) -> Any:
        preprocessor = self.parameters["preprocessor"]
        if isinstance(preprocessor, AbstractPreprocessor):
            return preprocessor
        return PolarsPreprocessor(preprocess_pipeline=preprocessor, spark_session=self.parameters.get("spark_session"))

    def _create_indexers(self) -> list[BaseIndexer]:
        params = self.parameters
        classes = {
            "cosine_similarity": PolarsCosSimIndexer,
            "sni": PolarsSortedNeighbourhoodIndexer,
            "naive": PolarsNaiveIndexer,
        }
        defaults = {
            "cosine_similarity": {
                "input_col": "preprocessed",
                "spark_session": params.get("spark_session"),
                "n_jobs": params.get("n_jobs", -1),
            },
            "sni": {"input_col": "preprocessed"},
            "naive": {},
        }
        definitions = params.get("indexers")
        if definitions is None:
            definitions = []
            for kind in ("sni", "cosine_similarity"):
                if params[kind]:
                    items = params[kind] if isinstance(params[kind], list) else [{}]
                    definitions.extend({"type": kind, **item} for item in items)
        indexers = []
        for definition in self._indexers_set_default_values(definitions):
            if isinstance(definition, dict):
                settings = definition.copy()
                kind = settings.pop("type")
                indexers.append(classes[kind](**{**defaults[kind], **settings}))
            elif isinstance(definition, BaseIndexer):
                indexers.append(definition)
        return indexers

    def _create_candidate_selection_step(self, indexers: list[BaseIndexer] | None = None) -> Any:
        if indexers is None:
            indexers = self._create_indexers()
        if not indexers:
            return None
        return PolarsCandidateSelectionTransformer(
            indexers=indexers,
            uid_col="entity_id",
            carry_on_cols=list(set(DEFAULT_CARRY_ON_COLS + self.parameters.get("carry_on_cols", []))),
            with_no_matches=self.parameters.get("with_no_matches", True),
        )

    def _create_supervised_step(self) -> PolarsSupervisedLayerTransformer | None:
        if self.parameters["supervised_on"] is not False:
            self._initialize_supervised_models()
            return PolarsSupervisedLayerTransformer(
                self.supervised_models, return_features=self.parameters["return_sm_features"]
            )
        return None

    def _create_aggregation_step(self) -> BaseEntityAggregation | None:
        aggregation = self.parameters.get("aggregation_layer", False)
        if isinstance(aggregation, BaseEntityAggregation):
            return aggregation
        if aggregation:
            return PolarsEntityAggregation(
                score_col="nm_score" if self.parameters["supervised_on"] else "score_0",
                freq_col="counterparty_account_count_distinct",
                aggregation_method=self.parameters["aggregation_method"],
                blacklist=self.parameters.get("aggregation_blacklist", []),
            )
        return None

    def _create_pipeline(self) -> SklearnPipelineWrapper:
        steps = [
            ("preprocess", self._create_preprocessor()),
            ("candidate_selection", self._create_candidate_selection_step()),
            ("supervised", self._create_supervised_step()),
            ("aggregation", self._create_aggregation_step()),
        ]
        return SklearnPipelineWrapper([(name, step) for name, step in steps if step is not None])

    def fit(self, ground_truth_df: pl.DataFrame, copy_ground_truth: bool = False) -> PolarsEntityMatching:
        """Fit indexers on ground-truth names."""
        if not isinstance(ground_truth_df, pl.DataFrame):
            msg = f"expected a Polars DataFrame, got {type(ground_truth_df).__name__}"
            raise TypeError(msg)
        with Timer("PolarsEntityMatching.fit") as timer:
            self._check_relevant_columns_present(ground_truth_df, ground_truth=True)
            ground_truth_df = self._normalize_column_names(ground_truth_df)
            if copy_ground_truth:
                self.ground_truth_df = ground_truth_df.clone()
            self.model = self.pipeline.fit(ground_truth_df)
            self.n_ground_truth = ground_truth_df.height
            timer.log_param("n", self.n_ground_truth)
        return self

    def transform(self, names_df: pl.DataFrame | pl.Series, top_n: int = -1) -> pl.DataFrame:
        """Match names against the fitted ground truth."""
        if self.model is None:
            msg = "indexing pipeline has not been trained. Did you already fit()?"
            raise TypeError(msg)
        if isinstance(names_df, pl.Series):
            names_df = names_df.to_frame()
        if not isinstance(names_df, pl.DataFrame):
            msg = f"expected a Polars DataFrame or Series, got {type(names_df).__name__}"
            raise TypeError(msg)
        with Timer("PolarsEntityMatching.transform") as timer:
            self._check_relevant_columns_present(names_df)
            names_df = self._normalize_column_names(names_df)
            columns = ["name"]
            columns.extend(col for col in ("entity_id", "country") if col in names_df.columns)
            if self.parameters["aggregation_layer"]:
                columns.extend(["account", "counterparty_account_count_distinct"])
            columns.extend(
                col
                for col in self.parameters.get("carry_on_cols", [])
                if col in names_df.columns and col not in columns
            )
            logger.info("Matching %d records against ground-truth with size %d.", names_df.height, self.n_ground_truth)
            result = self.model.transform(names_df.select(columns))
            if isinstance(top_n, int) and top_n > 0 and "best_rank" in result.columns:
                result = result.filter((pl.col("best_rank") <= top_n) & pl.col("gt_uid").is_not_null())
            timer.log_param("cands", result.height)
            return result

    def create_training_name_pairs(
        self,
        train_positive_names_to_match: pl.DataFrame,
        create_negative_sample_fraction: float = 0,
        n_train_ids: int = -1,
        random_seed: int = 42,
        drop_duplicate_candidates: bool | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Create labeled candidate pairs for classifier training."""
        if self.model is None:
            msg = "indexer pipeline not yet fit and train_gt not provided to do so."
            raise TypeError(msg)
        if n_train_ids > 0:
            id_col = self.parameters["entity_id_col"]
            ids = sorted(train_positive_names_to_match[id_col].unique().to_list())
            if len(ids) > n_train_ids:
                ids = np.random.default_rng(random_seed).choice(ids, n_train_ids, replace=False).tolist()
                train_positive_names_to_match = train_positive_names_to_match.filter(pl.col(id_col).is_in(ids))
        fraction = min(create_negative_sample_fraction, 1)
        if fraction > 0:
            self.increase_window_by_one_step()
        try:
            candidates = self.transform(train_positive_names_to_match)
        finally:
            if fraction > 0:
                self.decrease_window_by_one_step()
        candidates = candidates.drop("name", "gt_name")
        if "score" in candidates.columns:
            candidates = candidates.rename({"score": "score_0"})
        return prepare_name_pairs_polars(
            candidates,
            drop_duplicate_candidates=self.parameters.get("drop_duplicate_candidates", False)
            if drop_duplicate_candidates is None
            else drop_duplicate_candidates,
            create_negative_sample_fraction=fraction,
            positive_set_col=self.parameters.get("positive_set_col", "positive_set"),
            correct_col=self.parameters.get("correct_col", "correct"),
            uid_col="uid",
            gt_uid_col=self.parameters.get("gt_uid_col", "gt_uid"),
            preprocessed_col=self.parameters.get("preprocessed_col", "preprocessed"),
            gt_preprocessed_col=self.parameters.get("gt_preprocessed_col", "gt_preprocessed"),
            random_seed=random_seed,
            **kwargs,
        )

    def fit_classifier(  # noqa: PLR0917
        self,
        train_positive_names_to_match: pl.DataFrame | None = None,
        train_name_pairs: pl.DataFrame | None = None,
        create_negative_sample_fraction: float = 0,
        n_train_ids: int = -1,
        random_seed: int = 42,
        train_gt: pl.DataFrame | None = None,
        store_key: str = "nm_score",
        train_function: Callable = train_model_polars,
        score_columns: list[str] | None = None,
        drop_duplicate_candidates: bool | None = None,
        extra_features: list[str | tuple[str, Callable]] | None = None,
        **fit_kws: Any,
    ) -> PolarsEntityMatching:
        """Train and add a supervised model using Polars training pairs."""
        if not callable(train_function):
            msg = f'training function "{train_function}" is not callable.'
            raise TypeError(msg)
        if self.model is None and train_gt is None:
            msg = "indexer pipeline not yet fit and train_gt not provided to do so."
            raise TypeError(msg)
        if train_positive_names_to_match is None and train_name_pairs is None:
            msg = "Must provide either positive training names or training candidate name-pairs."
            raise TypeError(msg)
        if train_gt is not None:
            self.parameters["supervised_on"] = False
            self.pipeline = self._create_pipeline()
            self.fit(train_gt)
        old_steps = self.model.steps.copy()
        aggregation = next((step for step in old_steps if step[0] == "aggregation"), None)
        self.model.steps = [step for step in old_steps if step[0] not in ("supervised", "aggregation")]
        try:
            if train_positive_names_to_match is not None:
                train_name_pairs = self.create_training_name_pairs(
                    train_positive_names_to_match,
                    create_negative_sample_fraction,
                    n_train_ids=n_train_ids,
                    random_seed=random_seed,
                    drop_duplicate_candidates=drop_duplicate_candidates,
                )
            model = train_function(
                train_name_pairs,
                without_rank_features=self.parameters.get("without_rank_features", False),
                name_only=self.parameters.get("name_only", False),
                positive_set_col=self.parameters.get("positive_set_col", "positive_set"),
                score_columns=score_columns,
                with_legal_entity_forms_match=self.parameters.get("with_legal_entity_forms_match", False),
                extra_features=extra_features,
                **fit_kws,
            )
        except Exception:
            self.model.steps = old_steps
            raise
        if isinstance(self.supervised_models, dict):
            self.supervised_models.pop("X", None)
        self.parameters["supervised_on"] = True
        self.parameters["supervised_model_object"] = model
        self._add_supervised_model(model_key=store_key, overwrite=True)
        self.model.steps.append(("supervised", self._create_supervised_step()))
        if aggregation is not None:
            aggregation[1].score_col = store_key
            self.model.steps.append(aggregation)
        return self

    def test_classifier(self, test_names_to_match: pl.DataFrame, test_gt: pl.DataFrame | None = None) -> None:
        """Log the AUC of the supervised model on supplied names."""
        if self.model is None or self.parameters["supervised_on"] is False:
            msg = "No supervised model available."
            raise TypeError(msg)
        if test_gt is None:
            test_gt = self.ground_truth_df
        if test_gt is None:
            msg = "No ground truth names available."
            raise TypeError(msg)
        candidates = self.transform(test_names_to_match).filter(pl.col("best_match"))
        results = (
            test_names_to_match.with_row_index("__test_uid")
            .join(
                candidates.select("uid", "gt_entity_id", "nm_score", "score_0"),
                left_on="__test_uid",
                right_on="uid",
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.col("nm_score").fill_null(-1), pl.col("score_0").fill_null(-1))
        )
        gt_id_col = "id" if "id" in test_gt.columns else "entity_id"
        in_gt = pl.col("id").is_in(test_gt[gt_id_col].to_list())
        correct = ((in_gt & (pl.col("id") == pl.col("gt_entity_id"))) | (~in_gt & pl.col("id").is_null())).fill_null(
            False
        )
        labels = results.select(correct)["id"].to_numpy()
        logger.info("AUC of the supervised model: %.4f", roc_auc_score(labels, results["nm_score"].to_numpy()))

    def add_supervised_model(  # noqa: PLR0917
        self,
        path: str | None = None,
        model: Pipeline | None = None,
        name_only: bool = True,
        store_key: str = "nm_score",
        overwrite: bool = True,
        return_features: bool | None = None,
    ) -> None:
        """Add a trained supervised model to a fitted pipeline."""
        if path is None and model is None:
            msg = "Need to provided either path to trained model or model itself."
            raise TypeError(msg)
        if self.model is None:
            msg = "indexer pipeline not yet fit. Cannot add supervised layer."
            raise TypeError(msg)
        aggregation = next((step for step in self.model.steps if step[0] == "aggregation"), None)
        self.parameters.update(
            supervised_on=True, supervised_model_filename=path, supervised_model_object=model, name_only=name_only
        )
        self._add_supervised_model(model_key=store_key, overwrite=overwrite)
        if return_features is not None:
            self.parameters["return_sm_features"] = return_features
        self.model.steps = [step for step in self.model.steps if step[0] not in ("supervised", "aggregation")]
        self.model.steps.append(("supervised", self._create_supervised_step()))
        if aggregation is not None:
            aggregation[1].score_col = store_key
            self.model.steps.append(aggregation)

    def add_aggregation_layer(
        self,
        account_col: str | None = None,
        freq_col: str | None = None,
        aggregation_method: str | None = None,
        blacklist: list | None = None,
        aggregation_layer: BaseEntityAggregation | None = None,
    ) -> None:
        """Add or replace the aggregation layer."""
        if self.model is None:
            msg = "indexer pipeline not yet fit."
            raise TypeError(msg)
        if aggregation_layer is not None and not isinstance(aggregation_layer, BaseEntityAggregation):
            msg = "aggregation_layer does not have type BaseEntityAggregation"
            raise TypeError(msg)
        if aggregation_layer is not None:
            self.parameters["aggregation_layer"] = aggregation_layer
        else:
            self.parameters["aggregation_layer"] = True
            for key, value in (
                ("account_col", account_col),
                ("freq_col", freq_col),
                ("aggregation_method", aggregation_method),
                ("aggregation_blacklist", blacklist),
            ):
                if value is not None:
                    self.parameters[key] = value
        self.model.steps = [step for step in self.model.steps if step[0] != "aggregation"]
        self.model.steps.append(("aggregation", self._create_aggregation_step()))
        selector = self.model.named_steps.get("candidate_selection")
        if selector is not None:
            for column in ("account", "counterparty_account_count_distinct"):
                if column not in selector.carry_on_cols:
                    selector.carry_on_cols.append(column)

    def increase_window_by_one_step(self) -> None:
        """Increase each fitted indexer's search window."""
        if self.model is not None and "candidate_selection" in self.model.named_steps:
            self.model.named_steps["candidate_selection"].increase_window_by_one_step()

    def decrease_window_by_one_step(self) -> None:
        """Decrease each fitted indexer's search window."""
        if self.model is not None and "candidate_selection" in self.model.named_steps:
            self.model.named_steps["candidate_selection"].decrease_window_by_one_step()

    def set_return_sm_features(self, return_features: bool = True) -> None:
        """Toggle the return of supervised model features."""
        self.parameters["return_sm_features"] = return_features
        if self.model is not None and "supervised" in self.model.named_steps:
            self.model.named_steps["supervised"].return_features = return_features

    def save(self, emo_path: str, dump_func: Callable = IOFunc().writer) -> None:
        """Serialize the fitted model."""
        if self.model is None:
            msg = "indexer pipeline not yet fit. Nothing useful to store."
            raise TypeError(msg)
        spark_session = self.parameters.pop("spark_session", None)
        ground_truth = self.ground_truth_df
        self.ground_truth_df = None
        selector = self.model.named_steps.get("candidate_selection")
        if selector is not None:
            selector._reset_sni_ground_truth()
        try:
            dump_func(self, emo_path)
        finally:
            if spark_session is not None:
                self.parameters["spark_session"] = spark_session
            self.ground_truth_df = ground_truth
            if selector is not None:
                selector._set_sni_ground_truth()

    @staticmethod
    def load(
        emo_path: str,
        load_func: Callable = IOFunc().reader,
        override_parameters: Mapping[str, Any] | None = None,
        name_col: str | None = None,
        entity_id_col: str | None = None,
        **kwargs: Any,
    ) -> PolarsEntityMatching:
        """Load a saved model and optionally override its parameters."""
        overrides = dict(override_parameters or {})
        function_locals = locals()
        overrides.update({key: function_locals[key] for key in MODEL_PARAMS if function_locals.get(key) is not None})
        overrides.update(kwargs)
        model = load_func(emo_path)
        selector = model.model.named_steps.get("candidate_selection")
        if selector is not None:
            selector._set_sni_ground_truth()
        model.parameters.update(overrides)
        return model
