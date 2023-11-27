# Release notes

## Version 1.4.1

- Major updates of documentation for open-sourcing
- Add extra_features option to emm.fit_classifier function
- Add option drop_duplicate_candidates option to prepare_name_pairs_pd function
- Rename SupervisedLayerEstimator as SparkSupervisedLayerEstimator
- Consistent carry_on_cols behavior between pandas and spark indexing classes
- Significant cleanup of cleanup of parameters.py.
- Remove init_spark file and related calls
- Cleanup of util and spark_utils functions
- Remove unused dap related io functions

## Version 1.4.0

- Introduce `Timer` context for logging
- Removed backwards compatibility `unionByName` helper. Spark >= 3.1 required.
- Replaced custom "NON NFKD MAP" with `unidecode`
- Integration test speedup: split-off long-running integration test
- Removed: `verbose`, `compute_missing`, `use_tqdm`, `save_intermediary`, `n_jobs` options removed, `mlflow` dependencies
- Removed: prediction explanations (bloat), unused unsupervised model, "name_clustering" aggregation
- Perf: 5-10x speedup of feature computations
- Perf: `max_frequency_nm_score` and `mean_score` aggregation method short-circuit groups with only one record (2-3x speedup for skewed datasets)
- Tests: added requests retries with backoff for unstable connections

## Version 1.3.14

- Converted RST readme and changelog to Markdown
- Introduced new parameters for force execution and cosine similary threads.

## Version 1.3.5-1.3.13

See git history for changes.

## Version 1.3.4, Jan 2023

- Added helper function to activate mlflow tracking.
- Added spark example to example.py
- Minor updates to documentation.

## Version 1.3.3, Dec 2022

- Added sm feature indicating matches of legal entity forms between names. Turn on with parameter
 `with_legal_entity_forms_match=True`. Example usage in:
    `03-entity-matching-training-pandas-version.ipynb`. For
    code see `calc_features/cleanco_lef_matching.py`.
- Added code for calculating discrimination threshold curves:
    `em.calc_threshold()`. Example usage in:
    `03-entity-matching-training-pandas-version.ipynb`.
- Added example notebook for name aggregation. See:
    `04-entity-matching-aggregation-pandas-version.ipynb`.
