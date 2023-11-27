Pipeline
========


This example shows how to instantiate Pandas or Spark ``EntityMatching`` objects (using default settings).

.. code-block:: python

  from emm import PandasEntityMatching, SparkEntityMatching
  p = PandasEntityMatching()
  s = SparkEntityMatching()

An ``EntityMatching`` object consists of multiple components.


Four components
---------------

To solve the name matching problem at scale we follow a generic approach.
The  ``EntityMatching`` pipeline consists of two to four components,
where the last two are optional:


- Preprocessor:
	- Cleaning and standardization of input names and their legal entity forms. Here are the relevant objects (using default settings):

    .. code-block:: python

      import emm.preprocessing
      p_pr = emm.preprocessing.PandasPreprocessor()
      s_pr = emm.preprocessing.SparkPreprocessor()

    See the API section for more details on string preprocessing.

- Candidate selection:
	- Generation of name-pair candidates, also known as ``indexing``. Here we care about the running time and catching all relevant potential matches.

    .. code-block:: python

      from emm.indexing import pandas_candidate_selection, spark_candidate_selection
      p_cs = pandas_candidate_selection.PandasCandidateSelectionTransformer(indexers=[])
      s_cs = spark_candidate_selection.SparkCandidateSelectionEstimator(indexers=[])

    Both need as input a list of so-called indexers. More on this below.

- Supervised model (optional):
	- The classification of each name-pair, in order to pick the best name-pair candidate. This is optional but crucial for the accuracy of the model.

    .. code-block:: python

      import emm.supervised_model
      p_sm = emm.supervised_model.PandasSupervisedLayerTransformer(supervised_models={})
      s_sm = emm.supervised_model.SparkSupervisedLayerEstimator(supervised_models={})

    Both need as input a sklearn supervised model. More on this below.

- Aggregation (optional):
	- Optionally, the EMM package can also be used to match a group of company names that belong together, to a company name in the ground truth.
(For example, all different names used to address an external bank account.)
This step makes use of name-matching scores from the supervised layer.
We refer to this as the aggregation step. This step is not needed for standalone name matching.

    .. code-block:: python

      import emm.aggregation
      p_ag = emm.aggregation.PandasEntityAggregation(score_col='nm_score')
      s_ag = emm.aggregation.SparkEntityAggregation(score_col='nm_score')

    See the API section for more details on aggregation.


Candidate selection, the supervised model, and aggregation are discussed in more detail in the following subsections.







Candidate selection
-------------------


The candidate selection step, also known as ``indexing``, generates all relevant, potential
name-pair candidates belonging to a name-to-match.

Specifically we care about the speed and catching all relevant potential matches.

Three indexers are available to in the EMM package to do so.

- Word-based cosine similarity,
- Character 2-gram based cosine similarity, with blocking,
- Sorted neighbourhood indexing.

These are complementary, every indexer is able to detect different types of candidates.
Combining multiple indexers therefore gives boost in recall.
Together, they allow one to balance running time and accuracy of the model.

The three methods are discussed in more detail below.


Cosine similarity
~~~~~~~~~~~~~~~~~

The approach followed here:

- Transform both GT and names-to-match to sparse matrices, using the Spark or Sklearn TFIDFVectorizer.
- Multiply the sparse matrices.
- For each name select top-n best matches (top-n values in each row in matrix multiplication result) that pass a minimum threshold value.
- We allow for both word-based and character-based vectorization.
- Blocking between names is possible, for example based on the first character of each name.

Scikit sparse matrix multiplication is still too slow and requires too much memory.
As a solution, we have developed the much faster ``sparse_dot_topn`` library.

See for details: https://github.com/ing-bank/sparse_dot_topn

Word-based vectorization turns out to be a powerful and fast technique to generate relevant name-pair candidates.
It is fast enough not to require blocking.

But it misses possible typos. For this we need character-based vectorization. This is slower than word-based vectorization,
because the matrices are less sparse.
But by introducing blocking between names it can be sped up significantly.
When using character-based vectorization, by default we use 2-grams and blocking
based on the first character of each name.

.. code-block:: python

  from emm import indexing
  from emm.helper.blocking_functions import first
  p_cossim = indexing.PandasCosSimIndexer(tokenizer='words', ngram=1, num_candidates=10, cos_sim_lower_bound=0.2)
  s_cossim = indexing.SparkCosSimIndexer(tokenizer='characters', ngram=2, blocking_func=first, cos_sim_lower_bound=0.5)

See the API section for more details.

Sorted neighbourhood indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sorted neighbourhood indexing is a fast and simple technique, namely:

- Merge two name lists and sort them alphabetically.
- Pick a fixed odd-sized window around any name-to-match to search for ground-truth names.

Sorted neighbourhood indexing is good for matching names where one (or multiple) word(s) is missing
in one of the names.


.. code-block:: python

  from emm import indexing
  p_sni = indexing.PandasSortedNeighbourhoodIndexer(window_length=5)
  s_sni = indexing.SparkSortedNeighbourhoodIndexer(window_length=3)


(For Pandas we use the implementation from the brilliant ``recordlinkage`` package.)

See the API section for more details.




Supervised model
----------------

The supervised layer is there to pick the best name-pair candidate.
This is crucial for the accuracy of the model.

During training, for each generated name-pair it is known if it is a correct match or not.


Input features
~~~~~~~~~~~~~~

Four types of input features are used:

- String-based features,
    - Edit distance based metrics such as Cosine similarity, Levenshtein or Jaro distance.
- Rank features for a calibrated model,
    - Features to qualify differences between the various name-pair candidates that all belong to the same name-to-match.
- Legal entity form based features,
    - Legal entity forms can be extracted from the business names and compared for an exact, partial, or no match. For this
      the ``cleanco`` package is used. Missing legal entity forms are not matched.
- Extra features:
    - E.g. country comparison, or address, or legal entity form.


More details of these input features below.

Combined there are 41 string-based and rank features in total.
(These features are used as input for the classifier.)


The string-based features quantify the similarity between two strings.
Multiple related edit-distance metrics are used.


- Indexer scores,
- Abbreviation match,
- Length difference,
- Tokens overlap,
- Edit distance,
- Jaro distance,
- Common and rare words features.

We would like our model to give a calibrated probability that a name is a match or not.
For this the rank-based features are useful.
These quantify the differences between the various name-pair candidates that belong to one name-to-match.


- Rank of cosine similarity score,
- Distance to the maximum cosine similarity score,
- Cosine similarity distance between top-2 candidates,
- Cosine similarity distance to next/prev candidate.

The use of rank features can be turned off with the EMM parameter ``without_rank_features=True``.

The use of legal entity form matching can be turned on with the EMM parameter ``with_legal_entity_forms_match=True``.

Extra features are optional, and do not have to be provided.
For example, country information of the company name under study.

The use of extra features can be turned off with the EMM parameter ``name_only=True``.




Sklearn Pipeline
~~~~~~~~~~~~~~~~

The supervised model is a simple scikit learn pipeline that consists of two steps:

- ``PandasFeatureExtractor``: a custom transformer that calculates the input features (described above) of each name pair,
- ``XGBoost``: classifier which is run with near-default settings on each name pair.



The right model for you
~~~~~~~~~~~~~~~~~~~~~~~

Depending on the use-case, the model with or without rank features may be preferred.
When interested in all potentially good matches to a name, the model without rank features is useful:
simply select all candidate pairs with a high similarity score.
This list will likely contain false positives though.
When only interested in matches with high probability, use the model with the rank features and require a high
threshold.
Any names-to-match with multiple candidates will not make it through such a selection.


In practice the best use of both models could therefore be:
use the model without rank features to select any name-pairs with high string similarity.
From those pairs, select the one with the highest model score with rank features
to get the best possible match.







Aggregation
-----------

We may have multiple names-to-match that all belong to the same entity.
For example, a bank account can be addressed by many different names.
These name may have multiple candidates in the ground truth.
How to aggregate the name-matching score and pick the best candidate?

For this the aggregation step is used.

The aggregation is based on the frequency of the various names-to-match and the name-matching score
or each unique name-pair: essentially each name-matching score is weighted by the frequency of occurance.

In more detail:

- Similar names are grouped: Even though some names are not strictly equal, they are close enough to be considered as similar and it would be interesting to aggregate their frequency or scores.
- Frequency is important: If a large number of different people use similar names to address an account, it’s quite likely that the name is the “true” name we should focus on.
- Score also matters: It happens that some people use a very specific name with a very high score. Sometimes a perfect match. And this can’t be by chance.

Note that for normal name-matching the aggregation step is turned off.



