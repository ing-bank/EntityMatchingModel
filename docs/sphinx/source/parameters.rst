Parameters
==========

When instantiating an ``EntityMatching`` object one can tune multiple parameters, in particular:

- Which name column to use from the input data, and the id column of the GT,
- The setting of the preprocessing pipeline (defaults should work okay),
- Which indexers to use for the candidate selection, and with which settings,
- To turn on/off the supervised layer, and which input features to use (name-only features, without rank features),
- Whether to use name aggregation (turned off by default).

Below we go through the most important parameters to control the entity matching model.

Indexing
--------

- For the indexer parameters see the comments below.
- Important to set both ``name_col`` and ``entity_id_col`` as entity-matching parameters.
  The ground truth dataset needs both a name column and entity-id column.
  A list of names to match needs only a name column.


.. code-block:: python

  # three example name-pair candidate generators:
  # word-based and character-based cosine similarity, and sorted neighbouring indexing
  indexers = [
      {
          "type": "cosine_similarity",
          "tokenizer": "words",        # word-based cosine similarity
          "ngram": 1,                  # 1-gram tokens only
          "num_candidates": 10,        # max 10 candidates per name-to-match
          "cos_sim_lower_bound": 0.,   # lower bound on cosine similarity
      },
      {
          "type": "cosine_similarity",
          "tokenizer": "characters",   # character-based cosine similarity
          "ngram": 2,                  # 2-gram character tokens only
          "num_candidates": 5,         # max 5 candidates per name-to-match
          "cos_sim_lower_bound": 0.2,  # lower bound on cosine similarity
      },
      {
          "type": "sni",
          "window_length": 3,          # sorted neighbouring indexing window of size 3.
      },
  ]
  em_params = {
      "name_col": "Name",                     # important to set both index and name columns
      "entity_id_col": "Index",
      "indexers": indexers,
      "carry_on_cols": [],                    # names of columns in the GT and names-to-match dataframes passed on by the indexers. GT columns get prefix 'gt_'.
      "supervised_on": False,                 # no initial supervised model to select best candidates right now
      "name_only": True,                      # only consider name information for matching, e.g. not "country" info
      "without_rank_features": False,         # add rank-based features for improved probability of match
      "with_legal_entity_forms_match": True,  # add feature that indicates match of legal entity forms (eg. ltd != co)
      "aggregation_layer": False,
  }
  # initialize the entity matcher
  p = PandasEntityMatching(em_params)
  # prepare the indexers based on the ground truth names: e.g. fit the tfidf matrix of the first indexer.
  p.fit(ground_truth)

  # pandas dataframe with name-pair candidates, made by the indexers. all names have been preprocessed.
  candidates_pd = p.transform(test_names)
  candidates_pd.head()

In the candidates dataframe, the indexer output scores are called ``score_0, score_1, etc`` by default.

Supervised Layer
----------------

The classifier can be trained to give a string similarity score or a probability of match.
Both types of score are useful, in particular when there are many good-looking matches
to choose between.

- With ``name_only=True`` the entity-matcher only consider name information
  for matching. When set to false, it also considers country information, set with ``country_col``.
- The optional ``extra_features`` is a list of extra columns (and optionally function to process them) between GT and names-to-match that
  are used for feature calculation (GT==ntm).
  See class ``PandasFeatureExtractor`` for more details and also ``carry_on_cols`` indexer option above.)
  With ``name_only=False`` internally ``extra_features=['country']``.
- The use of rank features can be turned off with the EMM parameter ``without_rank_features=True``.
- The use of legal entity form matching can be turned on with the EMM parameter ``with_legal_entity_forms_match=True``.
- The flag ``create_negative_sample_fraction=0.5`` controls the fraction of positive names
  (those known to have a match) artificially converted into negative names (without a proper match).
- The flag ``drop_duplicate_candidates=True`` drop any duplicate training candidates and keep just one,
  if available keep the correct match. Recommended for string-similarity models, eg. with
  without_rank_features=True. default is False.

.. code-block:: python

  # create and fit a supervised model for the PandasEntityMatching object to pick the best match (this takes a while)
  # input is "positive" names column 'Name' that are all supposed to match to the ground truth,
  # and an id column 'Index' to check with candidate name-pairs are matching and which not.
  # A fraction of these names, here 0.50, can be artificially turned into negative names (no match to the ground truth).
  # (internally candidate name-pairs are automatically generated, which are input for the classification)
  # this call sets supervised_on=True.
  p.fit_classifier(train_positive_names_to_match=train_names, create_negative_sample_fraction=0.5,
                   drop_duplicate_candidates=True, extra_features=None)

  # generated name-pair candidates, now with classifier-based probability of match.
  # Input is the names' column 'Name'. In the output candidates df, see extra column 'nm_score'.
  candidates_scored_pd = p.transform(test_names)
  candidates_pd.head()

In the candidates dataframe, the classification output score is called ``nm_score`` by default.

The trained sklearn model is accessible under ``p.supervised_models['nm_score']``.

Instead of calling ``p.fit_classifier()``, an independently trained sklearn model can be provided
as well through ``p.add_supervised_model(skl_model)``.

Aggregation Layer
-----------------

Optionally, the EMM package can also be used to match a group of company names that
belong together, to a common company name in the ground truth.
For example, all different names used to address an external bank account.
This step aggregates the name-matching scores from the supervised layer into a
single match.

It is important to provide:

- ``account_col`` specifies which names belong together in one group. Default value is ``account``.
- ``freq_col`` specifies the weight of each name in a group. For example the frequency
  of how often a name has been encountered.
- The score column to aggregate is set with ``score_col``. By default set to the name-matching score ``nm_score``,
  e.g. but can also be a cosine similarity score such as ``score_0``.

.. code-block:: python

  # add aggregation layer to the EMM object
  # this sets aggregation_layer=True.
  p.fit(gt)
  p.add_aggregation_layer(
      score_col="nm_score",
      aggregation_method="max_frequency_nm_score",
      account_col="account",
      freq_col="counterparty_account_count_distinct",
  )
  candidates_pd = p.transform(account_data)
  candidates_pd.head()

The aggregate output score is called ``agg_score`` by default.
