Fit and Transform
=================

When using an ``EntityMatching`` model:

.. code-block:: python

  from emm import PandasEntityMatching
  model = PandasEntityMatching()


there are three main functions:

- ``model.fit()``
- ``model.fit_classifier()``
- ``model.transform()``




``model.fit(gt)``
~~~~~~~~~~~~~~~~~

This function is applied to the set of GT names.
It fits the candidate selection module, in particular it creates the TFIDF matrices of the cosine similarity indexers.
In addition, it fits the ``PandasFeatureExtractor`` step of the supervised model, which creates a vocabulary common words
from the GT.


``model.transform(names)``
~~~~~~~~~~~~~~~~~~~~~~~~~~


This function calls the preprocessor and candidate selection to generate name-pair candidates.
If a trained supervised model is present, each name pair gets a name-matching score under column ``nm_score``.


``model.fit_classifier(positive_names)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that fitting consists of two parts: of the indexers and the supervised model.
This function call fits the supervised model.
As input it needs so-called positive names.

Note there are two types of names to match:

1. Positive name: The name-to-match belongs to a name in the ground truth.
    - Positive correct: matched to the right name.
    - Positive incorrect: matched to the incorrect name.
2. Negative name: The name should NOT be matched to the ground truth.

We would like our model to give a calibrated probability that a name is a match or not.
For this, one needs to have a fraction of negative names during training.
Realize that any trained supervised model is giving a name-matching score based on assumed ratio of positive/negative names.
In reality we don’t know the correct negative fraction! And the correct value may be very big.

The ``fit_classifier()`` has the option ``create_negative_sample_fraction``,
which creates negative names from a fraction of the positive input names.

It is important to realize that the supervised model is tightly linked to (length of) ground truth.
The same supervised model should not be used of different GTs: a different GT ideally needs a newly trained
supervised model.

