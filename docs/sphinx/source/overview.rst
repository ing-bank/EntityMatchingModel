Overview
========

Why we built this
-----------------

The Entity Matching Model (EMM) package is an efficient library for company name matching at scale.
Our solution is designed to handle large datasets with millions of names.
EMM has both a Pandas and Spark implementation, giving identical name-matching results.

The problem at hand is to match names between two datasets, both possibly very large.
There is the ground truth (GT) list of names, often a carefully curated set, to which other names are matched.
Names from an external data set, possibly of low quality, are matched to the GT. For each name to match,
we calculate the similarity to all names from the GT, and then select the best
matches.

Name matching is a quadratic problem, one that easily becomes computationally intensive for large datasets.
The longer the GT, for example of 100k names or more,
the more good-looking false-positive candidates are found per name to match.
For example, take a GT set with 10M names and an external dataset with 30M unique names.
Comparing 10k name-pairs per second, matching all names to the full GT would take almost 1000 years!
We use the EMM package to do name matching at scale. In our cluster (~1000 nodes),
this example name-matching problem can be performed in about an hour.


How to use our package
----------------------

The EMM package solves two problems in order to
perform efficient company-name matching at scale, namely:

1. selecting all relevant name-pair candidates quickly enough, and
2. from those pairs accurately selecting the correct matches using clever features.

For both steps we have developed fast, intelligent, and tailored solutions.
The selection of all relevant name-pairs is called the "indexing" step, consisting of a number of unsupervised indexing methods
that select all promising name-pair candidates.
The second stage is called the supervised layer, and is done using a classification
model that is trained to select the matching name-pairs.
This is particularly relevant when there are many good-looking matches to choose between.

EMM can perform company name matching with or without the supervised layer present.

A name-pair classifier can be trained to give a string similarity score
or a probability of match. For this a training dataset of so-called positive names needs
to be provided by the user.
Positive names are alternative company names (eg. with missing words, misspellings, etc)
known to match to the ground truth.

If no positive names are available, these can be created artificially with EMM by adding noise to
the list of ground truth names. (The noise is not very realistic so this is is a suboptimal solution.)
Alternatively, when a list of names to match is available a user can manually label a
subset of name-pairs that come out of the indexing step as
correct and incorrect matches, and then simply train the supervised on those.
(EMM does not provide a labelling tool, but there are many around.)

Pandas and Spark support
------------------------

The EMM library contains both a Pandas and Spark implementation.

The Pandas and Spark version of ``EntityMatching`` both have almost the same API.
The Pandas version is much faster and meant for smaller data though.
There is no initialization overhead and it has much fewer dependencies (no spark).


