# Entity Matching model

[![Build](https://github.com/ing-bank/EntityMatchingModel/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/ing-bank/EntityMatchingModel/actions)
[![Latest Github release](https://img.shields.io/github/v/release/ing-bank/EntityMatchingModel)](https://github.com/ing-bank/EntityMatchingModel/releases)
[![GitHub release date](https://img.shields.io/github/release-date/ing-bank/EntityMatchingModel)](https://github.com/ing-bank/EntityMatchingModel/releases)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://static.pepy.tech/badge/emm)](https://pepy.tech/project/emm)


Entity Matching Model (EMM) solves the problem of matching company names between two possibly very
large datasets. EMM can match millions against millions of names with a distributed approach.
It uses the well-established candidate selection techniques in string matching,
namely: tfidf vectorization combined with cosine similarity (with significant optimization),
both word-based and character-based, and sorted neighbourhood indexing.
These so-called indexers act complementary for selecting realistic name-pair candidates.
On top of the indexers, EMM has a classifier with optimized string-based, rank-based, and legal-entity
based features to estimate how confident a company name match is.

The classifier can be trained to give a string similarity score or a probability of match.
Both types of score are useful, in particular when there are many good-looking matches to choose between.
Optionally, the EMM package can also be used to match a group of company names that belong together,
to a common company name in the ground truth. For example, all different names used to address an external bank account.
This step aggregates the name-matching scores from the supervised layer into a single match.

The package is modular in design and and works both using both Pandas and Spark. A classifier trained with the former
can be used with the latter and vice versa.

For release history see [GitHub Releases](https://github.com/ing-bank/EntityMatchingModel/releases).

## Notebooks

For detailed examples of the code please see the notebooks under `notebooks/`.

- `01-entity-matching-pandas-version.ipynb`: Using the Pandas version of EMM for name-matching.
- `02-entity-matching-spark-version.ipynb`: Using the Spark version of EMM for name-matching.
- `03-entity-matching-training-pandas-version.ipynb`: Fitting the supervised model and setting a discrimination threshold (Pandas).
- `04-entity-matching-aggregation-pandas-version.ipynb`: Using the aggregation layer and setting a discrimination threshold (Pandas).

## Documentation

For documentation, design, and API see [the documentation](https://entitymatchingmodel.readthedocs.io/en/latest/).


## Check it out

The Entity matching model library requires Python >= 3.7 and is pip friendly. To get started, simply do:

```shell
pip install emm
```

or check out the code from our repository:

```shell
git clone https://github.com/ing-bank/EntityMatchingModel.git
pip install -e EntityMatchingModel/
```

where in this example the code is installed in edit mode (option -e).

Additional dependencies can be installed with, e.g.:

```shell
pip install "emm[spark,dev,test]"
```

You can now use the package in Python with:


```python
import emm
```

**Congratulations, you are now ready to use the Entity Matching model!**

## Quick run

As a quick example, you can do:

```python
from emm import PandasEntityMatching
from emm.data.create_data import create_example_noised_names

# generate example ground-truth names and matching noised names, with typos and missing words.
ground_truth, noised_names = create_example_noised_names(random_seed=42)
train_names, test_names = noised_names[:5000], noised_names[5000:]

# two example name-pair candidate generators: character-based cosine similarity and sorted neighbouring indexing
indexers = [
  {
      'type': 'cosine_similarity',
      'tokenizer': 'characters',   # character-based cosine similarity. alternative: 'words'
      'ngram': 2,                  # 2-character tokens only
      'num_candidates': 5,         # max 5 candidates per name-to-match
      'cos_sim_lower_bound': 0.2,  # lower bound on cosine similarity
  },
  {'type': 'sni', 'window_length': 3}  # sorted neighbouring indexing window of size 3.
]
em_params = {
  'name_only': True,         # only consider name information for matching
  'entity_id_col': 'Index',  # important to set both index and name columns to pick up
  'name_col': 'Name',
  'indexers': indexers,
  'supervised_on': False,    # no supervided model (yet) to select best candidates
  'with_legal_entity_forms_match': True,   # add feature that indicates match of legal entity forms (e.g. ltd != co)
}
# 1. initialize the entity matcher
p = PandasEntityMatching(em_params)

# 2. fitting: prepare the indexers based on the ground truth names, eg. fit the tfidf matrix of the first indexer.
p.fit(ground_truth)

# 3. create and fit a supervised model for the PandasEntityMatching object, to pick the best match (this takes a while)
#    input is "positive" names column 'Name' that are all supposed to match to the ground truth,
#    and an id column 'Index' to check with candidate name-pairs are matching and which not.
#    A fraction of these names may be turned into negative names (= no match to the ground truth).
#    (internally, candidate name-pairs are automatically generated, these are the input to the classification)
p.fit_classifier(train_names, create_negative_sample_fraction=0.5)

# 4. scoring: generate pandas dataframe of all name-pair candidates.
#    The classifier-based probability of match is provided in the column 'nm_score'.
#    Note: can also call p.transform() without training the classifier first.
candidates_scored_pd = p.transform(test_names)

# 5. scoring: for each name-to-match, select the best ground-truth candidate.
best_candidates = candidates_scored_pd[candidates_scored_pd.best_match]
best_candidates.head()
```

For Spark, you can use the class `SparkEntityMatching` instead, with the same API as the Pandas version.
For all available examples, please see the tutorial notebooks under `notebooks/`.

## Project contributors

This package was authored by ING Analytics Wholesale Banking.

## Contact and support

Contact the WBAA team via Github issues.
Please note that INGA-WB provides support only on a best-effort basis.

## License

Copyright ING WBAA 2023. Entity Matching Model is completely free, open-source and licensed under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
