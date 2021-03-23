
# Fairness Aware Classification

This repository contains **tools** to address **fairness issues** in **classification problems**.

**Authors:** Kirill Myasoedov, [Simona Nitti](https://github.com/simonanitti), Bekarys Nurtay,, [Ksenia Osipova](https://github.com/Ksenia-Osipova),  and [Gabriel Rozzonelli](https://github.com/rozzong).

## Content

The module contatins the following:

- A few `classifiers` implemented in order to deal with fairness problems:

	| Classifier                  | Related paper                                                                                                                                             |
	|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
	| `AdaFairClassifier`         | [AdaFair: Cumulative Fairness Adaptive Boosting](https://arxiv.org/abs/1909.08982) by Iosifidis *et al*.                                                  |
	| `AdaptiveWeightsClassifier` | [Adaptive Sensitive Reweighting to Mitigate Bias in Fairness-aware Classification](https://dl.acm.org/doi/10.1145/3178876.3186133) by Krasanakis *et al*. |
	| `SMOTEBoostClassifier`      | [SMOTEBoost: Improving Prediction of the Minority Class in Boosting](https://link.springer.com/chapter/10.1007/978-3-540-39804-2_12) by Chawla *et al*.   |
- Some `metrics` to help assessing fairness:
	- DFPR, DFNR, Eq.Odds
	- *p*-rule
	- Sensitive TPR and TNR
- Some popular `datasets` to run experiments and play around:
    - [Adult Census Income](https://archive.ics.uci.edu/ml/datasets/Adult)
    - [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
    - [COMPAS](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)
    - [KDD Census Income](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD))

- A couple of `utils` functions to ease possible preprocessing steps.

## Installation

### Dependencies

In order to run the provided modules, the following packages are needed:

- `numpy==1.19.5`
- `pandas==1.1.5`
- `scikit-learn==0.24.1`

### Clone this repository

```bash
git clone https://github.com/rozzong/Fairness-Aware-Classification.git
```

## Examples

### Load a toy dataset

The module `datasets` contains some already preprocessed popular datasets for imbalanced classification problems leading to fairness issues.
```python
from sklearn.model_selection import train_test_split
from fairness_aware_classification.datasets import COMPASDataset

# Load the data
data = COMPASDataset()

# Split the data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    data.X,
    data.y,
    data.sensitive
)
```
In addition to the usual samples and targets, some classifiers require a mask containing information about sensitive samples as input. This mask can be retrieved with accessing `data.sensitive`.

### Load a custom dataset

For custom datasets, `utils` comes with a couple of functions to generate sensitive masks.

```python
import pandas as pd
from fairness_aware_classification.utils import sensitive_mask_from_features

# Load the data
df = pd.read_csv("my_dataset.csv")

# Set the target and do some feature selection
y = df.pop("target")
X = df.drop(["useless_feature_1"], axis=1)

# Compute the sensitive samples mask
sensitive_features = ["gender"]
sensitive_values = [0]
sensitive = sensitive_mask_from_features(X, sensitive_features, sensitive_values)
```

### Run a classifier

Classifiers from the module are meant to be used in a `scikit-learn` fashion. Some functions contained in `metrics` can be useful to define fairness-oriented objective functions.

```python
from sklearn.metrics import accuracy_score
from fairness_aware_classification.metrics import dfpr_score, dfnr_score
from fairness_aware_classification.classifiers import AdaptiveWeightsClassifier

# The criterion function `objective` should be customized
# depending on the data. It should be maximized.
def objective(y_true, y_pred, sensitive):
    acc = accuracy_score(y_true, y_pred)
    dfpr = dfpr_score(y_true, y_pred, sensitive)
    dfnr = dfnr_score(y_true, y_pred, sensitive)
    
    return 2 * acc - abs(dfpr) - abs(dfnr)

base_clf = LogisticRegression(solver="liblinear")
awc = AdaptiveWeightsClassifier(base_clf, objective)
awc.fit(X_train, y_train, s_train)
y_pred = awc.predict(X_test)
```

For each provided toy datasets, its suggested objective function is accessible with `data.objective`.

## Results

In `main.ipynb`, the implemented classifiers are compared with a simple original AdaBoost classifier. The results of these runs on the four provided datasets are presented below.

| **Adult Census Income** | **Bank marketing** |
|:-:|:-:|
| <img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/Adult.png" width="350" height="350"> | <img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/KDD.png" width="350" height="350"> |
| **COMPAS** | **KDD Census Income** |
| <img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/Adult.png" width="350" height="350"> | <img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/KDD.png" width="350" height="350"> |

