# Fairness-Aware-Classification

This repository contains three methods to address fairness issues in classification problems implemented by the team of Bekarys Nurtay, Simona Nitti, Ksenia Osipova, Kirill Myasoedov and Gabriel Rozzonelli during a Machine Learning Project at Skoltech, 2021.

## Problem discription
- *Problem:* AI-decision making systems are noticed in discrimination: 
    - advertising of highly paid jobs more men than women ([Google’s AdFicher](https://www.andrew.cmu.edu/user/danupam/dtd-pets15.pdf))
    - ignoring of city areas inhabited mostly by black people for eligibility to advanced services of [Amazon](https://www.bloomberg.com/graphics/2016-amazon-same-day/)

- *Protected (sensitive) group* - group of people which can be subjected to discrimination on the basis of gender, race, age, etc. (ex: females, blacks)

- *Sensitive feature* - feature characterizing discrimination basis (ex: gender, race)

- *Clasiifiers* implemented in order to deal with fairness problrm:

	| Classifier                  | Related paper                                                                                                                                           |
	|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
	| `AdaFairClassifier`         | [AdaFair: Cumulative Fairness Adaptive Boosting](https://arxiv.org/abs/1909.08982) by Iosifidis et al.                                                  |
	| `AdaptiveWeightsClassifier` | [Adaptive Sensitive Reweighting to Mitigate Bias in Fairness-aware Classification](https://dl.acm.org/doi/10.1145/3178876.3186133) by Krasanakis et al. |
	| `SMOTEBoostClassifier`      | [SMOTEBoost: Improving Prediction of the Minority Class in Boosting](https://link.springer.com/chapter/10.1007/978-3-540-39804-2_12) by Nitesh V. Chawla et al.                                                                                                                                 |

- *Equalized Odds* - fairness metrics, measures the difference of true classified samples between sensitive and non-sensitive groups

- Other **Metrics** to assess fairness:
    - Accuracy
    - Balanced accuracy
    - True Positive Rate for protected group
    - True Positive Rate for non-protected group
    - True Negative Rate for protected group
    - True Negative Rate for non-protected group

- *Datasets* for running experiments:
    - [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult)
    - [Bank marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
    - [Compass Data Set](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD))
    - [Census-Income (KDD) Data Set](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD))

## Prerequests

- `numpy==1.19.5`
- `pandas==1.1.5`
- `scikit-learn==0.24.1`

## Repositoty structure

- `main.py` - main notebook for launching the experinemnts for one dataset and accuracy scores calculation. Results are saved as csv-file;
- *fainess_aware_classification* folder contains the following:
    - *classifiers* folder contains implemented algorithms:
        - `adafair.py` - AdaFair algorithm;
        - `adaptive_weights.py` - Adaptive Sensitive Reweighting algorithm;
        - `smoteboost.py` - SMOTEBoost algorithm;
    - *datasets* folder contains:
        - *raw_data* folder with original datasets;
        - `dataset.py` - general preprocessing procedure for datasets;
        - `adult.py` - preprocessing procedure of Adult census income dataset;
        - `bank.py` - preprocessing procedure of Bank markiting dataset;
        - `compas.py` - preprocessing procedure of COMPASS dataset;
        - `kdd.py` - preprocessing procedure of KDD dataset;
        - `__init__.py` -  four mentioned preprocessed datasets;
    - `metrics.py` - a collection of metrics used for fairness assessment;
    - `utils.py` - additional functions used in the project

## Fairness-Aware-Classification package installation and usage

In order to get use github repositary, at first, you should to clone it:

`!git clone https://github.com/rozzong/Fairness-Aware-Classification`

On the second step you should set the path to the cloned package:

`import sys`

`sys.path.insert(0, "Fairness-Aware-Classification")`

After these steps you can import any item from the repository, for example:

`from fairness_aware_classification.classifiers import AdaFairClassifier`

## Results

Four algorithms (including original AdaBoost) were run and tested via bootstrap with 10 test-train splits of ration 0.5. The results of these runs on four datasets are represented below.

The fairness and accuracy scores comparison of four algoritms on Adult census income dataset:

<img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/Adult.png" width="350" height="350">

The fairness and accuracy scores comparison of four algoritms on Bank marketing dataset:

<img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/Bank.png" width="350" height="350">

The fairness and accuracy scores comparison of four algoritms on COMPASS dataset:

<img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/Compass.png" width="350" height="350">

The fairness and accuracy scores comparison of four algoritms on Census-income (KDD) dataset:

<img src="https://github.com/rozzong/Fairness-Aware-Classification/blob/main/results_images/KDD.png" width="350" height="350">
