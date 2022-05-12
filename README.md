# ABCEI_MAB

This notebook introduces the code framework for reproducing the results of the
NeurIPS paper,
Alan Malek, and Silvia Chiappa. "Asymptotically Best Causal Effect Identification
with Multi-Armed Bandits." Advances in Neural Information Processing Systems 34 (2021).
The project name is an abbreviation of the title.

Roughly, we have a causal effect and several estimators that can measure it. We will
try to select the estimator with the best cost-adjusted asymptotic variance in a
sequential decision making problem where each round, we choose an estimator and
obtain a sample from the covariates it requires. We use a best-arm-identification
algorithm to choose which estimator to sample from.

This project contains code to:
1) describe and simulate data from a graphical model
2) Fit the causal effects with nuisance functions given this data
3) Construct confidence intervals for this causal effect,
4) Run a bandit algorithm using these confidence intervals
5) Provide an example notebook that generates the plots in the paper.

## Usage

The companion colab notebook thoroughly describes the intended usesage.

[![Open In Colab](https://colab.sandbox.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/deepmind/abcei_mab/blob/main/notebooks/paper_experiments.ipynb)

## Citing this work

```
@article{malek2021asymptotically,
  title={Asymptotically Best Causal Effect Identification with Multi-Armed Bandits},
  author={Malek, Alan and Chiappa, Silvia},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Disclaimer

This is not an official Google product.
