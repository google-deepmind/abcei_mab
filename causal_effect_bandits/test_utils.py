# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code is needed to execute the simulations of
# Malek, Chiappa. "Asymptotically Best Causal Effect
# Identification with Multi-Armed Bandits." NeurIPS, 2021.
# https://proceedings.neurips.cc/paper/2021/...
#    hash/b8102d1fa5df93e62cf26cd4400a0727-Abstract.html
"""Utilities for the unit tests."""
from typing import Optional
from causal_effect_bandits import arms
from causal_effect_bandits import data
import numpy as np


def run_single_arm(arm: arms.VarianceEstimatorArm,
                   data_gen: data.DataGenerator,
                   max_samples: int,
                   increment: int,
                   total_delta: Optional[float] = .05):
  """Runs a single variance estimator arm.

  Recalculates the confidence intervals, ate, and variance estimators
  for a sequence of datasets, where each dataset is the previous with
  increment more samples.

  Args:
    arm: the arm to run
    data_gen: the DataGenerator
    max_samples: the number of samples used
    increment: the number of samples generated between updates of the arm
    total_delta: the total error for the confidence intervals of arm

  Returns:
    A dictionary with keys, values:
    'LCBs': np.ndarray of lower confidence bounds for the dataset
      sequence
    'UCBs': np.ndarray of upper confidence bounds for the dataset
      sequence
    'var_estimates': np.ndarray of variance estimates for the dataset
      sequence
    'ATE_estimates': np.ndarray of ATE estimates for the dataset
      sequence
    'n_samples': np.ndarray of sizes of the dataset sequence
  """
  lcbs = []
  ucbs = []
  var_estimates = []
  ate_estimates = []
  n_samples = []

  current_n_samples = 0
  arm_pulled = 0
  while current_n_samples < max_samples:
    current_n_samples += increment
    n_samples.append(current_n_samples)
    new_data = data_gen.generate(increment)

    arm.update(new_data, delta=total_delta)
    lcbs.append(arm.ci[0])
    ucbs.append(arm.ci[1])
    var_estimates.append(arm.var_est)
    ate_estimates.append(arm.ate)

  arm_pulled += 1

  return {
      'LCBs': lcbs,
      'UCBs': ucbs,
      'var_estimates': var_estimates,
      'ATE_estmitates': ate_estimates,
      'n_samples': n_samples
  }


def estimate_true_variance(
    data_gen: data.DataGenerator,
    arm: arms.VarianceEstimatorArm,
    n_samples: Optional[int] = 10000,
) -> float:
  """This method uses the true parameters to estimate the variance.

  Calculates
    E_n[(phi(W, eta) - tau)^2].
  We use an empirical estimation to approximate the expected value,
  at the true values of eta and tau.

  Args:
    data_gen: A data.DataGenerator used to produce the data
    arm: arm used to estimate the variance
    n_samples: the number of samples to generate

  Returns:
    A real number estimate of the true variance.
  """
  new_data = data_gen.generate(n_samples)
  arm.eta.set_to_truth(data_gen)

  return np.var(arm.eta.calculate_score(new_data))


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)
