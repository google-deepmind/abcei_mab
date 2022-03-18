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
"""Classes that estimate the asymptotic variance."""

import enum
from typing import Callable, Tuple
from causal_effect_bandits import data
from causal_effect_bandits import parameters
import numpy as np
import scipy


class VarianceEstimatorArm:
  """Calculate a confidence interval on the variance under nuisance.

  It uses a NuisanceParameter class, with its influence function, and
  builds a variance estimator on top.

  Data is persisted, so only incremental data updates should be
  provided. There are three technical pieces required:
  1. A model to fit the nuisance parameter
  2. Data processing that removes the unnecessary covarites from the data
  2. A meta algorithm that uses the influence function and nuisance estimator
     and creates an estimate of the variance

  Specific meta-algorithms will be implemented by sub-classes.

  Attributes:
    eta: parameters.NuisanceParameter The nuisance parameter
    _data_transformer: function processes an ExpData object into the format
      expected by
      the nuissanceParameter object: it must map nd.array of shape (n_units,
        n_features) to nd.array of shape (n_units, n_features_eta)
    _data: accumulated data (data.ExpData)
    ci: (float, float) The most recently calculated confidence interval
    delta: The delta used in calculating _ci.
    var_est: The most recent variance estimate.
    ate: a float of the average treatment effect estimate.
    name: str the name of the formula
    num_pulled: an int for the number of times the arm has been updated
    cost: the cost of sampling from this estimator
  Methods:
    update(new_data: data.ExpData): appends the data and recomputes the ci
    get_ci: returns the confidence interval
  """

  def __init__(self,
               name: str,
               eta: parameters.NuisanceParameter,
               data_transformer: Callable[[data.ExpData], data.ExpData],
               cost: float = 1):
    self._name = name
    self._ci = (-np.inf, np.inf)
    self._eta = eta
    self._data_transformer = data_transformer
    self._ate = None
    self._var_est = None
    self._num_pulled = 0
    self._split_data = None
    self.cost = cost

  def update(self, new_data: data.ExpData, delta: float) -> Tuple[float, float]:
    """Appends new_data to the dataset and recomputes the variance ci.

    Args:
      new_data: data.ExpData to append
      delta: the error tolerance

    Returns:
      A confidence interval that holds with probability 1-delta.
    """
    del new_data, delta  # for subclassing
    self._ci = (0, np.inf)  # default ci.
    self.num_pulled += 1
    return self._ci

  def reset(self) -> None:
    """Resets the arm to its default state."""
    self._ci = (None, None)
    self._eta.reset()
    self._ate = None
    self._var_est = None
    self._num_pulled = 0
    self._split_data = None

  def get_num_units(self) -> int:
    """Returns the number of units seen by this arm."""
    return 0

  @property
  def name(self) -> str:
    return self._name

  @property
  def ci(self) -> Tuple[float, float]:
    """Returns the most recent confidence interval."""
    return self._ci

  @property
  def var_est(self) -> float:
    return self._var_est

  @property
  def num_pulled(self) -> int:
    return self._num_pulled

  @property
  def ate(self) -> float:
    return self._ate

  @property
  def eta(self) -> parameters.NuisanceParameter:
    return self._eta


class CIAlgorithm(enum.Enum):
  CHI2 = "chi2"
  CLT = "CLT"
  FINITE_SAMPLE = "finite_sample"


class SampleSplittingArm(VarianceEstimatorArm):
  """Implements the sample-splitting variance estimator in the paper.

  This class assumes a uncentered influence function.

  Data is split into two folds
     - Fold 1 is used to fit eta
     - Fold 2 is used to estimate the variance.

  Further, the calculation of the confidence interval has several options.

  Attributes:
    stable_splits: bool incrementally add data to either split vs. resplitting
      all the data every update.
    sub_gscale: float the subGaussian parameter for phi(W, eta)
    eta_sub_gscale: float the subGaussian parameter for eta
    tau_bound: flout A bound on the maximum tau.
    burn_in: int When to start the confidence sequence. Equal to m in the paper.
    eta: a parameters.NuisanceParameter class
    delta: the high probability tolerance
    n_splits: the number of splits, ethier 2 or 3. A 3-way split will use a
      separate fold of data to estimate ATE.
    rho: hyperparameter for the Gaussian mixture cofidence sequence.
  """

  def __init__(
      self,
      name: str,
      eta: parameters.NuisanceParameter,
      data_transformer: Callable[[data.ExpData], data.ExpData],
      ci_algorithm: CIAlgorithm = CIAlgorithm.FINITE_SAMPLE,
      cost: float = 1,
      stable_splits: bool = True,
      sub_gscale: float = 1,
      eta_sub_gscale: float = 1,
      tau_bound: float = 1,
      burn_in: float = 0,
      rho: float = .1,
  ):
    super().__init__(name, eta, data_transformer, cost)
    self._ci_algorithm = ci_algorithm
    self._stable_splits = stable_splits
    self._sub_gscale = sub_gscale
    self._eta_sub_gscale = eta_sub_gscale
    self._tau_bound = tau_bound
    self._burn_in = burn_in
    self._rho = rho

  def update(
      self,
      new_data: data.ExpData,
      delta: float,
  ) -> Tuple[float, float]:
    """Appends new_data to the dataset and recomputes the variance ci.

    Args:
      new_data: data.ExpData to append
      delta: the error tolerance

    Returns:
      A confidence interval that holds with probability 1-delta.
    """
    if not 0 < delta < 1:
      raise ValueError(f"delta={delta}, but expected to be in (0,1)")

    self._num_pulled += 1

    self.delta = delta

    if self._split_data is None:
      self._split_data = self._data_transformer(new_data).k_fold(2)
    else:
      if self._stable_splits:
        new_split = self._data_transformer(new_data).k_fold(2)
        for i in range(2):
          self._split_data[i].append(new_split[i])
      else:
        self._split_data = self._data_transformer(new_data).k_fold(2)

    # Use the first split to train eta
    self._eta.fit(self._split_data[0])

    # We need to check that all folds and all control, treatment groups
    # Have at least one unit. Otherwise, return [0, np.inf) for the confidence.
    if not self._eta.has_enough_samples():
      self._ci = (0, np.inf)
      self._ate = np.nan
      self._var_est = np.inf
      return self._ci

    # Use the second split to learn the ATE
    scores = self._eta.calculate_score(self._split_data[1])
    self._ate = np.mean(scores)

    # Use the second or third split to calculate the variance
    self._var_est = np.mean(
        (self._eta.calculate_score(self._split_data[1]) - self._ate)**2)
    if np.isnan(self._var_est):
      raise ValueError("var_est is a nan!")

    if self._ci_algorithm is CIAlgorithm.CHI2:
      n = self._split_data[1].n_units
      ci_low = self.var_est * (n - 1) / scipy.stats.chi2.ppf(
          1 - delta / 2, df=n - 1)
      ci_high = self.var_est * (n - 1) / scipy.stats.chi2.ppf(
          delta / 2, df=n - 1)
      self._ci = (ci_low, ci_high)

    elif self._ci_algorithm is CIAlgorithm.CLT:
      clt_delta = delta / self._num_pulled**2
      n = self._split_data[1].n_units
      scores = self._eta.calculate_score(self._split_data[1])
      var_of_var = np.var((scores - self._ate)**2)
      ci_low = max(
          self.var_est +
          scipy.stats.norm.ppf(clt_delta / 2) * np.sqrt(var_of_var / n), 0)
      ci_high = self.var_est + scipy.stats.norm.ppf(1 - clt_delta /
                                                    2) * np.sqrt(var_of_var / n)
      self._ci = (ci_low, ci_high)

    elif self._ci_algorithm is CIAlgorithm.FINITE_SAMPLE:
      if self._tau_bound <= 0 or self._burn_in <= 1 or self._sub_gscale <= 0:
        raise ValueError("tau_bound, burn_in, and subgScale must be positive")
      eta_cs_width = self._eta.calculate_cs_width(
          delta / 2,
          sub_g=self._eta_sub_gscale,
          rho=self._rho,
      )
      n = len(self._split_data[1])
      if self._burn_in > n:
        self._ci = (0, np.inf)
        return self._ci
      # See the paper, section 3.5 for a justification of this formula
      lambdap = max(self._sub_gscale, 8 * self._sub_gscale**2)
      width = 5 / 8 * (np.sqrt(2 * self._sub_gscale) + self._tau_bound)
      width *= np.sqrt(2 * np.log(lambdap * n / self._burn_in) / n + .5 +
                       np.log(4 / delta)) / np.sqrt(n)
      width += eta_cs_width**2
      self._ci = (max(self.var_est - width, 0), self.var_est + width)
    else:
      raise ValueError(f"{self._ci_algorithm} is an unknown CI algorithm name.")

    return self._ci

  def get_num_units(self) -> int:
    """Returns the number of units seen by this arm."""
    if self._split_data is None:
      return 0
    return sum(len(data) for data in self._split_data)

  def reset(self) -> None:
    super().reset()
    self._split_data = None


class BootstrapArm(VarianceEstimatorArm):
  """Uses a bootstrap to estimate the variance.

  The bootstrap only has asymptotic guarantees but
  it could yield good empirical performance.
  """

  def __init__(
      self,
      name: str,
      eta: parameters.NuisanceParameter,
      data_transformer: Callable[[data.ExpData], data.ExpData],
      n_bootstraps: int = 0,
      stable_splits: bool = True,
  ):
    super().__init__(name, eta, data_transformer)
    self.n_bootstraps = n_bootstraps
    self.n_splits = 2
    self._stable_splits = stable_splits

  def update(
      self,
      new_data: data.ExpData,
      delta: float,
  ) -> Tuple[float, float]:
    """Appends new_data to the dataset and recomputes the variance ci.

    Args:
      new_data: data.ExpData to append
      delta: the error tolerance

    Returns:
      A confidence interval that holds with probability 1-delta.
    """
    if not 0 < delta < 1:
      raise ValueError(f"Given delta={delta}, but 0<delta<1 is required.")

    self._num_pulled += 1

    self.delta = delta

    if self._split_data is None:
      self._split_data = self._data_transformer(new_data).k_fold(self.n_splits)
    else:
      if self._stable_splits:
        new_split = self._data_transformer(new_data).k_fold(self.n_splits)
        for i in range(self.n_splits):
          self._split_data[i].append(new_split[i])
      else:
        self._split_data = self._data_transformer(new_data).k_fold(
            self.n_splits)

    # Use the first split to train eta
    self._eta.fit(self._split_data[0])

    if self.n_bootstraps == 0:
      n_bootstraps = int(np.ceil(4 / delta))
    else:
      n_bootstraps = self.n_bootstraps

    # Compute tau
    b_samples = []
    for _ in range(n_bootstraps):
      # Compute the data with the bth fold missing
      (in_data, out_data) = self._split_data[0].subsample(1.0 / n_bootstraps)

      # Calculate tau on this fold
      self._eta.fit(in_data)
      b_samples.append(np.var(self._eta.calculate_score(out_data)))

    # Next, we need to calculate the percentiles of the bootstrap samples
    b_samples = np.array(b_samples)
    b_samples.sort()
    ci_low = b_samples[int(np.floor(delta / 2 * n_bootstraps))]
    ci_high = b_samples[int(np.ceil((1 - delta / 2) * n_bootstraps)) - 1]
    self._ci = (ci_low, ci_high)

    # Use the second split to learn the ATE and variance
    self._ate = np.mean(self._eta.calculate_score(self._split_data[1]))

    self._var_est = np.var(self._eta.calculate_score(self._split_data[1]))

    return self._ci
