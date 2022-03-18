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
"""Class definitions for data holding objects."""

import collections
import dataclasses
from typing import Callable, List

import numpy as np


@dataclasses.dataclass
class ExpData:
  """A class to hold observational data.

  In experiments, we will routinely have three types of data:

    covariates cov: n_units * n_features
    exposure exp: n_features * 1
    response rsp: n_features * 1
  This is a wrapper class collecting all three. Data are assumed
  to be np.ndarrays.

  Methods:
    append: appends another ExpData object to this one
    k_fold(k): returns k roughly equally large ExpData objects
  """
  cov: np.ndarray
  exp: np.ndarray
  rsp: np.ndarray

  def __init__(self, cov: np.ndarray, exp: np.ndarray, y: np.ndarray):
    if cov is None:
      # Make an empty ExpData
      self.n_units = 0
      self.n_features = 0
    else:
      self.n_units = cov.shape[0]
      if len(cov.shape) == 1:  # one dimensional
        self.n_features = 1
      else:
        self.n_features = cov.shape[1]
    self.cov = cov
    self.rsp = y
    self.exp = exp

  def __len__(self):
    return self.n_units

  def append(self, new_data: "ExpData"):
    """Appends the new data."""
    if new_data.n_features != self.n_features:
      raise ValueError("The new data has incompatible shape")
    self.cov = np.r_[self.cov, new_data.cov]
    if self.exp is not None:
      if new_data.exp is None:
        raise ValueError("new_data.exp is missing but expected")
      self.exp = np.r_[self.exp, new_data.exp]
    if self.rsp is not None:
      if new_data.rsp is None:
        raise ValueError("new_data.rsp is missing but expected")
      self.rsp = np.r_[self.rsp, new_data.rsp]

    self.n_units = self.cov.shape[0]

  def k_fold(self, k: int):
    """Breaks the data into k folds according to some random permutation."""
    if k < 1:
      raise ValueError(
          f"Number of folds  must be a positive integer, but {k} was provided.")
    folds = np.random.permutation(np.mod(np.arange(self.n_units), k))

    data_folds = []
    for i in range(k):
      cov_fold = self.cov[folds == i]
      if self.exp is not None:
        exp_fold = self.exp[folds == i]
      else:
        exp_fold = None
      if self.rsp is not None:
        rsp_fold = self.rsp[folds == i]
      else:
        rsp_fold = None
      data_folds.append(ExpData(cov_fold, exp_fold, rsp_fold))

    return data_folds

  def subsample(self, p: float):
    """Randomly partitions the data into two groups.

    Args:
      p: the proportion to allocate to the first folt.

    Returns:
      Two ExpData objects with approximate size n_units*p and n_units*(1-p).
    """
    if not 0 <= p <= 1:
      raise ValueError(f"p={p} provided, but should be in the unit interval.")

    mask = np.random.choice(a=[True, False], size=(self.n_units), p=[p, 1 - p])

    if self.exp is not None:
      exp_in = self.exp[mask]
      exp_out = self.exp[~mask]
    else:
      exp_in = None
      exp_out = None
    if self.rsp is not None:
      rsp_in = self.rsp[mask]
      rsp_out = self.rsp[~mask]
    else:
      rsp_in = None
      rsp_out = None

    in_data = ExpData(self.cov[mask], exp_in, rsp_in)
    ouexp_data = ExpData(self.cov[~mask], exp_out, rsp_out)
    return (in_data, ouexp_data)

  def __str__(self):
    string = f"An ExpData class with {int(self.n_units)} units. The data are:"
    string += "\n covariates:" + str(self.cov)
    string += "\n exposures:" + str(self.exp)
    string += "\n response:" + str(self.rsp)
    return string

  def __getitem__(self, key):
    new_cov = self.cov[key]
    new_exp = None if self.exp is None else self.exp[key]
    new_rsp = None if self.rsp is None else self.rsp[key]
    return ExpData(new_cov, new_exp, new_rsp)


# Below are some common data_transformer functions.
def get_identity_fn() -> Callable[[ExpData], ExpData]:
  """This function returns the identity function for use as a data_transformer.

  Returns:
    The identity function for use as a data_transformer.
  """

  def fn(x):
    return x

  return fn


def get_remove_coordinates_fn(
    idx_to_remove: int,) -> Callable[[ExpData], ExpData]:
  """Returns a function that maps an ExpData to another ExpData.

  Args:
    idx_to_remove: an index to remove

  Returns:
    A function that maps ExpData to ExpData with the corresponding index of cov
    removed.
    Suitable for use as a data_transformer.
  """

  def fn(data):
    new_cov = np.delete(data.cov, idx_to_remove, axis=1)
    if new_cov.ndim == 1:  # new covariate is one dimensional
      new_cov = new_cov.reshape(-1, 1)
    return ExpData(new_cov, data.exp, data.rsp)

  return fn


def get_coordinate_mask_fn(
    idx_to_include: List[int],) -> Callable[[ExpData], ExpData]:
  """Returns a function that maps an ExpData to another ExpData.

  Args:
    idx_to_include:  indices to include

  Returns:
    A function that maps ExpData to ExpData with only indices
      in idx_to_include remaining.
      Suitable for use as a data_transformer.
  """

  def fn(d):
    return ExpData(d.cov[..., idx_to_include], d.exp, d.rsp)

  return fn


class DataGenerator:
  """Generates data from some joint probability model on X, T, and Y.

  The data are
    covariates X in R^d
    binary treatment T
    response Y in R
  Specific probability models, like SCMs or linear models, will be implemented
  as subclasses.

  Methods:
    generate(n_units): returns an ExpData with n_units
  """

  def __init__(self, name: str):
    self._name = name

  def generate(self, n_samples: int):
    return ExpData(
        np.zeros(n_samples, 1),
        np.zeros(n_samples),
        np.zeros(n_samples),
    )


class LinearDataGenerator(DataGenerator):
  """Generates data according to simple linear model.

  Specifically, the linear model is specified by
  Y = X beta + T*tau + epsilon
  T = Bernoulli(sigmoid(X gamma))
  where epsilon has a distribution given by noise_model and
  T is binary but correlated with the features X.

  Attributes:
    n_features: int the feature dimension of X. Features have a standard normal
      distribution
    beta: np.array of shape (n_features,)
    tau: float
    gamma: np.array of shape (n_features,)
    noise_model: a function accepting an integer n where noise_model(n) returns
      a np.array of length n.
  """

  def __init__(
      self,
      name: str,
      n_features: int,
      beta: np.ndarray,
      tau: float,
      noise_model: Callable[[int], np.ndarray] = np.random.randn,
      gamma=None,
  ):
    super().__init__(name)
    self._beta = beta
    self._tau = tau
    self._noise_model = noise_model
    if gamma is None:  # Default is no dependence between X and T.
      self._gamma = np.zeros(n_features)
    elif gamma.shape != (n_features,):
      raise ValueError(
          f"shape of gamma is {gamma.shape}, but ({n_features},) expected.")
    else:
      self._gamma = gamma

    self._n_features = n_features

  def generate(self, n_samples: int):
    x = np.random.randn(n_samples, self._n_features)

    t_means = 1 / (1 + np.exp(-x.dot(self._gamma)))
    t = np.random.binomial(1, t_means)
    noise = self._noise_model(n_samples)

    if len(noise) != n_samples:
      raise ValueError(
          f"noise_model's output is {len(noise)} but should be {n_samples}.")

    y = x.dot(self._beta) + self._tau * t + self._noise_model(n_samples)
    return ExpData(x, t, y)

  @property
  def tau(self):
    return self._tau

  @property
  def beta(self):
    return self._beta

  @property
  def gamma(self):
    return self._gamma


class TabularCPD:
  """Estimates a tabular CPD.

  Given finite support random variables Z and X, this class fits an estimator

  of P(Z|X).

  Because np.arrays are not hashable, all keys will use .tobytes() instead.
  Therefore, we need to keep dicts of .tobytes() to np.array values.

  Attributes:
    min_ratio_to_uniform: The model is fit so that the minimum ration between
      probabilities and a uniform distribution over the estimated support is at
      least this value. Larger values allows for smaller predicted probabilities
      an the expense of larger variance.
    n_units: and integer number of units
    table: a table, implemented as a dict, of counts
    x_marginal: marginal probabilities by value of X
    x_values: all the unique values of X seen
    z_values: all the unique values of Z seen.
    min_prob: the minimum probability in the table.
  """

  def __init__(self, min_ratio_to_uniform=20):
    default_fn = lambda: collections.defaultdict(lambda: 0.0)
    self.n_units = 0
    self.table = collections.defaultdict(default_fn)
    self.x_marginal = collections.defaultdict(lambda: 0.0)
    self.x_values = {}
    self.z_values = {}
    self.min_ratio_to_uniform = min_ratio_to_uniform

  def fit(self, x, z):
    """Fits the tabular CPD to data.

    Args:
      x: an np.ndarray of x observations of size (n_units,). Should have finite
        support
      z: an np.ndarray of z observations of size (n_units, n_features). Should
        have finite support.
    """
    x = np.asarray(x)
    z = np.asarray(z)

    if len(x) != len(z):
      raise ValueError("z and x must be the same length")
    self.n_units = len(x)
    default_fn = lambda: collections.defaultdict(lambda: 0.0)
    self.table = collections.defaultdict(default_fn)
    self.x_marginal = collections.defaultdict(lambda: 0.0)
    for xi, zi in zip(x, z):
      self.table[xi.tobytes()][
          zi.tobytes()] = self.table[xi.tobytes()][zi.tobytes()] + 1
      self.x_values[xi.tobytes()] = xi
      self.z_values[zi.tobytes()] = zi

    # adjust min_ratio_to_uniform based on the perceived support size
    # Note: we use a naive missing mass estimatior, but
    # we could use a better one, e.g. Good-Turning.
    support_size_estimate = len(self.z_values)
    self.min_prob = 1 / support_size_estimate / self.min_ratio_to_uniform
    if self.min_prob > 1:
      self.min_prob = .2

    # Next, we normalize the probabilities for every x
    for x_key in self.x_values:
      x_table = self.table[x_key]
      num_samples = sum(x_table.values())
      self.x_marginal[x_key] = num_samples / self.n_units
      non_violating_z = []
      total_violations = 0
      for z_key in self.z_values:
        x_table[z_key] = x_table[z_key] / num_samples
        if x_table[z_key] < self.min_prob:
          total_violations += self.min_prob - x_table[z_key]
          x_table[z_key] = self.min_prob
        elif x_table[z_key] > 1 - self.min_prob:
          total_violations += x_table[z_key] - (1 - self.min_prob)
          x_table[z_key] = 1 - self.min_prob
        else:
          non_violating_z.append(z_key)
      # Now, we adjust non_violating_z to make P(Z|x = x_key) sum to 1
      for z_key in non_violating_z:
        x_table[z_key] = (
            x_table[z_key] - total_violations / len(non_violating_z))

  def x_support(self):
    return list(self.x_values.values())

  def z_support(self):
    return list(self.z_values.values())

  def support_size(self):
    return len(self.z_values)

  def predict(self, x, z):
    """Returns P(z|x) for all elements in zip(z, x).

    Args:
      x: treatments to evaluate
      z: covariates to evaluate.

    Returns:
      A list of conditional probabilities of z|x.
    """
    probs = []
    x = np.asarray(x)
    z = np.asarray(z)

    for zi, xi in zip(z, x):
      probs.append(self.table[xi.tobytes()][zi.tobytes()])
    return np.array(probs)

  def __str__(self):
    s = ""
    for x in self.table.keys():
      s += f"Row for x = {self.x_values[x]}\n"
      for z in self.table[x].keys():
        s += f"  z={self.z_values[z]} : {self.table[x][z]}\n"
      s += "\n"
    return s
