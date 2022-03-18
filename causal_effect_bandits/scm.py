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
"""Implements a Structural Causal Model (SCM).

Also includes many common distributions and conditional
probability distributions (CPDs) as building blocks.
"""
from typing import List
from causal_effect_bandits import data
import numpy as np


class SCM(data.DataGenerator):
  """Implements a SCM model that can generate ExpData.

  An SCM is specified by a DAG and CPD for each node.
  The pgmpy class will encode this model and be responsible for sampling
  from it.

  Attributes:
    var_names: List[str] List of all variable names in the model, presented in
      lexicographic order
    parents: Mapping[str:List[str]] A dictionary with keys in var_names.
      parents[var_name] is a list of the keys corresponding to the parents of
      node var_name.
    cov_variables: List[str] List of all variable names that can be used as
      covariates
    treatment_variable: str name of variable that corresponds to the treatment
    response_variable: str name of variable that corresponds to the treatment
    cpds: a Mapping of functions A dict with var_name as keys; each accepts
      (n_samples, parent_samples) as arguments and returns a np.array of samples
      from the variable.
  """

  def __init__(
      self,
      name,
      var_names: List[str],
      parents,
      cpds,
      cov_variables: List[str],
      treatment_variable: str,
      response_variable: str,
  ):
    super().__init__(name)

    self._n_nodes = len(var_names)
    self.var_names = np.array(var_names)
    self.parents = parents
    if not set(var_names).issubset(parents):
      raise ValueError("Some variables do not have keys in parents")
    if not set(var_names).issubset(cpds):
      raise ValueError("some variables do not have keys in cpbs")

    if treatment_variable not in var_names:
      raise ValueError("The treatment variable must be a variable")
    if response_variable not in var_names:
      raise ValueError("The response variable must be a variable")
    if not set(cov_variables).issubset(var_names):
      raise ValueError("The covariate variable must be variables")

    self.treatment_variable = treatment_variable
    self.response_variable = response_variable
    self.cov_variables = cov_variables

    self.cpds = cpds

  def generate(self, n_samples: int):
    samples = {}
    for var_name in self.var_names:
      # Build parents sample
      if not set(self.parents[var_name]).issubset(samples):
        raise ValueError("var_names are not in lexicographic order")
      parent_samples = [samples[p] for p in self.parents[var_name]]

      samples[var_name] = self.cpds[var_name](n_samples, parent_samples)

    # ExpData was only designed to have a single covariate; hence, we
    # concatenate the samples corresponding to each covariate.
    # Perhaps in a future version we can do something better.
    data_to_concat = [samples[v] for v in self.cov_variables]
    x_data = np.column_stack(data_to_concat)

    return data.ExpData(
        x_data,
        samples[self.treatment_variable],
        samples[self.response_variable],
    )

  def compute_treatment_effect(self, n_samples: int):
    """Estmimates the treatment effect by changing the graph.

    Args:
      n_samples: the number of samples to use

    Returns:
      An estimate of the ATE of the SCM.
    """
    old_t_cpd = self.cpds[self.treatment_variable]
    treatment_cpd = lambda n, parents: self.deterministic_cpd(n, 1)
    self.cpds[self.treatment_variable] = treatment_cpd
    treatment_sample = self.generate(n_samples)

    control_cpd = lambda n, parents: self.deterministic_cpd(n, 0)
    self.cpds[self.treatment_variable] = control_cpd
    control_sample = self.generate(n_samples)

    self.cpds[self.treatment_variable] = old_t_cpd
    return np.mean(treatment_sample.rsp) - np.mean(control_sample.rsp)


def deterministic_cpd(n_samples, value):
  return np.full(n_samples, value)


def categorical_conditioning(parents, conditioning_on_idx, distributions):
  """A way to implement conditioning by a categorical random variable.

  When parents[treatment_idx] = i, returns distributions[i](parents, betas).

  Args:
    parents: value of parents
    conditioning_on_idx: the index used to select the distribution
    distributions: a list of distributions

  Returns:
    Samples where the each sample is selected from the  distribution
    corresponding to the respective conditioning_on_idx and evaluated
    on the parents.
  """
  if len(parents) <= conditioning_on_idx:
    raise ValueError(
        "Treatment_idx is greater than the number of distributions.")
  # Remove treatment_idx from parents
  distribution_indices = parents[conditioning_on_idx]
  del parents[conditioning_on_idx]

  samples = []
  for i, d in enumerate(distribution_indices):
    # Build the parents for this unit
    parent_sample = [p[i] for p in parents]
    samples.append(distributions[int(d)](1, parent_sample)[0])

  return np.array(samples)


def normal_cpd(parents, betas, mean, cov):
  """Simulates a linear gaussian based on the parents, betas.

  Args:
    parents: n_parent long list of np.array of shape (n_units, dim_i)
    betas: n_parent long list of np.arrays of shape (dim_i)
    mean: Float the mean of the random variable
    cov: the covariance matrix of the random variable

  Returns:
  n samples with
    X_j = sum_i parents[i,j].dot(gammas[i]) + epsilon
    with epsilon ~ normal(mean, cov)

  """
  n_units = len(parents[0])
  total_mean = parents[0].dot(betas[0])
  for i in range(1, len(parents)):
    total_mean += parents[i].dot(betas[i])
  if total_mean.ndim == 1:  # Y is univariate
    return total_mean + np.random.normal(mean, np.sqrt(cov), size=n_units)
  else:
    return total_mean + np.random.multivariate_normal(mean, cov, size=n_units)


def add_gaussian_noise(x, mean, cov):
  """Adds gaussian noise to x with the correct shape.

  Args:
    x: np.array with shape (n_units,) or (n_units, x_dim)
    mean: float or np.array with shape (1,) or (x_dim,1)
    cov: float or np.array with shape (x_dim, x_dim)

  Returns:
    np.array with shape:
    (n_units,) if mean is a float or shape (1,)
    (n_units, x_dim), otherwise.

  """
  n_units = len(x)
  if np.ndim(x) < 2 or (np.ndim(x) == 2 and
                        x.shape[1] == 1):  # variable is one-dimensional
    if np.ndim(mean) > 1:
      raise ValueError("The dimensions of x and mean are not compatible")
    return x.flatten() + np.random.normal(mean, np.sqrt(cov), size=n_units)
  else:  # variable is multi-dimensional
    if x.shape[1] != len(mean):
      raise ValueError("The dimensions of x and mean are not compatible")
    return x + np.random.multivariate_normal(mean, cov, size=n_units)


def transformed_normal_cpd(parents, betas, fns, mean, cov):
  """Simulates a transformed linear Gaussian.

  Specifically, it simulates
  Y = sum_i f_i(parents[i,j].dot(gammas[i])) + epsilon
  with epsilon ~ normal(mean, cov)

  Args:
    parents: n_parent long list of np.array of shape (n_units, dim_i)
    betas: n_parent long list of np.arrays of shape (dim_i)
    fns: n_parent long list of vector-to-vector mappings
    mean: Float the mean of the random variable
    cov: the variance of the random variable

  Returns:
     Samples of Y.
  """
  n_units = len(parents[0])
  if len(parents) != len(fns):
    raise ValueError("parents and fns should be the same length.")
  total_mean = fns[0](parents[0].dot(betas[0]))
  for i in range(1, len(parents)):
    total_mean += fns[i](parents[i].dot(betas[i]))

  # infer whether the output should be 1 or 2 dim from mean
  if np.ndim(mean) == 0:  # we should return (n_units,) array
    return total_mean.flatten() + np.random.normal(
        mean, np.sqrt(cov), size=n_units)
  else:
    return total_mean + np.random.multivariate_normal(mean, cov, size=n_units)


def structural_equation_with_noise(parents, f, mean, cov):
  """Simulates a random variable as a noisy function of the parents.

  Args:
    parents: n_parent long list of np.array of shape (n_units, dim_i)
    f: mapping we assume fn takes vectors (n_units, n_features) to (n_units,
      out_dim)
    mean: nd.array of shape (out_dim,) or scalar the mean of the random variable
    cov: nd.array of shape (out_dim, out_dim) or scalar the (co)variance of the
      random variable

  Returns:
    Samples of
    X_j = fn(parents[i,j]) + epsilon
    with epsilon ~ normal(mean, cov)

    The returned shape is inferred from the shape of mean; if mean has shape
    (d, ) or (d, 1) then a vector of shape (n_units, d) is returned.
    If mean is a scalar, then a vector or shape (n_units,) is returned.
  """
  n_units = len(parents[0])
  f_out = f(np.column_stack(parents))
  if np.ndim(mean) < 1:  # the output is a scalar
    return f_out.flatten() + np.random.normal(mean, np.sqrt(cov), size=n_units)
  else:
    return f_out + np.random.multivariate_normal(mean, cov, size=n_units)


def logistic_bernoulli_cpd(parents, f):
  """Returns n Bernoulli samples with mean softmax of parents.

  X_1,...,X_n ~ Bernoulli(sigmoid(f(parents))

  Args:
    parents: List(ndarray) n_parent long list of np.array of shape (n_units,
      dim_i)
    f: mapping This mapping takes a list of (n_unit, dim_i) arrays and returns a
      (n_unit,) list. It is up to the user to make sure that the dimensions are
      compatible.
  """
  params = f(parents)
  means = 1 / (1 + np.exp(-params))
  return np.random.binomial(1, means)


def logistic_linear_bernoulli_cpd(parents, gammas):
  """Returns n Bernoulli samples with mean softmax of linear of parents.

  X_j ~ Bernoulli(sigmoid(sum_i parents[i,j].dot(gammas[i]))

  Args:
    parents: n_parent long list of np.array of shape (n_units, dim_i)
    gammas: n_parent long list of np.arrays of shape (dim_i)
  """

  def f(parents):
    params = 0
    for i in range(len(parents)):
      params += parents[i].dot(gammas[i])
    return params

  return logistic_bernoulli_cpd(parents, f)
