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
"""Utilities for generating random nonlinear functions."""
import itertools
from typing import Callable, Optional

from absl import logging
import numpy as np
from sklearn import gaussian_process


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


def generate_random_nd_function(
    dim: int,
    out_dim: int = 1,
    *,
    kernel: Optional[gaussian_process.kernels.Kernel] = None,
    min_x: float = -1,
    max_x: float = 1,
    n_points: int = 10,
    seed: Optional[int] = None,
    alpha: float = 100,
    n_point_warning: int = 1500) -> Callable[[np.ndarray], np.ndarray]:
  """A multidimensional version of generate_random_function.

  Each component is R -> R^out_dim, and there and dim
  components. Each component is sampled from a Gaussian process prior
  over n_points. The prior is then fit to these points, which specifies
  a sample of the function.

  Args:
    dim: the input dimension
    out_dim: the output dimension
    kernel: which kernel to use for the GP
    min_x: the minimum value of x
    max_x: the maximum value of x
    n_points: the number of points used to fit the GP
    seed: random seed
    alpha: kernel hyperparameter
    n_point_warning: the number of points that can be used in the GP without
      raining a warning.

  Returns:
    mapping from (n, dim) to (n, out_dim)
  """
  if kernel is None:
    kernel = gaussian_process.kernels.ExpSineSquared(
        length_scale=1,
        periodicity=5.0,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(1.0, 10.0))
  if n_points**dim > n_point_warning:
    logging.warning("n_points**dim=%i>%i; GaussianProcessRegressor may crash.",
                    n_points**dim, n_point_warning)

  # Specify Gaussian Process
  x1d = np.linspace(0, 1, n_points)
  # create a cartesian product
  x = np.array(list(itertools.product(*[x1d] * dim))) * (max_x - min_x) + min_x
  # Choose a random starting state
  fns = []
  for _ in range(out_dim):
    fns.append(gaussian_process.GaussianProcessRegressor(kernel, alpha=alpha))
    if seed is None:
      seed = np.random.randint(10000)
    # Sample from a prior
    y = fns[-1].sample_y(x, 1, random_state=seed)
    # Fit the GP to this prior
    fns[-1].fit(x, y)

  # Now we need to map (n_units, n_dim)

  def out_f(x):
    output = []
    for d in range(out_dim):
      output.append(fns[d].predict(x.reshape(x.shape[0], -1)))
    # We want to return (n_units, out_dim)
    return np.column_stack(output)

  return out_f
