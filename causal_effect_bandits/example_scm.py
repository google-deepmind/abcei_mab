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
"""Example SCMs used for unit tests."""


from causal_effect_bandits import scm
import numpy as np


def back_door_scm():
  """A common SCM used for many of the unit tests."""
  var_names = ["Z", "X", "Y"]
  cov_variables = ["Z"]
  treatment_variable = "X"
  response_variable = "Y"

  ### Define the parents
  parents = {
      "Z": [],
      "X": ["Z"],
      "Y": ["Z", "X"],
  }
  p = np.random.uniform(.2, .8)
  beta = 10 * np.random.uniform(-.3, .3)
  alpha = np.random.uniform(-.3, .3)
  tau = 10
  mu_z = p
  var_z = p * (1 - p)
  mu_x = .5 + alpha * p
  var_x = mu_x * (1 - mu_x)
  mu_y = tau * mu_x + beta * mu_z
  var_y = 1 + tau**2 * var_x + beta**2 * var_z

  # populate the CPDs
  def y_cpd(_, parents):
    return (np.random.normal(size=len(parents[0])) + tau * parents[1] +
            beta * parents[0])

  cpds = {
      "Z": lambda n, parents: np.random.binomial(1, p, size=n),
      "X": lambda n, parents: np.random.binomial(1, .5 + alpha * parents[0]),
      "Y": y_cpd,
  }

  scm_gen = scm.SCM(
      name="back-door example",
      var_names=var_names,
      parents=parents,
      cpds=cpds,
      cov_variables=cov_variables,
      treatment_variable=treatment_variable,
      response_variable=response_variable,
  )

  # Verify the means and variances

  return (scm_gen, {
      "tau": tau,
      "mu_z": mu_z,
      "var_z": var_z,
      "mu_x": mu_x,
      "var_x": var_x,
      "mu_y": mu_y,
      "var_y": var_y,
      "beta": beta,
      "alpha": alpha,
  })


def frontdoor_scm():
  """A common SCM used for many of the unit tests."""
  var_names = ["U", "X", "Z", "Y"]
  cov_variables = ["U", "Z"]
  treatment_variable = "X"
  response_variable = "Y"

  # Define the parents
  parents = {
      "U": [],
      "X": ["U"],
      "Z": ["X"],
      "Y": ["Z", "U"],
  }
  p = np.random.uniform(.2, .8)
  beta = np.random.uniform(-1, 1)
  gamma = np.random.uniform(5, 10)
  alpha_1 = np.random.uniform(-.3, .3)
  alpha_2 = np.random.uniform(-.3, .3)

  def g(x):
    return 1 / (1 + np.exp(-x))

  mu_u = p
  var_u = p * (1 - p)
  mu_x = p * g(alpha_1) + (1 - p) / 2
  var_x = mu_x * (1 - mu_x)
  mu_z = mu_x * g(beta) + (1 - mu_x) / 2
  var_z = mu_z * (1 - mu_z)
  mu_y = gamma * mu_z + alpha_2 * mu_u
  tau = gamma * (g(beta) - .5)  # calculated by hand

  # Calculate the joint probabilities of X and u
  pz1u1 = (g(beta) * g(alpha_1) + .5 * (1 - g(alpha_1))) * p
  pz1u0 = (g(beta) / 2 + 1 / 4) * (1 - p)
  pz0u1 = ((1 - g(beta)) * g(alpha_1) + (1 - g(alpha_1)) / 2) * p
  # Take the expectation of Y^2 by hand.
  mu_y2 = 1 + (gamma**2 +
               alpha_2**2) * pz1u1 + gamma**2 * pz1u0 + alpha_2**2 * pz0u1
  var_y = mu_y2 - mu_y**2

  # populate the CPDs
  def y_cpd(_, parents):
    return (np.random.normal(size=len(parents[0])) + gamma * parents[0] +
            alpha_2 * parents[1])

  def x_cpd(n, parents):
    return np.random.binomial(1, g(alpha_1 * parents[0]), size=n)

  def z_cpd(n, parents):
    return np.random.binomial(1, g(beta * parents[0]), size=n)

  cpds = {
      "U": lambda n, parents: np.random.binomial(1, p, size=n),
      "X": x_cpd,
      "Z": z_cpd,
      "Y": y_cpd,
  }

  scm_gen = scm.SCM(
      name="frontdoor example",
      var_names=var_names,
      parents=parents,
      cpds=cpds,
      cov_variables=cov_variables,
      treatment_variable=treatment_variable,
      response_variable=response_variable,
  )
  return (scm_gen, {
      "tau": tau,
      "mu_z": mu_z,
      "var_z": var_z,
      "mu_x": mu_x,
      "var_x": var_x,
      "mu_y": mu_y,
      "var_y": var_y,
      "mu_u": mu_u,
      "var_u": var_u,
      "alpha_1": alpha_1,
      "alpha_2": alpha_2,
      "beta": beta,
      "gamma": gamma,
  })
