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
"""Contains utilities needed to generate the paper's plots."""
from typing import List, Optional, Tuple
import warnings
from causal_effect_bandits import arms
from causal_effect_bandits import bandits
from causal_effect_bandits import data
from causal_effect_bandits import nonlinear_utils
from causal_effect_bandits import parameters
from causal_effect_bandits import scm
import numpy as np
import sklearn as sk

warnings.filterwarnings('ignore', category=sk.exceptions.ConvergenceWarning)


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


def get_section_5_1_cpds(
    z_dim: List[int],
    v_dim: List[int],
    num_back_door_paths: int,
    seed: Optional[int] = None,
):
  """Returns the CPDs needed by this example.

  Args:
    z_dim: the dimension of Z
    v_dim: the dimension of V
    num_back_door_paths: the number of back_door paths, i.e. M
    seed: the random seed

  Returns:
    A dictionary of functions, each representing a conditional
    distsribution, indexed by the variable name strings.
  """
  rng = np.random.default_rng(seed)

  # Include a zero first index corresponding to z_0 and a placeholder for V_0.
  # This fixes the indices so, e.g. mu_Vi = mu_V[i], beta_i = beta[i].
  mu_v = [0]
  cov_v = [0]  # V_0 does not exist
  mu_z = [.2 * rng.standard_normal(size=z_dim[0])]
  cov_z = [.5 * np.identity(z_dim[0])]
  alpha = [0]
  gamma = []
  mu_y = 0
  cov_y = .1

  beta = [.2 * rng.standard_normal(size=(z_dim[0]))]
  for i in range(1, num_back_door_paths + 1):
    strength = rng.uniform(.1, 1.9)
    # V
    mu_v.append(0 * rng.standard_normal(size=v_dim[i]))
    cov_v.append(1 * np.identity(v_dim[i]))
    # Z_i
    mu_z.append(0 * rng.standard_normal(size=z_dim[i]))
    cov_z.append(.1 * np.identity(z_dim[i]))
    # Vi->X
    gamma.append(strength * rng.standard_normal(size=v_dim[i]))
    # Vi->Zi
    alpha.append(.5 * strength * rng.standard_normal(size=(v_dim[i], z_dim[i])))
    ## Zi->Y
    beta.append((2 - strength) * rng.standard_normal(size=(z_dim[i])))

  beta.append((1 - strength) * rng.standard_normal(size=(z_dim[i])))

  # Next, we define the CPDs
  def x_cpd(n, parents):
    del n
    return scm.logistic_linear_bernoulli_cpd(parents, gamma)

  def y_cpd(n, parents):
    del n
    return scm.normal_cpd(parents, beta, mean=mu_y, cov=cov_y)

  cpds = {
      'X': x_cpd,
      'Y': y_cpd,
  }

  def make_v_cpd(mean, cov):

    def f(n, parents):
      del parents  # unusued
      return np.random.multivariate_normal(mean=mean, cov=cov, size=n)

    return f

  def make_z_cpd(alpha, mean, cov):

    def f(n, parents):
      del n  # unused
      return scm.normal_cpd(parents, [alpha], mean, cov)

    return f

  for i in range(1, num_back_door_paths + 1):
    cpds['V' + str(i)] = make_v_cpd(mu_v[i], cov_v[i])
    cpds['Z' + str(i)] = make_z_cpd(alpha[i], mu_z[i], cov_z[i])

  ## Z_0
  support_size = 10  # the number of values Z_0 may have
  z0_support = rng.normal(loc=mu_z[0], scale=1, size=(support_size, z_dim[0]))

  tau = 0
  tau_max = 0
  # To select a tau on the larger side, we randomly sample 10 of them then
  # choose the first tau that is larger. See the secretary problem.
  for _ in range(10):
    # Generate two random categorical distributions for Z0
    p_z0x0 = softmax(.4 * rng.standard_normal(size=support_size))
    p_z0x1 = softmax(.4 * rng.standard_normal(size=support_size))
    tau_max = max(
        tau,
        p_z0x1.dot(z0_support).dot(beta[0]) -
        p_z0x0.dot(z0_support).dot(beta[0]))
  while tau < tau_max:  # make sure tau is big enough
    # Generate two random categorical distributions for Z0
    p_z0x0 = softmax(.4 * rng.standard_normal(size=support_size))
    p_z0x1 = softmax(.4 * rng.standard_normal(size=support_size))
    tau = p_z0x1.dot(z0_support).dot(beta[0]) - p_z0x0.dot(z0_support).dot(
        beta[0])

  # X->Z0
  def z0x0_cpd(n, parents):
    del parents
    idx = np.random.choice(support_size, size=n, p=p_z0x0)
    return np.array([z0_support[int(i)] for i in idx])

  def z0x1_cpd(n, parents):
    del parents
    idx = np.random.choice(support_size, size=n, p=p_z0x1)
    return np.array([z0_support[int(i)] for i in idx])

  def z0_cpd(n, parents):
    del n
    return scm.categorical_conditioning(
        parents, 0, distributions=[z0x0_cpd, z0x1_cpd])

  cpds['Z0'] = z0_cpd

  return cpds


def get_section_5_2_cpds(
    z_dim: List[int],
    v_dim: List[int],
    num_back_door_paths: int,
    seed: Optional[int] = None,
):
  """Returns the CPDs needed by this example.

  Args:
    z_dim: the dimension of Z
    v_dim: the dimension of V
    num_back_door_paths: the number of back_door paths, i.e. M
    seed: the random seed

  Returns:
    A dictionary of functions, each representing a conditional
    distsribution, indexed by the variable name strings.
  """
  rng = np.random.default_rng(seed)

  # Include a zero first index corresponding to z_0 and a placeholder for V_0.
  # This fixes the indices so, e.g. mu_Vi = mu_V[i], beta_i = beta[i].
  mu_v = [0]
  cov_v = [0]  # V_0 does not exist
  mu_z = [.2 * rng.standard_normal(size=z_dim[0])]
  cov_z = [.5 * np.identity(z_dim[0])]
  alpha = [0]
  gamma = []
  mu_y = 0
  cov_y = .1

  beta = [.2 * rng.standard_normal(size=(z_dim[0]))]
  for i in range(1, num_back_door_paths + 1):
    strength = np.random.uniform(.1, 1.9)
    # V
    mu_v.append(0 * rng.standard_normal(size=v_dim[i]))
    cov_v.append(1 * np.identity(v_dim[i]))
    # Z_i
    mu_z.append(0 * rng.standard_normal(size=z_dim[i]))
    cov_z.append(.1 * np.identity(z_dim[i]))
    # Vi->X
    gamma.append(strength * rng.standard_normal(size=v_dim[i]))
    # Vi->Zi
    alpha.append(.5 * strength * rng.standard_normal(size=(v_dim[i], z_dim[i])))
    ## Zi->Y
    beta.append((2 - strength) * rng.standard_normal(size=(z_dim[i])))

  beta.append((1 - strength) * rng.standard_normal(size=(z_dim[i])))

  ## Z_0
  support_size = 10  # the number of values Z_0 may have
  z0_support = rng.normal(loc=mu_z[0], scale=1, size=(support_size, z_dim[0]))

  tau = 0
  tau_max = 0
  # To select a tau on the larger side, we randomly sample 10 of them then
  # choose the first tau that is larger. See the secretary problem.
  for _ in range(10):
    # Generate two random categorical distributions for Z0
    p_z0x0 = softmax(.4 * rng.standard_normal(size=support_size))
    p_z0x1 = softmax(.4 * rng.standard_normal(size=support_size))
    tau_max = max(
        tau,
        p_z0x1.dot(z0_support).dot(beta[0]) -
        p_z0x0.dot(z0_support).dot(beta[0]))
  while tau < tau_max:  # make sure tau is big enough
    # Generate two random categorical distributions for Z0
    p_z0x0 = softmax(.4 * rng.standard_normal(size=support_size))
    p_z0x1 = softmax(.4 * rng.standard_normal(size=support_size))
    tau = p_z0x1.dot(z0_support).dot(beta[0]) - p_z0x0.dot(z0_support).dot(
        beta[0])

  # X->Z0
  def z0x0_cpd(n, parents):
    del parents
    idx = np.random.choice(support_size, size=n, p=p_z0x0)
    return np.array([z0_support[int(i)] for i in idx])

  def z0x1_cpd(n, parents):
    del parents
    idx = np.random.choice(support_size, size=n, p=p_z0x1)
    return np.array([z0_support[int(i)] for i in idx])

  def z0_cpd(n, parents):
    del n
    return scm.categorical_conditioning(
        parents, 0, distributions=[z0x0_cpd, z0x1_cpd])

  # Specify Gaussian Process
  kernel = sk.gaussian_process.kernels.RBF(
      length_scale=1, length_scale_bounds=(1e-1, 1000.0))

  f_ys = []
  for dim in z_dim:
    f_ys.append(
        nonlinear_utils.generate_random_nd_function(
            dim=dim,
            out_dim=1,
            kernel=kernel,
            min_x=-2,
            max_x=2,
            n_points=10,
            alpha=1e0))

  # f_y should have shape (n_units,1)
  f_y = lambda x: np.sum([f(x[i]) for i, f in enumerate(f_ys)], axis=0)

  f_xs = []
  for dim in v_dim[1:]:
    f_xs.append(
        nonlinear_utils.generate_random_nd_function(
            dim=dim,
            out_dim=1,
            kernel=kernel,
            min_x=-2,
            max_x=2,
            n_points=10,
            alpha=1e0))

  # f_x should have shape (n_units,1)
  f_x = lambda x: np.sum([f(x[i]) for i, f in enumerate(f_xs)], axis=0)

  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  def bernoulli(x):
    return np.random.binomial(1, x)

  add_noise = scm.add_gaussian_noise
  # Next, we define the CPDs
  cpds = {
      'X': lambda n, parents: bernoulli(sigmoid(f_x(parents))).flatten(),
      'Y': lambda n, parents: add_noise(f_y(parents), mean=mu_y, cov=cov_y),
  }
  cpds['Z0'] = z0_cpd

  def make_v_cpd(mean, cov):

    def f_v(n, parents):
      del parents  # unusued
      return np.random.multivariate_normal(mean=mean, cov=cov, size=n)

    return f_v

  def make_z_cpd(f, mean, cov):

    def f_z(n, parents):
      del n  # unused
      return scm.structural_equation_with_noise(
          parents, f=f, mean=mean, cov=cov)

    return f_z

  for i in range(1, num_back_door_paths + 1):
    f_z = nonlinear_utils.generate_random_nd_function(
        dim=v_dim[i],
        out_dim=z_dim[i],
        kernel=kernel,
        min_x=-1,
        max_x=1,
        n_points=10,
        alpha=1e0)
    cpds['V' + str(i)] = make_v_cpd(mean=mu_v[i], cov=cov_v[i])
    cpds['Z' + str(i)] = make_z_cpd(f=f_z, mean=mu_z[i], cov=cov_z[i])

  return cpds


def get_section_5_1_example(
    z_dim: List[int],
    v_dim: List[int],
    seed: int,
    z_cost: float,
    v_cost: float,
    z0_cost: float,
    num_back_door_paths: int,
    sample_limit: int,
    units_per_round: int,
    latex_names=False,
) -> Tuple[scm.SCM, List[arms.VarianceEstimatorArm], bandits.LUCBAlgorithm,
           bandits.SuccessiveEliminationAlgorithm, bandits.UniformAlgorithm,]:
  """Returns a random instance of an SCM from Section 5.1.

  Args:
    z_dim: the dimension of Z
    v_dim: the dimension of V
    seed: the random seed
    z_cost: the cost of observing a Z_i
    v_cost: the cost of observing a V_i
    z0_cost: the cost of observing Z_0, the frontdoor variable.
    num_back_door_paths: the number of back_door paths, i.e. M-1.
    sample_limit: the total number of samples before termination
    units_per_round: the number of units samples every round for each algorithm.
    latex_names: whether we should include latex-friendly names for the arms.

  Returns:
    a scm.SCM data generator as described in Section 5.1.
    arm_list: a list of VarianceEstimatorArms
    lucb_bandit: a LUCBAlgorithm over the arms
    se_bandit: a SuccessiveEliminationAlgorithm over the arms
    uniform_bandit: a UniformAlgorithm over the arms
  """
  cpd_dict = get_section_5_1_cpds(z_dim, v_dim, num_back_door_paths, seed)
  return generate_example_from_cpd(
      z_dim,
      v_dim,
      z_cost,
      v_cost,
      z0_cost,
      num_back_door_paths,
      cpd_dict,
      sample_limit,
      units_per_round,
      latex_names,
      arms.CIAlgorithm.FINITE_SAMPLE,
  )


def get_section_5_2_example(
    z_dim: List[int],
    v_dim: List[int],
    seed: int,
    z_cost: float,
    v_cost: float,
    z0_cost: float,
    num_back_door_paths: int,
    sample_limit: int,
    units_per_round: int,
    latex_names=False,
) -> Tuple[scm.SCM, List[arms.VarianceEstimatorArm], bandits.LUCBAlgorithm,
           bandits.SuccessiveEliminationAlgorithm, bandits.UniformAlgorithm,]:
  """Returns a random instance of an SCM from Section 5.2.

  Args:
    z_dim: the dimension of Z
    v_dim: the dimension of V
    seed: the random seed
    z_cost: the cost of observing a Z_i
    v_cost: the cost of observing a V_i
    z0_cost: the cost of observing Z_0, the frontdoor variable.
    num_back_door_paths: the number of back_door paths, i.e. M-1.
    sample_limit: the total number of samples before termination
    units_per_round: the number of units samples every round for each algorithm.
    latex_names: whether we should include latex-friendly names for the arms.

  Returns:
    a scm.SCM data generator as described in Section 5.2.
    arm_list: a list of VarianceEstimatorArms
    lucb_bandit: a LUCBAlgorithm over the arms
    se_bandit: a SuccessiveEliminationAlgorithm over the arms
    uniform_bandit: a UniformAlgorithm over the arms
  """
  cpd_dict = get_section_5_2_cpds(z_dim, v_dim, num_back_door_paths, seed)
  return generate_example_from_cpd(
      z_dim,
      v_dim,
      z_cost,
      v_cost,
      z0_cost,
      num_back_door_paths,
      cpd_dict,
      sample_limit,
      units_per_round,
      latex_names,
      arms.CIAlgorithm.CLT,
  )


def generate_example_from_cpd(
    z_dim: List[int],
    v_dim: List[int],
    z_cost: float,
    v_cost: float,
    z0_cost: float,
    num_back_door_paths: int,
    cpd_dict,
    sample_limit: int,
    units_per_round: int,
    latex_names: bool,
    ci_algorithm: arms.CIAlgorithm,
) -> Tuple[scm.SCM, List[arms.VarianceEstimatorArm], bandits.LUCBAlgorithm,
           bandits.SuccessiveEliminationAlgorithm, bandits.UniformAlgorithm]:
  """Returns a random instance of an SCM from Section 5.1.

  Args:
    z_dim: the dimension of Z
    v_dim: the dimension of V
    z_cost: the cost of observing a z_i
    v_cost: the cost of observing a V_i
    z0_cost: the cost of observing z_0, the frontdoor variable.
    num_back_door_paths: the number of back_door paths, i.e. M-1.
    cpd_dict: a dictionary of cpd functions.
    sample_limit: the total number of samples before termination
    units_per_round: the number of units samples every round for each algorithm.
    latex_names: whether we should include latex-friendly names for the arms.
    ci_algorithm: the algorithm used for the confidence intervals

  Returns:
    a scm.SCM data generator as described in Section 5.1.
    arm_list: a list of VarianceEstimatorArms
    lucb_bandit: a LUCBAlgorithm over the arms
    se_bandit: a SuccessiveEliminationAlgorithm over the arms
    uniform_bandit: a UniformAlgorithm over the arms
  """
  var_names = []
  parents = {
      'Z0': ['X'],
      'X': [],
      'Y': ['Z0'],
  }
  cov_variables = ['Z0']
  treatment_variable = 'X'
  response_variable = 'Y'

  for i in range(1, num_back_door_paths + 1):
    var_names.append('V' + str(i))
    var_names.append('Z' + str(i))
    cov_variables.append('V' + str(i))
    cov_variables.append('Z' + str(i))
    parents['V' + str(i)] = []
    parents['Z' + str(i)] = ['V' + str(i)]
    parents['Y'].append('Z' + str(i))
    parents['X'].append('V' + str(i))

  var_names.extend(['X', 'Z0', 'Y'])

  scm_gen = scm.SCM(
      'Section 5 SCM',
      var_names,
      parents,
      cpd_dict,
      cov_variables,
      treatment_variable,
      response_variable,
  )

  # creat lists of indices: Z_idx[i], V_idx[i] are the indices occupied by Z_i,
  # V_i, respectively.
  total_z_dim = sum(z_dim)
  z_idx = [list(range(z_dim[0]))]
  v_idx = [[]]
  cum_z_dim = np.cumsum(z_dim)
  cum_v_dim = np.cumsum(v_dim)

  for i in range(num_back_door_paths):
    z_idx.append(list(range(cum_z_dim[i], cum_z_dim[i + 1])))
    v_idx.append(
        list(range(total_z_dim + cum_v_dim[i], total_z_dim + cum_v_dim[i + 1])))
  # the mask for Z0
  select_z0_fn = data.get_coordinate_mask_fn(z_idx[0])

  def generate_selection_masks(n, z_cost, v_cost):
    if n == 1:
      return [z_idx[n], v_idx[n]], ['Z_1', 'V_1'], [z_cost, v_cost]
    masks, names, costs = generate_selection_masks(n - 1, z_cost, v_cost)
    masks_with_z = [np.r_[m, z_idx[n]] for m in masks]
    names_with_z = [name + 'Z_' + str(n) for name in names]
    costs_with_z = [z_cost + c for c in costs]
    masks_with_v = [np.r_[m, v_idx[n]] for m in masks]
    names_with_v = [name + 'V_' + str(n) for name in names]
    costs_with_v = [v_cost + c for c in costs]
    return (
        masks_with_z + masks_with_v,
        names_with_z + names_with_v,
        costs_with_z + costs_with_v,
    )

  (masks, names, costs) = generate_selection_masks(num_back_door_paths, z_cost,
                                                   v_cost)
  if latex_names:
    names = ['$' + n + '$' for n in names]

  selection_fns = [data.get_coordinate_mask_fn(mask) for mask in masks]
  # Calculate the dimension of the covariates
  dims = [len(mask) for mask in masks]

  # At this point, for arm i, we have
  # - fns[i] converts data generated by scm_gen (which includes all covariates)
  #   to data that is needed by the ith estimator (which only includes the
  #   necessary covariates).
  # - costs[i]: the observational cost
  # - name[i]: the name
  # - dims[i]: the dimension of the covariates

  d = scm_gen.generate(1000)
  sub_gscale = np.var(d.exp) + np.var(d.cov) + np.var(d.rsp)

  # Next, we define the arms, beginning with the frontdoor arm
  arm_list = [
      arms.SampleSplittingArm(
          'frontdoor',
          eta=parameters.FrontDoorLinearFiniteZ(
              n_features=z_dim[0],
              min_units=5,
              min_overlap=.25,
          ),
          data_transformer=select_z0_fn,
          ci_algorithm=ci_algorithm,
          cost=z0_cost,
          sub_gscale=sub_gscale,
          tau_bound=2,
          burn_in=1000,
          rho=1,
      )
  ]
  # And including all the back_door arms (skipping the first element that
  # corresponded to Z0.
  for fn, name, cost, dim in zip(selection_fns, names, costs, dims):
    arm_list.append(
        arms.SampleSplittingArm(
            name,
            eta=parameters.AIPWLinearLogistic(
                n_features=dim, min_units=5, min_overlap=.25),
            data_transformer=fn,
            ci_algorithm=ci_algorithm,
            cost=cost,
            sub_gscale=sub_gscale,
            tau_bound=2,
            burn_in=1000,
            rho=1,
        ))

  # And the three bandit algorithms are defined below.
  lucb_bandit = bandits.LUCBAlgorithm(
      arm_list,
      prob_model=scm_gen,
      error_prob=.05,
      sample_limit=sample_limit,
      units_per_round=units_per_round,
  )
  se_bandit = bandits.SuccessiveEliminationAlgorithm(
      arm_list,
      prob_model=scm_gen,
      error_prob=.05,
      sample_limit=sample_limit,
      units_per_round=units_per_round,
  )
  uniform_bandit = bandits.UniformAlgorithm(
      arm_list,
      prob_model=scm_gen,
      error_prob=.05,
      sample_limit=sample_limit,
      units_per_round=units_per_round,
  )

  return (
      scm_gen,
      arm_list,
      lucb_bandit,
      se_bandit,
      uniform_bandit,
  )
