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
"""Implementations of bandit algorithms over Arms."""

import dataclasses
from typing import List, Mapping, Optional
from causal_effect_bandits import arms
from causal_effect_bandits import data
import numpy as np


@dataclasses.dataclass
class BanditData:
  """Stores the results of running a bandit algorithm.

  Attributes
    lcb_by_arm: dict(arm object, np.array)
      A list of the lcbs returned by of the arm for every round
    ucb_by_arm: dict(arm object, np.array)
      A list of the ucbs returned by of the arm for every round
    var_est_by_arm: dict(arm object, np.array)
      A list of the variance estimates returned by of the arm for every round.
      Is populated even when the arm is not updated.
    best_arm: List[arm]
      A list of arms with the lowest ucb of all the arms with the lowest
      estimate of the variance. Could be more than one.
    samples_by_arm: Mapping(arm object, np.array)
      a dict of the samples allocated to each arm during each period
    cum_samples: np.array
      a list of the cumulative number of samples used for all arms.
  """
  lcb_by_arm: Mapping[arms.VarianceEstimatorArm, List[float]]
  ucb_by_arm: Mapping[arms.VarianceEstimatorArm, List[float]]
  var_est_by_arm: Mapping[arms.VarianceEstimatorArm, List[float]]
  samples_by_arm: Mapping[arms.VarianceEstimatorArm, List[float]]
  cum_samples: List[float]
  best_arm: List[arms.VarianceEstimatorArm]


class BAIBanditAlgorithm:
  """A class for best arm identification in bandit problems.

  Parameters:
  _arms: a list of arms.VarianceEstimatorArms
    The arms that will be pulled by this algorithm

  _num_arms: int
    The number of arms.

  _prob_model: data.DataGenerator
    a way to generate i.i.d. (X, T, Y) data

  error_prob: float in (0,1)
    the desired probability of correctness (usually denoted delta)

  error_tol: float > 0
    the allowed suboptimality of the returned arm (usually denoted epsilon)

  confidence: float in (0,1)
    the desired probability of returning the best arm.
    Set to None if we want a fixed budget problem.

  sample_limit: int > 0
    the maximum number of rounds. Set to None if you want at fixed
    confidence problem.

  units_per_round: int
    The number of units sampled every banit round.

  Methods:
  run:
    runs the bandit until the completion criteria are met

  collect_bounds:
    Assembles the ci and var_est of the arms into three np.arrays.

  _check_fixed_conf_end_condition():
    checks if the end condition condition of a fixed confidence bandit are met

  _select_arm_with_mask(mask)
    given a List[boolean] with num_arms entries, this method returns a sublist
    of the arms where the corresponding entry in mask is True.

  """

  def __init__(
      self,
      arm_list: List[arms.VarianceEstimatorArm],
      prob_model: data.DataGenerator,
      error_prob: float = .05,
      error_tol: float = 0,
      confidence: float = .05,
      sample_limit=np.inf,
      units_per_round: int = 10,
  ):
    self._arms = arm_list
    self._num_arms = len(self._arms)
    self._prob_model = prob_model
    if not 0.0 < error_prob < 1.0:
      raise ValueError(
          f"error_prob={error_prob}; must be in the unit interval.")
    self.error_prob = error_prob
    if error_tol < 0.0:
      raise ValueError(f"error_tol={error_tol}; must be nonnegative.")
    self.error_tol = error_tol
    if confidence is None and np.isinf(sample_limit):
      raise ValueError(
          "Either the confidence or the number of rounds must be specified")
    if confidence is not None:
      if not 0 < confidence < 1:
        raise ValueError(
            f"confidence={confidence}; must be in the unit interval")

    self.confidence = confidence
    self.sample_limit = sample_limit
    self.units_per_round = units_per_round

  def reset(self):
    """Resets the bandit to a just-instantiated state."""
    for a in self._arms:
      a.reset()

  def _check_fixed_conf_end_condition(self):
    """Contains the logic to determine if the bandit algorithm is done.

    Uses the fixed confidence setting.

    The game is done if the ucb of the best arm (the lowest ucb of all the arms
    that have the minimum variance estimate) is lower than lcb of the remaining
    arms (all arms that do not share the minimum variance estimate).

    Returns:
      boolean: is the best arm (epsilon, delta)-PAC?
      best_arm: the best arm
    """
    if self.error_prob is None:  # we are a fixed budget bandit
      return (False, [])

    lcbs, ucbs, var_ests = self.collect_bounds()
    if any(np.isinf(ucbs)):  # one of the ucbs is infinity
      return (False, [])
    min_var = min(var_ests)
    if np.isnan(min_var):  # one of the arms returned a nan
      return (False, [])

    # find the arm with the smallest variance.
    min_var_arm = np.argmin(var_ests)
    # check to see if the lcb of the remaining arms is higher than the ucb
    # of min_var_arm.
    remaining_lcbs = [
        lcbs[i] for i in range(len(self._arms)) if i != min_var_arm
    ]
    return (ucbs[min_var_arm] < min(remaining_lcbs) - self.error_tol,
            [self._arms[min_var_arm]])

  def _select_arms_with_mask(self, boolean_mask):
    """Returns the arms where boolean_mask is true.

    Args:
      boolean_mask: a list of booleans

    Returns:
      the arms corresponding to the true positions in the mask.
    """
    assert len(boolean_mask) == self._num_arms
    return [a for (a, m) in zip(self._arms, boolean_mask) if m]

  def collect_bounds(self):
    """Collects the upper and lower confidence intervals and variance estimates.

    Returns:
      Lists of the lcbs, ucbs, and point estimates of the asymptotic variances.
    """
    ucbs = np.array([a.cost * a.ci[1] for a in self._arms])
    lcbs = np.array([a.cost * a.ci[0] for a in self._arms])
    sigma2s = np.array([a.cost * a.var_est for a in self._arms])
    return (lcbs, ucbs, sigma2s)


class UniformAlgorithm(BAIBanditAlgorithm):
  """Implements the trivial algorithm that just pulls all arms equally."""

  def run(self, all_data: Optional[data.ExpData] = None) -> BanditData:
    """Runs the bandit algorithm until termination.

    Terminations by happen if either the confidence is reached or
    sample_limit is reached.


    If all_data is passed, then we assume we should sample in stable order:
    every arm will sequentially read data from the beginning of data
    independently of the other arms or sample selection strategy.
    This ensures that the confidence sequences for an arm are identical across
    different algorithms, making a comparison easier.

    Args:
      all_data: optional data.ExpData specifying all the samples the algorithm
        could use. The algorithm could terminate before consuming all the data.
        If all_data is not passed, then data is generated from self.prob_model.

    Returns:
      A BanditData dataclass.
    """
    stable_order = all_data is not None

    # Allocate variables for tracking the history
    lcb_by_arm = {a: [] for a in self._arms}
    ucb_by_arm = {a: [] for a in self._arms}
    var_est_by_arm = {a: [] for a in self._arms}
    cum_samples = []
    samples_by_arm = {a: [] for a in self._arms}

    total_samples = 0

    while True:
      if stable_order:
        # All remaining arms have been pulled the same number of times.
        n_start = self._arms[0].get_num_units()
        new_samples = all_data[n_start:n_start + self.units_per_round]
      else:
        new_samples = self._prob_model.generate(self.units_per_round)

      # We need to calculate the necessary delta:
      for arm in self._arms:
        arm.update(new_samples, self.error_prob / len(self._arms))
        lcb_by_arm[arm].append(arm.cost * arm.ci[0])
        ucb_by_arm[arm].append(arm.cost * arm.ci[1])
        var_est_by_arm[arm].append(arm.cost * arm.var_est)
        samples_by_arm[arm].append(self.units_per_round)

      total_samples += self._num_arms * self.units_per_round
      cum_samples.append(total_samples)

      if total_samples > self.sample_limit:
        _, _, var_ests = self.collect_bounds()
        # Select the arms with the lowest var estimates.
        best_arms = self._select_arms_with_mask(
            [v == min(var_ests) for v in var_ests])
        break

      # Check end condition
      if self.error_prob is not None:  # This is a fixed confidence bandit
        (game_ends, best_arms) = self._check_fixed_conf_end_condition()
        if game_ends:
          break

    return BanditData(lcb_by_arm, ucb_by_arm, var_est_by_arm, samples_by_arm,
                      cum_samples, best_arms)


class LUCBAlgorithm(BAIBanditAlgorithm):
  """Implements the LUCB method."""

  def __init__(
      self,
      arm_list: List[arms.VarianceEstimatorArm],
      prob_model: data.DataGenerator,
      error_prob: float = .05,
      error_tol: float = 0,
      confidence=None,
      sample_limit=None,
      units_per_round=200,
  ):
    super().__init__(arm_list, prob_model, error_prob, error_tol, confidence,
                     sample_limit, units_per_round)

  def run(self, all_data: Optional[data.ExpData] = None) -> BanditData:
    """Runs the bandit algorithm until termination.

    Terminations by happen if either the confidence is reached or
    sample_limit is reached.


    If all_data is passed, then we assume we should sample in stable order:
    every arm will sequentially read data from the beginning of data
    independently of the other arms or sample selection strategy.
    This ensures that the confidence sequences for an arm are identical across
    different algorithms, making a comparison easier.

    Args:
      all_data: optional data.ExpData specifying all the samples the algorithm
        could use. The algorithm could terminate before consuming all the data.
        If all_data is not passed, then data is generated from self.prob_model.

    Returns:
      lcb_by_arm: dict(arm object, np.array)
        A list of the lcbs returned by of the arm for every round
      ucb_by_arm: dict(arm object, np.array)
        A list of the ucbs returned by of the arm for every round
      var_est_by_arm: dict(arm object, np.array)
        A list of the variance estimates returned by of the arm for every round
      best_arm: List[arm]
        A list of arms with the lowest ucb of all the arms with the lowest
        estimate of the variance. Could be more than one.
      samples_by_arm: Mapping(arm object, np.array)
        a dict of the samples allocated to each arm during each period
      cum_samples: np.array
        a list of the cumulative number of samples used
    """
    stable_order = all_data is not None

    if stable_order:
      new_samples = all_data[:self.units_per_round]
    else:
      new_samples = self._prob_model.generate(self.units_per_round)

    # Allocate variables for tracking the history
    lcb_by_arm = {a: [] for a in self._arms}
    ucb_by_arm = {a: [] for a in self._arms}
    var_est_by_arm = {a: [] for a in self._arms}
    cum_samples = []
    samples_by_arm = {a: [] for a in self._arms}

    # Update all arms once
    total_samples = self._num_arms * self.units_per_round
    for arm in self._arms:
      arm.update(new_samples, self.error_prob / len(self._arms))
      lcb_by_arm[arm].append(arm.cost * arm.ci[0])
      ucb_by_arm[arm].append(arm.cost * arm.ci[1])
      var_est_by_arm[arm].append(arm.cost * arm.var_est)
      samples_by_arm[arm].append(self.units_per_round)

    cum_samples.append(total_samples)

    while True:
      if total_samples > self.sample_limit:
        _, _, var_ests = self.collect_bounds()
        # Select the arms with the lowest var estimates.
        best_arm = self._select_arms_with_mask(
            [v == min(var_ests) for v in var_ests])
        break

      arms_to_update = self.select_arms()

      for arm in arms_to_update:
        if not stable_order:
          new_samples = self._prob_model.generate(self.units_per_round)
        else:
          n_start = arm.get_num_units()
          new_samples = all_data[n_start:n_start + self.units_per_round]
        arm.update(new_samples, self.error_prob / len(self._arms))

      # Record the arm outputs
      for arm in self._arms:
        lcb_by_arm[arm].append(arm.cost * arm.ci[0])
        ucb_by_arm[arm].append(arm.cost * arm.ci[1])
        var_est_by_arm[arm].append(arm.cost * arm.var_est)
        samples_by_arm[arm].append(self.units_per_round if arm in
                                   arms_to_update else 0)

      total_samples += 2 * self.units_per_round
      cum_samples.append(total_samples)

      # Check end condition

      if self.error_prob is not None:  # This is a fixed confidence bandit
        (game_ends, best_arm) = self._check_fixed_conf_end_condition()
        if game_ends:
          break

    return BanditData(lcb_by_arm, ucb_by_arm, var_est_by_arm, samples_by_arm,
                      cum_samples, best_arm)

  def select_arms(self) -> List[arms.VarianceEstimatorArm]:
    """Picks the arms to sample next.

    For LUCB, we choose between the arm with the lowest upper confidence bound,
    or the arm among the remaining arms with the highest lower confidence bound.

    Returns:
      A list of arms selected by the algorithm.
    """
    lcbs, ucbs, var_ests = self.collect_bounds()
    if any([v is None for v in var_ests]):
      return np.random.choice(self._arms, 2, replace=False)  # choose at random
    lowest_var = min(var_ests)
    if np.isnan(lowest_var):
      print(
          "Warning: some of the confidence intervals are nan; choosing the next arms uniformly at random"
      )
      return np.random.choice(self._arms, size=2, replace=False)

    best_arm_mask = [v == lowest_var for v in var_ests]

    # Select the arm in this set with the highest ucb
    high_ucb = max(ucbs[best_arm_mask])
    high_ucb_mask = [u == high_ucb for u in ucbs]

    # Also select the arm with the lowest lcb of the remaining arms
    low_lcb = min(lcbs[[not i for i in best_arm_mask]])
    low_lcb_mask = [l == low_lcb for l in lcbs]

    # We will pick between these two arms
    u_index = np.random.choice(np.arange(self._num_arms)[high_ucb_mask])
    l_index = np.random.choice(np.arange(self._num_arms)[low_lcb_mask])
    return [self._arms[u_index], self._arms[l_index]]


class SuccessiveEliminationAlgorithm(BAIBanditAlgorithm):
  """Implements the successive elimination algorithm.

  Attributes:
    arms: list of arms.VarianceEstimatorArm objects
    prob_model: data.DataGenerator object
    sample_limit: int an upper bound on the total number of samples
    units_per_round: int the number of units to sample for every epoch of the
      algorithm
    stable_order: whether the algorithm reads the data sequentially or not.
  """

  def __init__(
      self,
      arm_list: List[arms.VarianceEstimatorArm],
      prob_model: data.DataGenerator,
      error_prob=.05,
      error_tol=0,
      confidence=None,
      sample_limit=None,
      units_per_round=200,
  ):
    super().__init__(arm_list, prob_model, error_prob, error_tol, confidence,
                     sample_limit, units_per_round)

  def run(self, all_data: Optional[data.ExpData] = None):
    """Runs the bandit algorithm until termination.

    Terminations by happen if either the confidence is reached or
    sample_limit is reached.


    If all_data is passed, then we assume we should sample in stable order:
    every arm will sequentially read data from the beginning of data
    independently of the other arms or sample selection strategy.
    This ensures that the confidence sequences for an arm are identical across
    different algorithms, making a comparison easier.

    Args:
      all_data: optional data.ExpData specifying all the samples the algorithm
        could use. The algorithm could terminate before consuming all the data.
        If all_data is not passed, then data is generated from self.prob_model.

    Returns:
      lcb_by_arm: dict(arm object, np.array)
        A list of the lcbs returned by of the arm for every round
      ucb_by_arm: dict(arm object, np.array)
        A list of the ucbs returned by of the arm for every round
      var_est_by_arm: dict(arm object, np.array)
        A list of the variance estimates returned by of the arm for every round
      best_arm: List[arm]
        A list of arms with the lowest ucb of all the arms with the lowest
        estimate of the variance. Could be more than one.
      samples_by_arm: Mapping(arm object, np.array)
        a dict of the samples allocated to each arm during each period
      cum_samples: np.array
        a list of the cumulative number of samples used
    """
    stable_order = all_data is not None

    # Allocate variables for tracking the history
    lcb_by_arm = {a: [] for a in self._arms}
    ucb_by_arm = {a: [] for a in self._arms}
    var_est_by_arm = {a: [] for a in self._arms}
    cum_samples = []
    samples_by_arm = {a: [] for a in self._arms}
    total_samples = 0

    # Initialize
    candidate_arms = list(self._arms)  # copy arms.
    epoch = 1
    while len(candidate_arms) > 1:
      epoch += 1
      if total_samples > self.sample_limit:
        _, _, var_ests = self.collect_bounds()
        # Select the arms with the lowest var estimates.
        candidate_arms = self._select_arms_with_mask(
            [v == min(var_ests) for v in var_ests])
        break

      for arm in candidate_arms:
        if not stable_order:
          new_samples = self._prob_model.generate(self.units_per_round)
        else:
          n_start = arm.get_num_units()
          new_samples = all_data[n_start:n_start + self.units_per_round]
        arm.update(new_samples, self.error_prob / len(self._arms))

      total_samples += len(candidate_arms) * self.units_per_round

      # Record the arm outputs
      cum_samples.append(total_samples)
      for arm in self._arms:
        lcb_by_arm[arm].append(arm.cost * arm.ci[0])
        ucb_by_arm[arm].append(arm.cost * arm.ci[1])
        var_est_by_arm[arm].append(arm.cost * arm.var_est)
        samples_by_arm[arm].append(0)

      for arm in candidate_arms:
        samples_by_arm[arm][-1] = self.units_per_round

      # Now we decide the ending condition
      # trim down the candidate set by finding the lowest ucb of the arms
      # with the minimum var_est.
      candidate_var_ests = [var_est_by_arm[arm][-1] for arm in candidate_arms]
      lowest_var = min(candidate_var_ests)
      low_var_mask = [v == lowest_var for v in candidate_var_ests]
      lowest_ucb = min([
          ucb_by_arm[arm][-1]
          for i, arm in enumerate(candidate_arms)
          if low_var_mask[i]
      ])

      # We now eliminate all arms with lcbs that are larger than lowest_ucb.
      new_candidate_arms = []
      for arm in candidate_arms:
        if lcb_by_arm[arm][-1] < lowest_ucb:
          new_candidate_arms.append(arm)
      candidate_arms = new_candidate_arms

      # Check end condition
      if len(candidate_arms) == 1:
        break

    return BanditData(lcb_by_arm, ucb_by_arm, var_est_by_arm, samples_by_arm,
                      cum_samples, candidate_arms)
