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
"""A unit test for VarianceEstimator arms."""
from typing import Callable, Tuple

from absl.testing import absltest
from causal_effect_bandits import arms
from causal_effect_bandits import bandits
from causal_effect_bandits import data
from causal_effect_bandits import example_scm
import numpy as np


class ConstantArm(arms.VarianceEstimatorArm):
  """A simple arm used for testing the bandit algs.

  Always returns a constant var_est, and the CI width is width_fn(n),
  where n is the total number of samples.
  """

  def __init__(
      self,
      name: str,
      var_est: float,
      width_fn: Callable[[int], float],
  ):
    self._name = name
    self._var_est = var_est
    self.width_fn = width_fn
    self.n_samples = 0
    self._ci = (0, np.inf)
    self.cost = 1

  def update(self, new_data: data.ExpData, delta: float) -> Tuple[float, float]:
    self.n_samples += len(new_data)
    self._ci = (self.var_est - self.width_fn(self.n_samples),
                self.var_est + self.width_fn(self.n_samples))
    return self._ci


class BanditTest(absltest.TestCase):

  def test_uniform_bandit(self):
    """Tests the UniformAlgorithm bandit on easy data.

    This method verifies that the correct arms are pulled
    the correct number of times and that the confidence
    intervals are correctly accumulated.
    """
    scm_gen, _ = example_scm.frontdoor_scm()

    def width_fn(n):
      return np.sqrt(20) / np.sqrt(n)

    arm_list = [
        ConstantArm('arm1', var_est=1, width_fn=width_fn),
        ConstantArm('arm2', var_est=2, width_fn=width_fn),
        ConstantArm('arm3', var_est=2, width_fn=width_fn)
    ]

    bandit = bandits.UniformAlgorithm(
        arm_list,
        prob_model=scm_gen,
        confidence=.05,
        sample_limit=50,
        units_per_round=5,
    )
    bdata = bandit.run()

    # Verify the correct best arm is returned
    np.testing.assert_equal(bdata.best_arm, [arm_list[0]])

    # Test that the correct var estimates are returned
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[0]], [1] * 4)
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[1]], [2] * 4)
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[2]], [2] * 4)

    # Test that the correct number of samples are returned: every arm is
    # sampled every time
    correct_samples = [5] * 4
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[0]],
                               correct_samples)
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[1]],
                               correct_samples)
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[2]],
                               correct_samples)

    # Test that the lower confidence bounds are correct
    correct_width = [width_fn(n) for n in 5 * np.arange(1, 5)]
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[0]],
                               np.ones(4) - correct_width)
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[1]],
                               2 * np.ones(4) - correct_width)
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[2]],
                               2 * np.ones(4) - correct_width)

    # Test that the upper confidence bounds are correct
    correct_width = [width_fn(n) for n in 5 * np.arange(1, 5)]
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[0]],
                               np.ones(4) + correct_width)
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[1]],
                               2 * np.ones(4) + correct_width)
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[2]],
                               2 * np.ones(4) + correct_width)

  def test_successive_elimination_bandit(self):
    """Tests the SuccessiveEliminationAlgorithm bandit on easy data.

    This method verifies that the correct arms are pulled
    the correct number of times and that the confidence
    intervals are correctly accumulated.
    """
    scm_gen, _ = example_scm.frontdoor_scm()

    def width_fn(n):
      return np.sqrt(20) / np.sqrt(n)

    arm_list = [
        ConstantArm('arm1', var_est=1, width_fn=width_fn),
        ConstantArm('arm2', var_est=2.25, width_fn=width_fn),
        ConstantArm('arm3', var_est=3, width_fn=width_fn)
    ]

    bandit = bandits.SuccessiveEliminationAlgorithm(
        arm_list,
        prob_model=scm_gen,
        confidence=.05,
        sample_limit=100,
        units_per_round=10,
    )
    bdata = bandit.run()

    # Test that the correct best arm is returned
    np.testing.assert_equal(bdata.best_arm, [arm_list[0]])

    # Test that the variance estimates are correct
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[0]], [1] * 5)
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[1]], [2.25] * 5)
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[2]], [3] * 5)

    # Test that the correct number of samples are returned
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[0]], [10] * 5)
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[1]], [10] * 5)
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[2]],
                               [10, 10, 0, 0, 0])

    correct_width_1 = [width_fn(n) for n in 10 * np.arange(1, 6)]
    correct_width_2 = [width_fn(n) for n in 10 * np.arange(1, 6)]
    correct_width_3 = [width_fn(n) for n in [10, 20, 20, 20, 20]]
    # Test that the lower confidence bounds are correct
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[0]],
                               np.ones(5) - correct_width_1)
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[1]],
                               2.25 * np.ones(5) - correct_width_2)
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[2]],
                               3 * np.ones(5) - correct_width_3)

    # Test that the upper confidence bounds are correct
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[0]],
                               np.ones(5) + correct_width_1)
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[1]],
                               2.25 * np.ones(5) + correct_width_2)
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[2]],
                               3 * np.ones(5) + correct_width_3)

  def test_lucb_bandit(self):
    """Tests the LUCBAlgorithm bandit on easy data.

    This method verifies that the correct arms are pulled
    the correct number of times and that the confidence
    intervals are correctly accumulated.
    """

    scm_gen, _ = example_scm.frontdoor_scm()

    def width_fn(n):
      return np.sqrt(20) / np.sqrt(n)

    arm_list = [
        ConstantArm('arm1', var_est=1, width_fn=width_fn),
        ConstantArm('arm2', var_est=2, width_fn=width_fn),
        ConstantArm('arm3', var_est=2.5, width_fn=width_fn)
    ]

    bandit = bandits.LUCBAlgorithm(
        arm_list,
        prob_model=scm_gen,
        confidence=.05,
        sample_limit=50,
        units_per_round=5,
    )
    bdata = bandit.run()

    # Test that the correct best arm is returned
    np.testing.assert_equal(bdata.best_arm, [arm_list[0]])

    # Test that the variance estimates are correct
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[0]], [1] * 5)
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[1]], [2] * 5)
    np.testing.assert_allclose(bdata.var_est_by_arm[arm_list[2]], [2.5] * 5)

    # Test that the correct number of samples are returned
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[0]], [5] * 5)
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[1]],
                               [5, 5, 0, 5, 5])
    np.testing.assert_allclose(bdata.samples_by_arm[arm_list[2]],
                               [5, 0, 5, 0, 0])

    correct_width_1 = [width_fn(n) for n in 5 * np.arange(1, 6)]
    correct_width_2 = [width_fn(n) for n in [5, 10, 10, 15, 20]]
    correct_width_3 = [width_fn(n) for n in [5, 5, 10, 10, 10]]

    # Test that the lower confidence bounds are correct
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[0]],
                               np.ones(5) - correct_width_1)
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[1]],
                               2 * np.ones(5) - correct_width_2)
    np.testing.assert_allclose(bdata.lcb_by_arm[arm_list[2]],
                               2.5 * np.ones(5) - correct_width_3)

    # Test that the upper confidence bounds are correct
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[0]],
                               np.ones(5) + correct_width_1)
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[1]],
                               2 * np.ones(5) + correct_width_2)
    np.testing.assert_allclose(bdata.ucb_by_arm[arm_list[2]],
                               2.5 * np.ones(5) + correct_width_3)


if __name__ == '__main__':
  absltest.main()
