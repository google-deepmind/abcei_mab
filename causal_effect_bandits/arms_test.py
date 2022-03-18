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

from absl.testing import absltest
from causal_effect_bandits import arms
from causal_effect_bandits import data
from causal_effect_bandits import example_scm
from causal_effect_bandits import parameters
from causal_effect_bandits import test_utils
import numpy as np
from sklearn import linear_model


class ArmTest(absltest.TestCase):

  def test_frontdoor_arm(self):
    """Test the statistics of the distribution."""
    np.random.seed(0)
    scm_gen, args = example_scm.frontdoor_scm()

    n = 50000
    samples = scm_gen.generate(n)

    arm = arms.SampleSplittingArm(
        'test_arm',
        eta=parameters.FrontDoorLinearFiniteZ(
            n_features=1, min_ratio_to_uniform=5),
        # remove the U coordinate
        data_transformer=data.get_remove_coordinates_fn(0),
        ci_algorithm=arms.CIAlgorithm.FINITE_SAMPLE,
        sub_gscale=1,
        tau_bound=2,
        burn_in=1000,
        rho=1,
    )
    # Next, we run the arm on a sequence of data and make sure it approaches
    # a sensible value.
    arm_output = test_utils.run_single_arm(
        arm, data_gen=scm_gen, max_samples=10000, increment=1500)

    def g(x):
      return 1 / (1 + np.exp(-x))

    # Set the nuisance parameters to their true values
    arm._eta.exp_prob = args['mu_x']
    beta = args['beta']
    one = np.ones(1, dtype=int)  # Tabular must have matching d-types
    zero = np.zeros(1, dtype=int)

    arm._eta.cov_given_exp.table[one.tobytes()][one.tobytes()] = g(beta)
    arm._eta.cov_given_exp.table[one.tobytes()][zero.tobytes()] = 1 - g(beta)
    arm._eta.cov_given_exp.table[zero.tobytes()][one.tobytes()] = .5
    arm._eta.cov_given_exp.table[zero.tobytes()][zero.tobytes()] = .5

    if isinstance(arm._eta.y_response, linear_model.Ridge):
      arm._eta.y_response.coef_ = np.array([args['alpha_2'], args['gamma']])
      arm._eta.y_response.intercept_ = args['alpha_2'] * args['mu_u']

    var_approx = np.var(
        arm._eta.calculate_score(data.get_remove_coordinates_fn(0)(samples)))

    # Check estimators
    np.testing.assert_allclose(arm.var_est, var_approx, rtol=.05)

    # Check the LCB is less than the var_estimate
    np.testing.assert_array_less(arm_output['LCBs'],
                                 arm_output['var_estimates'])

    # Check the UCB is greater than the var_estimate
    np.testing.assert_array_less(arm_output['var_estimates'],
                                 arm_output['UCBs'])

    # Check the number of samples is correct
    np.testing.assert_equal(arm_output['n_samples'], 1500 * np.arange(1, 8))

  def test_back_door_arm(self):
    np.random.seed(0)
    scm_gen, _ = example_scm.back_door_scm()

    arm = arms.SampleSplittingArm(
        'test_arm',
        eta=parameters.AIPWLinearLogistic(n_features=1),
        data_transformer=data.get_identity_fn(),
        ci_algorithm=arms.CIAlgorithm.FINITE_SAMPLE,
        sub_gscale=1,
        tau_bound=10,
        burn_in=1000,
        rho=1,
    )
    # Next, we run the arm on a sequence of data and make sure it approaches
    # a sensible value.
    arm_output = test_utils.run_single_arm(
        arm, data_gen=scm_gen, max_samples=10000, increment=1500)

    # Check the LCB is less than the var_estimate
    np.testing.assert_array_less(arm_output['LCBs'],
                                 arm_output['var_estimates'])

    # Check the UCB is greater than the var_estimate
    np.testing.assert_array_less(arm_output['var_estimates'],
                                 arm_output['UCBs'])

    # Check the number of samples is correct
    np.testing.assert_equal(arm_output['n_samples'], 1500 * np.arange(1, 8))


if __name__ == '__main__':
  absltest.main()
