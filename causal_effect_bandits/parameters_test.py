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
from causal_effect_bandits import data
from causal_effect_bandits import example_scm
from causal_effect_bandits import parameters
import numpy as np


class EstimatorTest(absltest.TestCase):

  def test_frontdoorlinearfinitez(self):
    """Tests the FrontDoorLinearFiniteZ NuisanceParameter."""
    np.random.seed(0)
    scm_gen, params = example_scm.frontdoor_scm()
    n = 100000

    eta = parameters.FrontDoorLinearFiniteZ(
        n_features=1, min_ratio_to_uniform=5)

    # Now we check the data generation process and the estimators
    d = scm_gen.generate(n)
    eta.fit(data.get_remove_coordinates_fn(0)(d))

    tol = .1

    def g(x):
      return 1 / (1 + np.exp(-x))

    # Check the model is being fit correctly.
    one = np.ones(1, dtype=int)  # Tablular must have matching d-types
    zero = np.zeros(1, dtype=int)
    np.testing.assert_allclose(eta.exp_prob, params["mu_x"],
                               tol * params["mu_x"])
    np.testing.assert_allclose(
        eta.cov_given_exp.predict(x=one, z=one), g(params["beta"]), tol)
    np.testing.assert_allclose(
        eta.cov_given_exp.predict(x=one, z=zero), 1 - g(params["beta"]), tol)
    np.testing.assert_allclose(
        eta.cov_given_exp.predict(x=zero, z=one), .5, tol)
    np.testing.assert_allclose(
        eta.cov_given_exp.predict(x=zero, z=zero), .5, tol)

    np.testing.assert_allclose(
        eta.y_response.predict([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]), [
            0, params["gamma"], params["alpha_2"] * params["mu_u"],
            params["gamma"] + params["alpha_2"]
        ],
        atol=.2)

  def test_AIPWLinearLogistic(self):
    """Tests the AIPWLinearLogistic NuisanceParameter."""
    np.random.seed(0)
    scm_gen, params = example_scm.back_door_scm()
    n = 100000

    eta = parameters.AIPWLinearLogistic(n_features=1)

    # Now we check the data generation process and the estimators
    d = scm_gen.generate(n)
    eta.fit(d)
    tol = .1
    np.testing.assert_allclose(eta.get_response_parameters()[0],
                               [params["tau"], params["beta"]], tol)


if __name__ == "__main__":
  absltest.main()
