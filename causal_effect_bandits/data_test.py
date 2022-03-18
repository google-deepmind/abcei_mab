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
"""A unit test for the back_door formula."""
from absl.testing import absltest
from causal_effect_bandits import data
from causal_effect_bandits import example_scm
import numpy as np


class ExpDataTest(absltest.TestCase):

  def test_k_fold(self):
    z = np.arange(8).reshape(4, 2)
    x = np.arange(4) - 2
    y = np.arange(4) * 2
    new_data = data.ExpData(z, x, y)
    folds = new_data.k_fold(1)
    assert len(folds) == 1
    np.testing.assert_array_equal(folds[0].cov, new_data.cov)
    np.testing.assert_array_equal(folds[0].exp, new_data.exp)
    np.testing.assert_array_equal(folds[0].rsp, new_data.rsp)

    np.random.seed(0)
    folds = new_data.k_fold(2)
    assert len(folds) == 2
    np.testing.assert_array_equal(folds[0].cov, [[0, 1], [6, 7]])
    np.testing.assert_array_equal(folds[1].cov, [[2, 3], [4, 5]])
    np.testing.assert_array_equal(folds[0].exp, [-2, 1])
    np.testing.assert_array_equal(folds[1].exp, [-1, 0])
    np.testing.assert_array_equal(folds[0].rsp, [0, 6])
    np.testing.assert_array_equal(folds[1].rsp, [2, 4])


class TestTabularCPD(absltest.TestCase):

  def test_fit(self):
    table = data.TabularCPD(10)
    x = np.array([0, 1, 2] * 3)
    z = np.array([3] * 3 + [4] * 3 + [5] * 3)
    table.fit(x, z)

    np.testing.assert_array_equal(table.x_support(), [0, 1, 2])
    np.testing.assert_array_equal(table.z_support(), [3, 4, 5])
    for xi in [0, 1, 2]:
      for zi in [3, 4, 5]:
        assert table.predict([xi], [zi]) == 1 / 3.0


class SCMTest(absltest.TestCase):

  def test_scm_frontdoor_statistics(self):
    """Tests scm generating by checking statistics."""
    (scm_gen, params) = example_scm.frontdoor_scm()
    n = 100000
    new_data = scm_gen.generate(n)

    tol = 4 * max(new_data.rsp)**2 / np.sqrt(n)

    ## Test the statistics of the distribution

    # check means
    np.testing.assert_allclose(np.mean(new_data.exp), params["mu_x"], atol=tol)
    np.testing.assert_allclose(np.mean(new_data.rsp), params["mu_y"], atol=tol)
    np.testing.assert_allclose(
        np.mean(new_data.cov[:, 1]), params["mu_z"], atol=tol)
    np.testing.assert_allclose(
        np.mean(new_data.cov[:, 0]), params["mu_u"], atol=tol)

    # check variances
    np.testing.assert_allclose(
        np.var(new_data.cov[:, 0]), params["var_u"], atol=tol * params["var_u"])
    np.testing.assert_allclose(
        np.var(new_data.cov[:, 1]), params["var_z"], atol=tol * params["var_z"])
    np.testing.assert_allclose(
        np.var(new_data.exp), params["var_x"], atol=tol * params["var_x"])
    np.testing.assert_allclose(
        np.var(new_data.rsp), params["var_y"], atol=4 * tol * params["var_y"])

  def test_scm_back_door_statistics(self):
    (scm_gen, params) = example_scm.back_door_scm()

    # Test the statistics of the distribution
    n = 1000000
    new_data = scm_gen.generate(n)
    tol = np.mean(new_data.rsp**2) / np.sqrt(n)

    assert params["tau"] == 10

    # check means
    np.testing.assert_allclose(np.mean(new_data.exp), params["mu_x"], tol)
    np.testing.assert_allclose(np.mean(new_data.rsp), params["mu_y"], tol)
    np.testing.assert_allclose(np.mean(new_data.cov), params["mu_z"], tol)

    # check variances
    np.testing.assert_allclose(
        np.var(new_data.exp), params["var_x"], atol=tol * params["var_x"])
    np.testing.assert_allclose(
        np.var(new_data.cov), params["var_z"], atol=tol * params["var_z"])
    np.testing.assert_allclose(
        np.var(new_data.rsp), params["var_y"], atol=2 * tol * params["var_y"])


if __name__ == "__main__":
  absltest.main()
