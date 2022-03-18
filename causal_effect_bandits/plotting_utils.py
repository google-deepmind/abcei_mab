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
"""Utilities for plotting."""
from typing import List, Optional
from causal_effect_bandits import arms
from causal_effect_bandits import bandits
import matplotlib.pyplot as plt
import numpy as np


def plot_bandit_results(ax,
                        arm_list: List[arms.VarianceEstimatorArm],
                        bdata: bandits.BanditData,
                        initial_points_to_ignore: Optional[int] = 0):
  """Plots a collection of variance estimates and confidence intervals.

  Both are provided as dictionaries with List[float] values and
  VarianceEstimatorArm obects ans keys.

  Args:
    ax: matplotlib axis object
    arm_list: a list of varianceEstimatorArm objects to plot.
    bdata: bandits.BanditData, as returned by BanditAlgorithm.run()
    initial_points_to_ignore: number of initial rounds of the algorithm to omit
      from the plot.

  Returns:
    matplotlib axis object.
  """
  for arm in arm_list:
    color = "tab:blue"
    num_samples = bdata.cum_samples[initial_points_to_ignore:]
    mid = bdata.var_est_by_arm[arm][initial_points_to_ignore:]
    lower = bdata.lcb_by_arm[arm][initial_points_to_ignore:]
    upper = bdata.ucb_by_arm[arm][initial_points_to_ignore:]

    ax.plot(num_samples, mid, label=arm.name)
    ax.plot(num_samples, lower, color=color, alpha=0.1)
    ax.plot(num_samples, upper, color=color, alpha=0.1)
    ax.fill_between(num_samples, lower, upper, alpha=0.2)

  ax.set_xlabel("Samples")
  ax.set_ylabel("Variance")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.legend(loc="upper right")
  return ax


def plot_3d_function(f, x_low, x_high, y_low, y_high):
  """Generates a 3d plot of f."""
  n_points = 100

  x = np.linspace(x_low, x_high, n_points)
  y = np.linspace(y_low, y_high, n_points)
  x, y = np.meshgrid(x, y)

  z = f(np.c_[x.flatten(), y.flatten()])
  z = np.reshape(z, (n_points, n_points))

  ax = plt.axes(projection="3d")
  ax.plot_surface(
      x, y, z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
