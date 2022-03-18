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
"""Contains the class definitions for the estimators."""

from typing import Tuple

from absl import logging
from causal_effect_bandits import data
import numpy as np
import sklearn
from sklearn import linear_model
from typing_extensions import Protocol


class NuisanceParameter(Protocol):
  """Encodes the interface for nuisance parameters.

  This class contains all the code to fit, model, and compute a nuisance
  parameter from data, its influence function, and the ATE from it.

  The definition of eta and phi depend on the structure of the
    nuisance parameter, so we encapulate them together.

  Attributes:
    _nus_dim: int the output dimension of the nuisance parameter
    _cov_dim: int the dimension of the covariates
    _model: object
      models of the nuisance parameters (eta). Maps from:
      from: ExpData: n_units * (_cov_dim + treatment_dim + response_dim)
      to: n_units * _nus_dim
    _data: ExpData the last ExpData that we used
    _phi: function
      the phi function: accepts n_units * _nus_dim, returns n_units * 1
    _has_been_fit: boolean True if the model has been fit on enough data to make
      a sensible prediction, e.g. both control and treatment groups have units
      in them.
    recent_data: ExpData The most recently fit data.
  Methods to be fit by sub classes:
    fit: fits the model using an ExpData object.
    transform: returns an element-wise nuisance parameter evaluated on ExpData
    calculate_score: returns elementwise phi(calculate_score)
    has_enough_samples: True if the model has seen enough data to be fit.
    reset: resets all the models to their default state
    set_to-truth: sets the nuisance parameters to their true value.
  """

  def fit(self, d: data.ExpData) -> None:
    """Fits eta to the provided experimental data.

    Args:
      d: ExpData used to fit the nuisance parameter. Expected shapes are
        d.cov: (n_units, n_features)
        d.exp: (n_units,)
        d.rsp: (n_units,)
    """
    ...

  def reset(self) -> None:
    ...

  def _phi(self, eta: np.ndarray, d: data.ExpData) -> np.ndarray:
    """Calculates phi(W, eta) for all W in d and eta(W).

    Args:
      eta: np.ndarray of shape (data.n_units, _nus_dim), which has been
        evaluated on W.
      d: ExpData of shape (data.n_units, _cov_dim)

    Returns:
      np.array of shape (data.n_units, )
    """
    ...

  def transform(self, d: data.ExpData) -> np.ndarray:
    """Return eta(W) for all W in d.

    Args:
      d: ExpData to be evaluated with n_units

    Returns:
      a np.ndarray of shape (n_units,_nus_dim) for eta(W), W in d.
    """
    ...

  def has_enough_samples(self) -> bool:
    ...

  def calculate_cs_width(self, delta: float, sub_g: float, rho: float) -> float:
    """Calculates the width of a confidence sequence on the norm of eta error.

    Guaranteed to hold with probability delta.

    See the paper for a justification of the formula.

    Args:
      delta: the error probability tolerance
      sub_g: the subGaussian parameter
      rho: hyperparameter for the mixture of the boundary

    Returns:
      The width of a confidence sequence
    """
    ...

  def set_to_truth(self, data_gen: data.DataGenerator) -> None:
    """Sets eta to the true value given a DataGenerator object.

    Assumes it is identifiable in the framework of this nuisance parameter.

    Args:
      data_gen: a DataGenerator that is used to fit the true parameters.
    """
    ...

  def calculate_score(self, d: data.ExpData) -> np.ndarray:
    """Calculates eta(W) and phi(W, eta) for all W in d.

    Args:
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    ...


class LinearParameterBinaryTreatment(NuisanceParameter):
  """We model each conditional response as a linear function.

  Specifically, we will fit
    mu(1,x) = X * beta_treat
    mu(0,x) = X * beta_cont
  The corresponding phi function is
    phi(W,eta) = mu(1,X) - mu(0,X)
  """

  def __init__(self, n_features):
    super().__init__()
    self._cov_dim = n_features
    self._nus_dim = 2  # The dimension of the nuisance parameter
    self.treat_response = linear_model.Ridge(alpha=1.0)
    self.cont_response = linear_model.Ridge(alpha=1.0)
    self._model = (self.cont_response, self.treat_response)

    self._data = None
    self._has_been_fit = False
    self.recent_data = None

  def reset(self):
    self.__init__(self._cov_dim)

  def fit(self, d: data.ExpData) -> None:
    """Fits eta to the provided experimental data.

    Args:
      d: ExpData used to fit the nuisance parameter. Expected shapes are
        d.cov: (n_units, n_features)
        d.exp: (n_units,)
        d.rsp: (n_units,)
    """
    if d.n_features != self._cov_dim:
      raise ValueError(
          f"Data has {d.n_features} dimension but {self._cov_dim} was expected")

    # Build the two datasets
    treat_data = []
    cont_data = []
    treat_labels = []
    cont_labels = []

    for i, x in enumerate(d.cov):
      if d.exp[i] == 1:
        treat_data.append(x)
        treat_labels.append(d.rsp[i])
      else:
        cont_data.append(x)
        cont_labels.append(d.rsp[i])

    self.recent_data = d

    if treat_data == 0 or cont_data == 0:
      logging.warning(
          "Nuisance parameter was fit on data not including treatment and control"
      )
      logging.warning("Not fitting the models yet")
      return

    self.treat_response.fit(treat_data, treat_labels)
    self.cont_response.fit(cont_data, cont_labels)

    self._has_been_fit = True

  def transform(self, d: data.ExpData) -> np.ndarray:
    """Return eta(W) for all W in d.

    Args:
      d: ExpData to be evaluated with n_units

    Returns:
      a np.ndarray of shape (n_units,_nus_dim) for eta(W), W in d.
    """
    if not self._has_been_fit:
      logging.warning("Model has not been properly fit yet")
      nan_array = np.empty((d.n_units, self._nus_dim))
      nan_array[:] = np.nan
      return nan_array

    if d.n_features != self._cov_dim:
      raise ValueError(
          f"Data has {d.n_features} dimension but {self._cov_dim} was expected")

    eta = np.array(
        [self.cont_response.predict(d.cov),
         self.treat_response.predict(d.cov)]).transpose()

    return eta

  def _phi(self, eta: np.ndarray, d: data.ExpData) -> np.ndarray:
    """Calculates phi(W, eta) for all W in d and eta(W).

    Args:
      eta: np.ndarray of shape (d.n_units, _nus_dim), which has been evaluated
        on W.
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    if eta.shape[1] != self._nus_dim:
      raise ValueError(
          f"eta has dimension {eta.shape[1]} but {self._nus_dim} was expected.")

    return eta[:, 1] - eta[:, 0]

  def calculate_score(self, d: data.ExpData) -> np.ndarray:
    """Calculates eta(W) and phi(W, eta) for all W in d.

    Args:
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    eta = self.transform(d)
    return self._phi(eta, d)

  def set_to_truth(self, data_gen: data.LinearDataGenerator):
    """Sets eta to the true value given a DataGenerator object.

    Assumes it is identifiable in the framework of this nuisance parameter.

    Args:
      data_gen: a DataGenerator that is used to fit the true parameters.
    """
    self.treat_response.intercept_ = data_gen.tau
    self.treat_response.coef_ = data_gen.beta
    self.cont_response.coef_ = data_gen.beta
    self.cont_response.intercept_ = 0


class Predictor(Protocol):

  def fit(self, x: np.ndarray, y: np.ndarray) -> None:
    ...

  def predict(self, x: np.ndarray) -> np.ndarray:
    ...


class ProbPredictor(Protocol):

  def fit(self, x: np.ndarray, y: np.ndarray) -> None:
    ...

  def predict_proba(self, x: np.ndarray) -> np.ndarray:
    ...


class AIPW(NuisanceParameter):
  """An AIWP estimator with arbitrary sklearn-style predictors.

  We use any sklearn-style estimator to model the conditional
  response and the propensity functions.

  Attributes:
  response: a Predictor that models mu(z,x) = Ex[Y|Z=z,X=x]
  propensity: a ProbPredictor that models e(z) = P[X=1|Z=z]  The subclasses
    AIPWLinearLogistic and AIPWKernelRidgeLogistic are specific choices of
    regressor and propensity.
  """

  def __init__(
      self,
      n_features,
      response: Predictor,
      propensity: ProbPredictor,
      min_units=5,
      min_overlap=.05,
  ):
    super().__init__()
    self._model = None  # The model used
    self._has_been_fit = False
    self.recent_data = None
    self._cov_dim = n_features
    self.response = response
    self.propensity = propensity

    self._nus_dim = 4  # The dimension of the nuisance parameter

    self._min_units = min_units
    self._min_overlap = min_overlap

  def reset(self):
    self.response = sklearn.clone(self.response)
    self.propensity = sklearn.clone(self.propensity)
    self._has_been_fit = False
    self.recent_data = None

  def fit(self, d: data.ExpData):
    """Fits eta to the provided experimental d.

    Args:
      d: ExpData used to fit the nuisance parameter. Expected shapes are
        d.cov: (n_units, n_features)
        d.exp: (n_units,)
        d.rsp: (n_units,)
    """
    n_units = len(d)
    if d.n_features != self._cov_dim:
      raise ValueError(
          f"Data has {d.n_features} dimension but {self._cov_dim} was expected")

    self.recent_data = d

    if sum(d.exp) < self._min_units or sum(d.exp) > n_units - self._min_units:
      message = (f"{sum(d.exp)} treatment units of {n_units} total units is not"
                 " enough units in control or treatment to fit eta.")
      logging.warning(message)
      return

    self.response.fit(np.c_[d.exp, d.cov], d.rsp)
    self.propensity.fit(d.cov, d.exp)

    self._has_been_fit = True

  def calculate_cs_width(self, delta, sub_g, rho) -> float:
    """Calculates the width of a confidence sequence on the norm of eta error.

    Guaranteed to hold with probability delta.

    See the paper for a justification of the formula.

    Args:
      delta: the error probability tolerance
      sub_g: the subGaussian parameter
      rho: hyperparameter for the mixture of the boundary

    Returns:
      The width of a confidence sequence
    """
    x = self.recent_data.cov
    d = self._cov_dim
    v = x.transpose().dot(x)

    numerator = sub_g * np.log(np.linalg.det(sub_g * v + rho * np.identity(d)))
    denominator = sub_g * np.log(rho**d * delta**2)
    if numerator <= denominator:  # not enough samples have been reached
      return np.inf

    radius = np.sqrt(numerator - denominator)

    norm_matrix = v.dot(np.linalg.inv(v + rho / sub_g * np.identity(d))).dot(v)
    w, _ = np.linalg.eig(norm_matrix)
    mu_radius = radius / np.sqrt(min(w))

    eta_radius = mu_radius

    return mu_radius + eta_radius * (1 / self._min_overlap - 2)

  def transform(self, d: data.ExpData) -> np.ndarray:
    """Return eta(Z) for all Z in d.

    Args:
      d: ExpData to be evaluated with n_units

    Returns:
      a np.ndarray of shape (n_units,_nus_dim) for eta(W), W in d.
    """
    if not self._has_been_fit:
      logging.warning("Warning: model has not been properly fit yet")
      nan_array = np.empty((d.n_units, self._nus_dim))
      nan_array[:] = np.nan
      return nan_array

    if d.n_features != self._cov_dim:
      raise ValueError(
          f"Data has {d.n_features} dimension but {self._cov_dim} was expected")

    n_units = d.n_units
    predicted_probs = self.propensity.predict_proba(d.cov)[:, 1]
    predicted_probs = np.clip(predicted_probs, self._min_overlap,
                              1 - self._min_overlap)
    eta = np.array([
        self.response.predict(np.c_[np.zeros(n_units), d.cov]),
        self.response.predict(np.c_[np.ones(n_units), d.cov]),
        self.response.predict(np.c_[d.exp, d.cov]), predicted_probs
    ]).transpose()

    return eta

  def _phi(self, eta: np.ndarray, d: data.ExpData) -> np.ndarray:
    """Calculates phi(W, eta) for all W in d and eta(W).

    Args:
      eta: np.ndarray of shape (d.n_units, _nus_dim), which has been evaluated
        on W.
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    if eta.shape[1] != self._nus_dim:
      raise ValueError(
          f"eta has dimension {eta.shape[1]} but {self._nus_dim} was expected.")
    if any(eta[:, 3] < 1e-15) or any(eta[:, 3] > 1 - 1e-15):
      raise ValueError("Eta is to close to exiting the unit interval")
    cond_difference = eta[:, 1] - eta[:, 0]
    ipw = (d.rsp - eta[:, 2]) * (
        d.exp / eta[:, 3] - (1 - d.exp) / (1 - eta[:, 3]))
    return cond_difference + ipw

  def calculate_score(self, d: data.ExpData) -> np.ndarray:
    """Calculates eta(W) and phi(W, eta) for all W in d.

    Args:
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    eta = self.transform(d)
    return self._phi(eta, d)

  def set_to_truth(self, data_gen: data.DataGenerator):
    """Sets eta to the true value given a DGenerator object.

    Assumes it is identifiable in the framework of this nuisance parameter.

    Args:
      data_gen: a DGenerator that is used to fit the true parameters.
    """
    if not isinstance(data_gen, data.LinearDataGenerator):
      raise ValueError(("The DataGenerator is not a LinearDataGenerator and is "
                        "not identifiable in this model"))
    self.response.coef_ = np.r_[data_gen.tau, data_gen.beta]
    self.propensity.coef_ = data_gen.gamma.reshape(1, self._cov_dim)
    self.propensity.intercept_ = 0

  def has_enough_samples(self) -> bool:
    return self._has_been_fit


class AIPWLinearLogistic(AIPW):
  """AIPW where the response is linear and the propensity logistic.

  Specifically, we will fit
    mu(z,x) = z * beta + x tau
    e(Z) = logit(Z*gamma)
  The corresponding phi function is
    phi(W,eta) = mu(1,Z) - mu(0,Z) + (Y-mu(X,Z))(X/e(Z) - (1-X)/(1-e(Z)))
  """

  def __init__(self, n_features, min_units=5, min_overlap=.05):
    super().__init__(
        n_features,
        response=linear_model.Ridge(alpha=1.0, fit_intercept=False),
        propensity=linear_model.LogisticRegression(warm_start=True),
        min_units=min_units,
        min_overlap=min_overlap,
    )

  def get_response_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(self.response, linear_model.Ridge):
      return (self.response.coef_, self.response.intercept_)
    else:
      return (np.nan, np.nan)


class AIPWKernelRidgeLogistic(AIPW):
  """AIPW with kernel ridge regression, logistic regression.

  Specifically, we will fit
    mu(X,x) = Z * beta + X * tau
    e(Z) = logit(Z*gamma)
  The corresponding phi function is
    phi(W,eta) = mu(1,Z) - mu(0,Z) + (Y-mu(X,Z))(X/e(Z) - (1-X)/(1-e(Z)))
  """

  def __init__(self, n_features, min_units=5, min_overlap=.05):
    super().__init__(
        n_features,
        response=sklearn.kernel_ridge.KernelRidge(alpha=1.0, kernel="rbf"),
        propensity=linear_model.LogisticRegression(warm_start=True),
        min_units=min_units,
        min_overlap=min_overlap,
    )


class AIPWKernelGradientBoosting(AIPW):
  """An AIPW with GP response and gradient boosted propensity."""

  def __init__(self, n_features, min_units=5, min_overlap=.05):
    super().__init__(
        n_features,
        response=sklearn.kernel_ridge.KernelRidge(alpha=1.0, kernel="rbf"),
        propensity=sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=5, learning_rate=1.0, max_depth=2, random_state=0),
        min_units=min_units,
        min_overlap=min_overlap,
    )


class FrontDoorLinearFiniteZ(NuisanceParameter):
  """This nuisance parameter estimates for the frontdoor formula.

  We must fit models for:
  - mu_Y(Z,X) : y_response
  - P(Z|X): cov_given_
  - P(X)

  The corresponding phi function is
    phi(W,eta) =
      (Y-mu_Y(Z,X))(P(Z|X=1) - P(Z|X=0)) / P(Z|X)
      + sum_x' mu_Y(x',X)(P(Z=x'|X=1) - P(Z=x'|X=0))
      + (X/P(X) - (1-X)(1-P(X)) * {
        sum_{t'} mu_Y(Z,t')f(t')
        - sum_{x', t'} mu_Y(x', t')P(Z|t')P(t')
      }

  We assume a binary X. We thus need to estimate two densities, P(Z|X=1) and
  P(Z|X=0), and we need to be able to integrate over these densities.

  If we assume that $Z$ has finite support, the simplest density estimator is
  a (smoothed) histogram.
  """

  def __init__(
      self,
      n_features,
      min_units=5,
      min_overlap: float = .05,
      min_ratio_to_uniform: float = 10,
  ):
    super().__init__()
    self._cov_dim = 0
    self._nus_dim = 0  # The dimension of the nuisance parameter maps to
    self._data = None
    self._has_been_fit = False
    self.recent_data = None

    self._cov_dim = n_features
    self._min_overlap = min_overlap
    self._min_units = min_units
    self._min_ratio_to_uniform = min_ratio_to_uniform
    self._nus_dim = 8  # The dimension of the nuisance parameter

    self.y_response = linear_model.Ridge(
        alpha=1.0, fit_intercept=True)  # E[Y|Z, X]
    self.cov_given_exp = data.TabularCPD(min_ratio_to_uniform)  # P(Z|X)
    self.exp_prob = .5  # P(X)
    self._model = (self.y_response, self.cov_given_exp, self.exp_prob)

  def reset(self):
    self.__init__(
        self._cov_dim,
        self._min_units,
        self._min_overlap,
        self._min_ratio_to_uniform,
    )
    self._has_been_fit = False
    self.recent_data = None

  def fit(self, d: data.ExpData):
    """Fits eta to the provided experimental d.

    Args:
      d: ExpData used to fit the nuisance parameter. Expected shapes are
        d.cov: (n_units, n_features)
        d.exp: (n_units,)
        d.rsp: (n_units,)
    """
    if d.n_features != self._cov_dim:
      raise ValueError(
          f"Data has {d.n_features} dimension but {self._cov_dim} was expected")

    self.recent_data = d

    if sum(d.exp) < self._min_units or sum(d.exp) > d.n_units - self._min_units:
      print(
          f"---Frontdoor: {sum(d.exp)} treatment units of {d.n_units} total units "
          "is not enough units in control or treatment to fit eta.")
      return
    self.y_response.fit(np.array(np.c_[d.exp, d.cov]), d.rsp)
    self.exp_prob = np.clip(
        np.mean(d.exp), self._min_overlap, 1 - self._min_overlap)
    self.cov_given_exp.fit(d.exp, d.cov)

    self._has_been_fit = True

  def calculate_cs_width(self, delta, sub_g, rho):
    """Calculates the width of a confidence sequence on the norm of eta error.

    Guaranteed to hold with probability delta.
    A linear regression CS is generated for both, then map it through
    the sigmoid.
    See the paper for a justification of the formula.

    Args:
      delta: the error probability tolerance
      sub_g: the subGaussian parameter
      rho: hyperparameter for the mixture of the boundary

    Returns:
      The width of a confidence sequence
    """
    x = np.array(np.c_[self.recent_data.exp, self.recent_data.cov])
    d = self._cov_dim + 1
    v = x.transpose().dot(x)

    delta = delta / 3  # we are making three confidence sequences
    numerator = sub_g * np.log(np.linalg.det(sub_g * v + rho * np.identity(d)))
    denominator = sub_g * np.log(rho**d * delta**2)
    if numerator <= denominator:  # not enough samples have been reached
      return np.inf

    radius = np.sqrt(numerator - denominator)

    norm_matrix = v.dot(np.linalg.inv(v + rho / sub_g * np.identity(d))).dot(v)
    w, _ = np.linalg.eig(norm_matrix)
    mu_radius = radius / np.sqrt(min(w))

    def uniform_boundary(n, delta):
      # check if the radius is well defined
      if n <= delta**2 / 4 / rho:
        return np.inf
      else:
        return np.sqrt(1 / n * (.25 + rho / n) * np.log(
            (n / 4 + rho) / (delta**2 * rho)))

    t_radius = uniform_boundary(len(x), delta / 3)

    cov_given_exp_radius = 2 * self.cov_given_exp.support_size() * t_radius
    if np.isnan(t_radius) or np.isnan(cov_given_exp_radius) or np.isnan(
        mu_radius):
      print("Warning, Nans found when calculating CS width.")

    return (mu_radius + t_radius * (1 / self._min_overlap - 1) + mu_radius * 2 *
            (1 / self._min_overlap - 1))

  def transform(self, d: data.ExpData):
    """Returns the nuisance parameter evaluated on data d.

    computes, for every W_i in d
    eta_i = (
      0:  mu_Y(z_i, x_i),
      1:  P(z_i|X=1),
      2:  P(z_i|X=0),
      3:  P(z_i|x_i),
      4:  W(z_i) = (z_i / P(X=1) - (1-z_i) / P(X=0))),
      5:  f_1(x_i) = sum_z' mu_Y(z',x_i)(P(Z=z'|X=1) - P(Z=z'|X=0)),
      6:  f_2(z_i) = sum_{x'} mu_Y(z_i,x') P(X=x'),
      7:  f_3(z_i) = sum_{z', x'} mu_Y(z', x') P(z_i|X=x') P(X=x'),
    )

    The corresponding phi function is
    phi(W,eta) =
      (Y-mu_Y(Z,X))(P(Z|X=1) - P(Z|X=0)) / P(Z|X)
      + sum_z' mu_Y(z',X)(P(Z=z'|X=1) - P(Z=z'|X=0))
      + (X/P(X) - (1-X)(1-P(X)) * {
        sum_{x'} mu_Y(Z,x')P(x')
        - sum_{z', x'} mu_Y(z', x')P(Z|x')P(x')
      }
    Which translates to
     phi(W,eta) =
     (Y-eta[:,0]) * (eta[:,1] / eta[:,3] - eta[:,2] / eta[:, 3])
     + eta[:, 5]
     + eta[:,4] * (eta[:,6] - eta[:,7])

    Args:
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, 7)
    """
    if d.n_features != self._cov_dim:
      raise ValueError(
          f"Data is of dimension {d.n_features} but {self._cov_dim} is expected."
      )

    if not self._has_been_fit:
      print("Warning: model has not been properly fit yet.")
      nan_array = np.empty((d.n_units, self._nus_dim))
      nan_array[:] = np.nan
      return nan_array

    n_units = len(d)
    mu_y_zx = self.y_response.predict(np.array(np.c_[d.exp, d.cov]))
    p_z_x1 = self.cov_given_exp.predict(np.ones(n_units, dtype=int), d.cov)
    p_z_x0 = self.cov_given_exp.predict(np.zeros(n_units, dtype=int), d.cov)
    p_z_x = self.cov_given_exp.predict(d.exp, d.cov)

    p_z_x1 = np.clip(p_z_x1, self._min_overlap, 1 - self._min_overlap)
    p_z_x0 = np.clip(p_z_x0, self._min_overlap, 1 - self._min_overlap)
    p_z_x = np.clip(p_z_x, self._min_overlap, 1 - self._min_overlap)

    z_support = self.cov_given_exp.z_support()

    mu_yz_ix1 = self.y_response.predict(
        np.array(np.c_[np.ones(len(d), dtype=int), d.cov]))  # mu_y(z_i, X=1)
    mu_yz_ix0 = self.y_response.predict(
        np.array(np.c_[np.zeros(len(d), dtype=int), d.cov]))  # mu_y(z_i, X=0)

    f_2 = self.exp_prob * mu_yz_ix1 + (1 - self.exp_prob) * mu_yz_ix0

    f_1 = np.zeros(len(d))
    f_3 = np.zeros(len(d))
    for z in z_support:
      mu_yzx_i = self.y_response.predict(
          np.array(np.c_[d.exp,
                         np.tile(z,
                                 (len(d), 1))]))  # mu_y(z, X=1), z in support
      f_1 += mu_yzx_i * (
          self.cov_given_exp.predict(np.array([1]), z) -
          self.cov_given_exp.predict(np.array([0]), z))

      f_3 += (
          self.y_response.predict([np.insert(z, 0, 1)]) * self.exp_prob * p_z_x1
          + self.y_response.predict([np.insert(z, 0, 0)]) *
          (1 - self.exp_prob) * p_z_x0)

    w = d.exp / self.exp_prob - (1 - d.exp) / (1 - self.exp_prob)

    eta = np.array([
        mu_y_zx,
        p_z_x1,
        p_z_x0,
        p_z_x,
        w,
        f_1,
        f_2,
        f_3,
    ]).transpose()

    return eta

  def _phi(self, eta: np.ndarray, d: data.ExpData):
    """Maps eta to phi(eta).

    Recall that
      eta_i(z_i,x_i) = (
      0:  mu_Y(z_i, x_i),
      1:  P(z_i|X=1),
      2:  P(z_i|X=0),
      3:  P(z_i|x_i),
      4:  W(x_i) = (x_i / P(X=1) - (1-x_i) / P(X=0))),
      5:  f_1(x_i) = sum_z' mu_Y(z',x_i)(P(Z=z'|X=1) - P(Z=z'|X=0)),
      6:  f_2(z_i) = sum_{x'} mu_Y(z_i,x') P(X=x'),
      7:  f_3(z_i) = sum_{z', x'} mu_Y(z', x') P(Z=z_i|X=x') P(X=x'),
    )

    The corresponding phi function is
    phi(W,eta) =
      (Y-mu_Y(Z,X))(P(Z|X=1) - P(Z|X=0)) / P(Z|X)
      + sum_z' mu_Y(z',X)(P(Z=z'|X=1) - P(Z=z'|X=0))
      + (X/P(X) - (1-X)(1-P(X)) * {
        sum_{x'} mu_Y(Z,x')P(x')
        - sum_{z', x'} mu_Y(z', x')P(Z|x')P(x')
      }
    Which translates to
     phi(W,eta) =
     (Y-eta[:,0]) * (eta[:,1] / eta[:,3] - eta[:,2] / eta[:, 3])
     + eta[:, 5]
     + eta[:,4] * (eta[:,6] - eta[:,7])


    Args:
      eta: np.ndarray of shape (d.n_units, _nus_dim), which has been evaluated
        on W.
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    if eta.shape[1] != self._nus_dim:
      raise ValueError(
          f"eta has dimension {eta.shape[1]} but {self._nus_dim} was expected.")

    phi = ((d.rsp - eta[:, 0]) *
           (eta[:, 1] / eta[:, 3] - eta[:, 2] / eta[:, 3]) + eta[:, 5] +
           eta[:, 4] * (eta[:, 6] - eta[:, 7]))

    return phi

  def calculate_score(self, d: data.ExpData) -> np.ndarray:
    """Calculates eta(W) and phi(W, eta) for all W in d.

    Args:
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    eta = self.transform(d)
    return self._phi(eta, d)

  def has_enough_samples(self) -> bool:
    return self._has_been_fit


class LinearRegressionParameter(NuisanceParameter):
  """Fits a linear model where X can be continuous.

  We assume the model
    Y = Z beta_1 + X tau + epsilon_1
  and so fitting tau corresponds to using linearly regressing Y on (X, Z) then
  returning the first coordinate of the parameter vector (corresponding to the
  X component); this vector is beta_hat.

  Because of the linear structure, the influence function is simply
    beta_hat - tau,
  meaning that phi is the identity and the nuisance function is eta = beta_hat.

  However, for the purpose of estimating the variance, let's consider estimating
  the covariance matrix separately; this covariance estimation is what is
  abstracted by this NuisanceParameter subclass.

  While linear regression is defined for a single set of samples, the analysis
  will appreciate the flexibility of letting the covariance matrix be fit on
  a separate sample. Hence, the nuisance parameter eta will be the inverse
  covariance matrix of [X Z] and phi(W, ) will evaluate
    e_1^T eta [X Z]^T Y
  """

  def __init__(self, n_features, l_reg=0.1):
    super().__init__()
    self._model = None  # The model used
    self._cov_dim = 0
    self._nus_dim = 0  # The dimension of the nuisance parameter maps to
    self._data = None
    self._has_been_fit = False
    self.recent_data = None

    self._cov_dim = n_features
    # The dimension of the nuisance parameter
    self._nus_dim = (n_features + 1)**2

    # The model is just the covariance matrix
    self._model = np.zeros(((n_features + 1), (n_features + 1)))
    self.l_reg = l_reg

  def fit(self, d: data.ExpData):
    """Fits eta to the provided experimental d.

    Args:
      d: ExpData used to fit the nuisance parameter. Expected shapes are
        d.cov: (n_units, n_features)
        d.exp: (n_units,)
        d.rsp: (n_units,)
    """
    if d.n_units < 1:  # not enough data to fit
      return
    data_matrix = np.array(np.c_[d.exp, d.cov])

    self._model = np.linalg.inv(data_matrix.transpose().dot(data_matrix) +
                                self.l_reg *
                                np.identity(self._cov_dim + 1)) * d.n_units
    self._has_been_fit = True
    self.recent_data = d

  def transform(self, d: data.ExpData):
    return self._model

  def _phi(self, eta: np.ndarray, d: data.ExpData):
    """Evaluates e_1^T eta [X Z]^T Y for a vector X, Z, Y.

    Args:
      eta: np.ndarray of shape (d.n_units, _nus_dim), which has been evaluated
        on W.
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    xzy = np.c_[d.exp, d.cov].transpose().dot(np.diag(d.rsp))
    return eta.dot(xzy)[0, :]

  def calculate_score(self, d: data.ExpData) -> np.ndarray:
    """Calculates eta(W) and phi(W, eta) for all W in d.

    Args:
      d: ExpData of shape (d.n_units, _cov_dim)

    Returns:
      np.array of shape (d.n_units, )
    """
    eta = self.transform(d)
    return self._phi(eta, d)

  def has_enough_samples(self) -> bool:
    return self._has_been_fit
