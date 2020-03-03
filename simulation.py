from abc import ABC, abstractmethod

import GPy
import numpy as np
from sklearn.datasets import make_regression, make_friedman1
from sklearn.model_selection import train_test_split

SIMULATION_NOISE_VAR = 0.05


class Data(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.y_test = y_test
        self.y_train = y_train
        self.X_test = X_test
        self.X_train = X_train


class Simulator(ABC):
    """Simulates data with a method specified in _simulate."""

    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.random_state = 42

    @property
    def _n_test_plus_train(self):
        return self.n_samples * 2

    @abstractmethod
    def _simulate(self, n_features, **kwargs):
        """Simulate the data using a certain function."""
        pass

    def simulate(self, n_features, **kwargs):
        X, y = self._simulate(n_features, **kwargs)
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=self.random_state)
        return Data(X_train, X_test, y_train, y_test)


class FriedMan1Simulator(Simulator):
    """Simulate data for the Friedman #1 problem."""

    def __init__(self, n_samples):
        super().__init__(n_samples)

    def _simulate(self, n_features, **kwargs):
        return make_friedman1(n_samples=self._n_test_plus_train, noise=np.sqrt(SIMULATION_NOISE_VAR),
                              n_features=n_features, random_state=self.random_state)


class LinearSimulator(Simulator):
    """Simulate linear data."""

    def _simulate(self, n_features, **kwargs):
        """Simulate linear data.

        Note that if **kwargs do not specify n_informative, the default is n_informative=10
        """
        return make_regression(n_samples=self._n_test_plus_train, noise=np.sqrt(SIMULATION_NOISE_VAR),
                               n_features=n_features, random_state=self.random_state, **kwargs)


class RBFSimulator(Simulator):
    """Simulate data with an RBF kernel."""

    def _simulate(self, n_features, **kwargs):
        k = GPy.kern.RBF(n_features)
        # X = np.linspace(0, 10, n)[:, None]
        X, _ = make_regression(n_samples=self._n_test_plus_train, noise=np.sqrt(SIMULATION_NOISE_VAR),
                               n_features=n_features, random_state=self.random_state, **kwargs)
        X = 3 * X
        covariance = k.K(X) + np.eye(self._n_test_plus_train) * np.sqrt(SIMULATION_NOISE_VAR)
        y = np.random.multivariate_normal(np.zeros(self._n_test_plus_train), covariance).reshape(-1, 1)
        return X, y
