import GPy
import numpy as np

from main import SIMULATION_NOISE_VAR


def simulate_data_sklearn(function, n, **kwargs):
    X, y = function(n_samples=n, noise=np.sqrt(SIMULATION_NOISE_VAR), **kwargs)
    y = y.reshape(-1, 1)
    return X, y


def simulate_data(n, input_dim, k_class=GPy.kern.RBF):
    """
    Simulate data using gaussian noise and a certain kernel
    :return:
    """
    k = k_class(input_dim)
    # X = np.linspace(0, 10, 50)[:, None]
    X = np.reshape(np.linspace(0, 10, n * input_dim)[:, None], (n, input_dim))
    # np.random.shuffle(X)
    # y = np.random.multivariate_normal(np.zeros(N), np.eye(N) * np.sqrt(SIMULATION_NOISE_VAR)).reshape(-1, 1)
    y = np.random.multivariate_normal(np.zeros(n), k.K(X) + np.eye(n) * np.sqrt(SIMULATION_NOISE_VAR)).reshape(-1, 1)

    return X, y
