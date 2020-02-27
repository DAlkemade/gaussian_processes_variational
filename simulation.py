import GPy
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

SIMULATION_NOISE_VAR = 0.05

class Data:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.y_test = y_test
        self.y_train = y_train
        self.X_test = X_test
        self.X_train = X_train


def simulate_data_sklearn(function, n, **kwargs) -> Data:
    """
    Simulate linear data with noise using a sklearn dataset creation function.
    :param function: make_regression, make_friedman1,2 or 3
    :param n: number of samples for the training set and test set (each)
    :param kwargs: parameters for the sklearn data creation function
    :return: dataset
    """
    X, y = function(n_samples=n * 2, noise=np.sqrt(SIMULATION_NOISE_VAR), **kwargs)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    data = Data(X_train, X_test, y_train, y_test)
    # data.y_train, data.y_test = data.y_train.reshape(-1, 1), data.y_test.reshape(-1, 1)
    return data


def simulate_data(n, input_dim, k_class=GPy.kern.RBF) -> Data:
    """
    Simulate data using gaussian noise and a certain kernel
    :return:
    """
    k = k_class(input_dim)
    # X = np.linspace(0, 10, 50)[:, None]
    X = np.reshape(np.linspace(0, 10, n * input_dim)[:, None], (n, input_dim))
    # X, _ = make_regression(n, n_features=input_dim)
    # np.random.shuffle(X)
    # y = np.random.multivariate_normal(np.zeros(N), np.eye(N) * np.sqrt(SIMULATION_NOISE_VAR)).reshape(-1, 1)
    y = np.random.multivariate_normal(np.zeros(n), k.K(X) + np.eye(n) * np.sqrt(SIMULATION_NOISE_VAR)).reshape(-1, 1)

    return Data(X, None, y, None)
