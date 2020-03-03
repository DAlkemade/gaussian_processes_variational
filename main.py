# The following notebook was used as the starting point: https://github.com/SheffieldML/notebook/blob/master/GPy/sparse_gp_regression.ipynb
import configparser
import os
from argparse import ArgumentParser

import GPy
import matplotlib.pyplot as plt
import numpy as np

from compare import diff_marginal_likelihoods, find_mse
from non_gp_alternatives import fit_svm, linear_regression
from simulation import Data, LinearSimulator, FriedMan1Simulator, RBFSimulator

np.random.seed(101)

RBF = 'rbf'
LINEAR = 'linear'
GPy.plotting.change_plotting_library('matplotlib')


def plot_covariance_matrix(cov_matrix):
    """Plot covariance matrix.

    :param cov_matrix: covariance matrix to be plotted
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cov_matrix, interpolation='none')
    fig.colorbar(im)
    plt.title("Covariance matrix")
    plt.show()


def create_full_gp(X, y, kernel, optimizer = None, plot=False):
    """Create non-sparse Gaussian Process.

    :param X: inputs
    :param y: noisy function values at inputs
    :param kernel_type: class of kernel
    :param plot: whether to plot the posterior
    :return:
    """
    m = GPy.models.GPRegression(X, y, kernel=kernel)
    m.optimize(optimizer)
    if plot:
        m.plot()
        plt.title("Full GP model")
        plt.show()
        print(m)
    return m


def create_kernel(X, kernel_class):
    """Return kernel of given type with correct dimensions.

    :param X:
    :param kernel_class:
    :return:
    """
    input_dim = X.shape[1]
    kernel = kernel_class(input_dim)
    return kernel


def create_sparse_gp(X, y, kernel, num_inducing, plot=False, fix_inducing_inputs=False, fix_variance=False,
                     fix_lengthscale=False, optimizer=None):
    """Create sparse Gaussian Process using the method of Titsias, 2009

    :param kernel_type: class of kernel
    :param X: inputs
    :param y: noisy function values at inputs
    :param num_inducing: number of inducing variables
    :param Z: inducing variables locations
    :param plot: whether to plot the posterior
    :param fix_inducing_inputs:
    :param fix_variance:
    :param fix_lengthscale:
    :return:
    """

    m = GPy.models.SparseGPRegression(X, y, num_inducing=num_inducing, kernel=kernel)

    # m.likelihood.variance = NOISE_VAR
    m.Z.unconstrain()
    if fix_inducing_inputs:
        m.inducing_inputs.fix()
    if fix_variance:
        m.rbf.variance.fix()
    if fix_lengthscale:
        m.rbf.lengthscale.fix()

    m.optimize(optimizer=optimizer)
    if plot:
        m.plot()
        # m.plot(plot_limits=(-10, 30))
        plt.title("Sparse GP model")
        plt.show()
        print(m)
    return m


def evaluate_sparse_gp(X_test: np.array, y_test: np.array, m_sparse, m_full):
    """Create a sparse GP and compare it against a full GP.

    :param data: contains training and test data
    :param num_inducing: number of inducing points
    :param kernel_type: type of kernel
    :param plot_figures: whether to plot figures
    """
    # Create GPs

    diff = diff_marginal_likelihoods(m_sparse, m_full, True)
    print(f"diff log likelihoods: {diff}")

    # Show covar of inducing inputs and of full gp
    plot_covariance_matrix(m_sparse.posterior.covariance)

    if X_test is not None:
        mse_test_sparse = find_mse(m_sparse, X_test, y_test)
        mse_test_full = find_mse(m_full, X_test, y_test)

        print(f'MSE test full: {mse_test_full}; MSE test sparse: {mse_test_sparse}')


def main():
    """Run the experiment using a certain config defined in the config file."""
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='playground.ini')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', args.config))
    input_dim = config['DATA'].getint('input_dim')
    n_samples = config['DATA'].getint('n')
    simulation_function_string = config['DATA']['simulation_function']
    num_inducing = config['SPARSE_GP'].getint('num_inducing')
    if config['GP']['kernel'] == RBF:
        kernel_class = GPy.kern.RBF
    elif config['GP']['kernel'] == LINEAR:
        kernel_class = GPy.kern.Linear
    else:
        raise ValueError("Unknown kernel")
    plot = input_dim <= 1
    # Sample function
    if simulation_function_string == 'make_regression':
        n_informative = config['DATA'].getint('input_dim_informative')
        if n_informative is None:
            n_informative = input_dim
        simulator = LinearSimulator(n_samples)
        data = simulator.simulate(input_dim, n_informative=n_informative)
    elif simulation_function_string == 'make_friedman1':
        simulator = FriedMan1Simulator(n_samples)
        data = simulator.simulate(input_dim)
    elif simulation_function_string == 'rbf':
        simulator = RBFSimulator(n_samples)
        data = simulator.simulate(input_dim)
    else:
        raise ValueError("Unknown simulation function given")

    kernel = create_kernel(data.X_train, kernel_class)
    kernel_sparse = create_kernel(data.X_train, kernel_class)

    m_full = create_full_gp(data.X_train, data.y_train, kernel, plot=plot)
    m_sparse = create_sparse_gp(data.X_train, data.y_train, kernel_sparse, num_inducing, plot=plot)

    evaluate_sparse_gp(data.X_test, data.y_test, m_sparse, m_full)

    # Test SVM
    fit_svm(data.X_train, data.y_train, plot=plot)
    linear_regression(data.X_train, data.y_train, plot=plot)


if __name__ == "__main__":
    main()
