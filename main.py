# The following notebook was used as the starting point: https://github.com/SheffieldML/notebook/blob/master/GPy/sparse_gp_regression.ipynb
from argparse import ArgumentParser

import os
import GPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3
import configparser

from compare import diff_marginal_likelihoods, find_mse
from non_gp_alternatives import fit_svm, linear_regression
from simulation import simulate_data_sklearn, simulate_data, Data

np.random.seed(101)

DEFAULT_KERNEL = GPy.kern.RBF
RBF = 'rbf'
GPy.plotting.change_plotting_library('matplotlib')


def plot_covariance_matrix(cov_matrix):
    """
    Plot covariance matrix.
    :param cov_matrix: covariance matrix to be plotted
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cov_matrix, interpolation='none')
    fig.colorbar(im)
    plt.title("Covariance matrix")
    plt.show()


def create_full_gp(X, y, kernel_type=DEFAULT_KERNEL, plot=False):
    """
    Create non-sparse Gaussian Process.
    :param X: inputs
    :param y: noisy function values at inputs
    :param kernel_type: class of kernel
    :param plot: whether to plot the posterior
    :return:
    """
    kernel = create_kernel(X, kernel_type)
    m = GPy.models.GPRegression(X, y, kernel=kernel)
    m.optimize('bfgs')
    if plot:
        m.plot()
        plt.title("Full GP model")
        plt.show()
        print(m)
    return m


def create_kernel(X, kernel_class):
    """
    Return kernel of given type with correct dimensions
    :param X:
    :param kernel_class:
    :return:
    """
    input_dim = X.shape[1]
    kernel = kernel_class(input_dim)
    return kernel


def create_sparse_gp(X, y, num_inducing=None, Z=None, plot=False, fix_inducing_inputs=False, fix_variance=False,
                     fix_lengthscale=False, kernel_type=DEFAULT_KERNEL):
    """
    Create sparse Gaussian Process using the method of Titsias, 2009
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
    if num_inducing is None and Z is None:
        raise ValueError("Neither num_inducing or Z was defined")

    kernel = create_kernel(X, kernel_type)

    if Z is not None:
        m = GPy.models.SparseGPRegression(X, y, Z=Z, kernel=kernel)
    else:
        m = GPy.models.SparseGPRegression(X, y, num_inducing=num_inducing, kernel=kernel)
    # m.likelihood.variance = NOISE_VAR
    m.Z.unconstrain()
    if fix_inducing_inputs:
        m.inducing_inputs.fix()
    if fix_variance:
        m.rbf.variance.fix()
    if fix_lengthscale:
        m.rbf.lengthscale.fix()

    m.optimize('bfgs')
    if plot:
        # m.plot()
        m.plot(plot_limits=(-10, 30))
        plt.title("Sparse GP model")
        plt.show()
        print(m)
    return m


def evaluate_sparse_gp(data: Data, num_inducing, kernel_type=GPy.kern.RBF, plot_figures=False):
    """
    Create a sparse GP and compare it against a full GP.
    :param data: contains training and test data
    :param num_inducing: number of inducing points
    :param kernel_type: type of kernel
    :param plot_figures: whether to plot figures
    """
    # Create GPs
    m_full = create_full_gp(data.X_train, data.y_train, kernel_type=kernel_type, plot=plot_figures)
    m_sparse = create_sparse_gp(data.X_train, data.y_train, num_inducing=num_inducing, kernel_type=kernel_type,
                                plot=plot_figures)

    print(f"diff log likelihoods: {diff_marginal_likelihoods(m_sparse, m_full, True)}")
    print(f"diff likelihoods: {diff_marginal_likelihoods(m_sparse, m_full, False)}")

    # Show covar of inducing inputs and of full gp
    plot_covariance_matrix(m_sparse.posterior.covariance)

    if data.X_test is not None:
        mse_test_sparse = find_mse(m_sparse, data.X_test, data.y_test)
        mse_test_full = find_mse(m_full, data.X_test, data.y_test)

        print(f'MSE test full: {mse_test_full}; MSE test sparse: {mse_test_sparse}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='linear_high_dim_creates_warnings.ini')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join('configs', args.config))

    input_dim = config['DATA'].getint('input_dim')
    n_samples = config['DATA'].getint('n')
    simulation_function_string = config['DATA']['simulation_function']
    num_inducing = config['SPARSE_GP'].getint('num_inducing')

    if config['GP']['kernel'] == RBF:
        kernel_class = GPy.kern.RBF
    else:
        raise ValueError("Unknown kernel")

    plot = input_dim <= 1

    # Sample function
    if simulation_function_string == 'make_regression':
        n_informative = config['DATA'].getint('input_dim_informative')
        if n_informative is None:
            n_informative = input_dim
        data = simulate_data_sklearn(make_regression, n_samples, n_features=input_dim, n_informative=n_informative)
    elif simulation_function_string == 'make_friedman1':
        data = simulate_data_sklearn(make_friedman1, n_samples, n_features=input_dim)
    elif simulation_function_string == 'make_friedman2':
        data = simulate_data_sklearn(make_friedman2, n_samples)
    elif simulation_function_string == 'make_friedman3':
        data = simulate_data_sklearn(make_friedman3, n_samples)
    elif simulation_function_string == 'rbf':
        data = simulate_data(n_samples, input_dim)
    else:
        raise ValueError("Unknown simulation function given")

    evaluate_sparse_gp(data, num_inducing, kernel_type=kernel_class, plot_figures=plot)

    # Test SVM
    fit_svm(data.X_train, data.y_train, plot=plot)
    linear_regression(data.X_train, data.y_train, plot=plot)
