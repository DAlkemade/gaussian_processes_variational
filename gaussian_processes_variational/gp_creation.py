import GPy
import numpy as np
from matplotlib import pyplot as plt

from gaussian_processes_variational.compare import diff_marginal_likelihoods, find_mse
from gaussian_processes_variational.parameter_containers import FixedParameterSettings


def plot_covariance_matrix(cov_matrix):
    """Plot covariance matrix.

    :param cov_matrix: covariance matrix to be plotted
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cov_matrix, interpolation='none')
    fig.colorbar(im)
    plt.title("Covariance matrix")
    plt.show()


def create_full_gp(X, y, kernel, optimizer=None, plot=False):
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


def create_sparse_gp(X, y, kernel, num_inducing, fixedparameters: FixedParameterSettings, plot=False, optimizer=None):
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
    if fixedparameters.fix_inducing_inputs:
        m.inducing_inputs.fix()
    if fixedparameters.fix_gaussian_noise_variance:
        m.Gaussian_noise.variance.fix()
    if fixedparameters.fix_variance:
        try:
            m.kern.variance.fix()
        except AttributeError:
            m.kern.variances.fix()
    if fixedparameters.fix_lengthscale:
        try:
            m.kern.lengthscale.fix()
        except AttributeError:
            pass # some kernels don't have a lengthscale

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
