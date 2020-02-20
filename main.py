# The following notebook was used as the starting point: https://github.com/SheffieldML/notebook/blob/master/GPy/sparse_gp_regression.ipynb
import argparse

import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.core.parameterization.variational import NormalPosterior
from sklearn.svm import SVR

np.random.seed(101)

N = 50
SIMULATION_NOISE_VAR = 0.05
DEFAULT_KERNEL = GPy.kern.RBF
RBF = 'rbf'


def simulate_data(k=GPy.kern.RBF(1)):
    """
    Simulate data using gaussian noise and a certain kernel
    :return:
    """
    X = np.linspace(0, 10, 50)[:, None]
    y = np.random.multivariate_normal(np.zeros(N), k.K(X) + np.eye(N) * np.sqrt(SIMULATION_NOISE_VAR)).reshape(-1, 1)
    return X, y


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


def create_posterior_object(m, samples):
    """
    Create a NormalPosterior object.
    :param m: GP with which to create posterior
    :param samples: function inputs to create the posterior at
    :return:
    """
    mu, covar = m.predict_noiseless(samples, full_cov=True)
    variances = covar.diagonal()
    variances = np.reshape(variances, (len(samples), 1))
    return NormalPosterior(means=mu, variances=variances)


def KL_divergence(model_1, model_2, samples):
    """
    Determine KL divergence between the posteriors of two GPs
    :param model_1:
    :param model_2:
    :param samples: function inputs over which the posteriors are created
    :return:
    """
    posterior_1 = create_posterior_object(model_1, samples)
    posterior_2 = create_posterior_object(model_2, samples)
    return posterior_1.KL(posterior_2)


def create_full_gp(X, y, kernel_type=DEFAULT_KERNEL, plot=True):
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


def create_sparse_gp(X, y, num_inducing=None, Z=None, plot=True, fix_inducing_inputs=False, fix_variance=False,
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
        m.plot()
        plt.title("Sparse GP model")
        plt.show()
        print(m)
    return m


def diff_marginal_likelihoods(variational_gp, full_gp, log: bool):
    """
    Calculate difference between true marginal likelihood and variational distribution.
    :param variational_gp:
    :param full_gp:
    :param log:
    :return:
    """
    log_likelihood_variational = variational_gp.log_likelihood()[0][0]
    log_likelihood_true = full_gp.log_likelihood()

    if log:
        return log_likelihood_true - log_likelihood_variational
    else:
        likelihood_variational = 2 ** log_likelihood_variational
        likelihood_true = 2 ** log_likelihood_true
        return likelihood_true - likelihood_variational


def fit_svm(X, y, plot):
    """
    Fit an SVM
    :param X:
    :param y:
    :param plot: whether to plot output
    :return:
    """
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X, y.flatten())
    z = clf.predict(X)
    if plot:
        plt.plot(X, z)
        plt.scatter(X, y)
        plt.title("SVM")
        plt.show()
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inducing', type=int, default=6)
    parser.add_argument('--kernel', type=str, default=RBF)

    args = parser.parse_args()
    num_inducing = args.num_inducing
    if args.kernel == RBF:
        kernel_class = GPy.kern.RBF
    else:
        raise ValueError("Unknown kernel")

    # Sample function
    X, y = simulate_data()


    # Create GPs
    m_full = create_full_gp(X, y, kernel_type=kernel_class)
    # Z = np.hstack((np.linspace(2.5,4.,3),np.linspace(7,8.5,3)))[:,None]
    m_sparse = create_sparse_gp(X, y, num_inducing=num_inducing, kernel_type=kernel_class)

    # KL divergence
    # samples = X = np.linspace(0,10,1000)[:,None]
    # samples = X
    samples = m_sparse.Z
    divergence = KL_divergence(m_sparse, m_full, samples)
    print(f'KL divergence posteriors over inducing inputs {divergence}')

    print(f"diff log likelihoods: {diff_marginal_likelihoods(m_sparse, m_full, True)}")
    print(f"diff likelihoods: {diff_marginal_likelihoods(m_sparse, m_full, False)}")

    # Show covar of inducing inputs and of full gp
    plot_covariance_matrix(m_sparse.posterior.covariance)

    # Test SVM
    fit_svm(X, y, plot=True)
