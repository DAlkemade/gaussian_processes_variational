from typing import Type

import numpy as np
from GPy.core.parameterization.variational import NormalPosterior
from GPy.kern import Kern
from sklearn.metrics import mean_squared_error



def KL_divergence(model_1, model_2, samples):
    """Determine KL divergence between the posteriors of two GPs

    :param model_1:
    :param model_2:
    :param samples: function inputs over which the posteriors are created
    :return:
    """
    posterior_1 = create_posterior_object(model_1, samples)
    posterior_2 = create_posterior_object(model_2, samples)
    return posterior_1.KL(posterior_2)


def diff_marginal_likelihoods(variational_gp, full_gp, log: bool):
    """Calculate difference between true marginal likelihood and variational distribution."""
    log_likelihood_variational = variational_gp.log_likelihood()[0][0]
    log_likelihood_true = full_gp.log_likelihood()

    if log:
        return log_likelihood_true - log_likelihood_variational
    else:
        likelihood_variational = 2 ** log_likelihood_variational
        likelihood_true = 2 ** log_likelihood_true
        return likelihood_true - likelihood_variational


def create_posterior_object(m, samples):
    """Create a NormalPosterior object.

    :param m: GP with which to create posterior
    :param samples: function inputs to create the posterior at
    :return:
    """
    mu, covar = m.predict_noiseless(samples, full_cov=True)
    variances = covar.diagonal()
    variances = np.reshape(variances, (len(samples), 1))
    return NormalPosterior(means=mu, variances=variances)


def find_mse(model, samples, y_true):
    """Find mean square error for a certain model on a certain set of inputs.

    :param model: GP
    :param samples: inputs
    :param y_true: true outputs (including noise)
    :return: Mean square error
    """
    mu, covar = model.predict_noiseless(samples, full_cov=True)
    return mean_squared_error(y_true, mu)


def calc_K_tilda(kernel: Type[Kern], X_train: np.array, X_m: np.array):
    """Find K_tilda = Cov(f|fm)."""
    Knn = kernel.K(X_train, X_train)
    Knm = kernel.K(X_train, X_m)
    Kmn = kernel.K(X_m, X_train)
    Kmm = kernel.K(X_m, X_m)
    temp = np.dot(np.dot(Knm, np.linalg.inv(Kmm)), Kmn)
    K_tilda = np.subtract(Knn, temp)
    return K_tilda


def calc_metric3(K_tilda):
    """Calculate difference between trace and determinant of K_tilda"""
    trace = np.trace(K_tilda)
    # determinant = np.linalg.det(K_tilda)
    _, log_determinant = np.linalg.slogdet(K_tilda)
    diff = trace - log_determinant
    print(trace, log_determinant, diff)
    return diff


def find_logKy(X_train, model):
    Knn = model.kern.K(X_train, X_train)
    # try:
    #     variance = kernel_full.variance
    # except AttributeError:  # Some kernels use a different attribute name
    #     variance = kernel_full.variances
    # if variance.size == 1:
    #     variance_valuez = variance.max()
    #     variance_value = variance_valuez[0]
    # else:
    #     raise NotImplementedError("Only implemented for a single variance")
    variance_value = model.likelihood.variance
    variance_value = float(variance_value)
    variances = [variance_value] * len(Knn)
    Isigma2 = np.diag(variances)
    Ky = Knn + Isigma2
    _, logKy = np.linalg.slogdet(Ky)
    # noiseless = model.predict_noiseless(X_train, full_cov=True)
    # not_noiseless = model.predict(X_train, full_cov=True)
    return logKy