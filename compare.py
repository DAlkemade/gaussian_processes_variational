import numpy as np
from GPy.core.parameterization.variational import NormalPosterior
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
