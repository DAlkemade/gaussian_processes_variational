from main import create_posterior_object


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