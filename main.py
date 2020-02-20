# The following notebook was used as the starting point: https://github.com/SheffieldML/notebook/blob/master/GPy/sparse_gp_regression.ipynb

import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.core.parameterization.variational import NormalPosterior
from sklearn.svm import SVR

np.random.seed(101)


def plot_covariance_matrix(cov_matrix):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cov_matrix, interpolation='none')
    fig.colorbar(im)
    plt.title("Covariance matrix")
    plt.show()


def create_posterior_object(m, samples):
    mu, covar = m.predict_noiseless(samples, full_cov=True)
    variances = covar.diagonal()
    variances = np.reshape(variances, (len(samples), 1))
    return NormalPosterior(means=mu, variances=variances)


def KL_divergence(model_1, model_2, samples):
    posterior_1 = create_posterior_object(model_1, samples)
    posterior_2 = create_posterior_object(model_2, samples)
    return posterior_1.KL(posterior_2)

if __name__ == "__main__":

    # Sample function
    N = 50
    noise_var = 0.05

    X = np.linspace(0, 10, 50)[:, None]
    X_new = np.linspace(10, 15, 50)[:, None]
    k = GPy.kern.RBF(1)
    y = np.random.multivariate_normal(np.zeros(N), k.K(X) + np.eye(N) * np.sqrt(noise_var)).reshape(-1, 1)

    # Full GP fit
    m_full = GPy.models.GPRegression(X, y)
    m_full.optimize('bfgs')
    m_full.plot()
    plt.title("Full GP model")
    plt.show()
    print(m_full)

    # Z = np.hstack((np.linspace(2.5,4.,3),np.linspace(7,8.5,3)))[:,None]
    # m = GPy.models.SparseGPRegression(X,y,Z=Z)
    m = GPy.models.SparseGPRegression(X, y, num_inducing=6)
    m.likelihood.variance = noise_var
    # m.inducing_inputs.fix()
    m.rbf.variance.fix()
    m.rbf.lengthscale.fix()
    m.Z.unconstrain()
    m.optimize('bfgs')
    m.plot()
    plt.title("Sparse GP model")
    plt.show()
    print(m)

    # KL divergence
    # samples = X = np.linspace(0,10,1000)[:,None]
    # samples = X
    samples = m.Z

    # divergence = m.posterior.KL(full_posterior)
    divergence = KL_divergence(m, m_full, samples)
    print(divergence)

    log_likelihood_sparse_model = m.log_likelihood()[0][0]
    log_likelihood_full_model = m_full.log_likelihood()
    likelihood_sparse_model = 2 ** log_likelihood_sparse_model
    likelihood_full_model = 2 ** log_likelihood_full_model
    print(f"diff log likelihoods: {log_likelihood_full_model - log_likelihood_sparse_model}")
    print(f"diff likelihoods: {likelihood_full_model - likelihood_sparse_model}")

    # Show covar of inducing inputs and of full gp
    plot_covariance_matrix(m.posterior.covariance)

    # Test SVM
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X, y.flatten())
    z = clf.predict(X)
    plt.plot(X, z)
    plt.scatter(X, y)
    plt.title("SVM")
    plt.show()
