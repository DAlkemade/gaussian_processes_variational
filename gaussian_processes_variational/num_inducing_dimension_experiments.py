import os
import pickle
import time

import numpy as np
import tqdm
from sklearn.metrics import mean_squared_error

from gaussian_processes_variational.compare import find_mse, diff_marginal_likelihoods, calc_K_tilda, find_logKy
from gaussian_processes_variational.experiment_data_containers import ExperimentResultsDimInd
from gaussian_processes_variational.gp_creation import create_full_gp, create_sparse_gp
from gaussian_processes_variational.non_gp_alternatives import bayesian_ridge_regression


def run_single_experiment(tag: str, kernel_type, simulator_type, n: int, dimensions: list, num_inducings: list,
                          fixedparameters,
                          fix_dimension_at: int = None, ARD=False):
    """Run experiment with changing number of inducing variables."""
    print(f'Running with kernel {kernel_type} and data simulator {simulator_type}')
    gp_kernel_type = kernel_type

    simulator = simulator_type(n)
    results = ExperimentResultsDimInd(dimensions, num_inducings, tag)

    # Increase the number of inducing inputs until n==m
    # Note that the runtime of a single iteration increases with num_inducing squared

    for i in tqdm.tqdm(range(len(dimensions))):
        dim = dimensions[i]
        n_informative = dim if fix_dimension_at is None else fix_dimension_at
        try:
            data = simulator.simulate(dim, n_informative=n_informative)
        except Exception:
            print("Data simulation went wrong, skipping this one")
            continue

        kernel_full = gp_kernel_type(dim, ARD=ARD)
        m_full = create_full_gp(data.X_train, data.y_train, kernel_full)
        results.logKy_full[i, :] = find_logKy(data.X_train, m_full)
        z_bayesian_ridge, _ = bayesian_ridge_regression(data.X_train, data.y_train, data.X_test)
        results.mse_bayesian_ridge[i, :] = mean_squared_error(np.ravel(data.y_test), z_bayesian_ridge)
        for j in range(len(num_inducings)):
            num_inducing = num_inducings[j]
            kernel_sparse = gp_kernel_type(dim, ARD=ARD)
            before = time.time()
            m_sparse = create_sparse_gp(data.X_train, data.y_train, kernel_sparse, num_inducing, fixedparameters)
            if fixedparameters.fix_variance:
                try:
                    m_sparse.kern.variance = m_full.kern.variance
                except AttributeError:
                    m_sparse.kern.variances = m_full.kern.variances
            if fixedparameters.fix_gaussian_noise_variance:
                m_sparse.Gaussian_noise.variance = m_full.Gaussian_noise.variance
            if fixedparameters.fix_lengthscale:
                try:
                    m_sparse.kern.lengthscale = m_full.kern.lengthscale
                except AttributeError:
                    pass # Some kernels don't have a lengthscale

            results.runtime[i, j] = time.time() - before
            mse_full = find_mse(m_full, data.X_test, data.y_test)
            mse_sparse = find_mse(m_sparse, data.X_test, data.y_test)
            divergence = diff_marginal_likelihoods(m_sparse, m_full, True)
            results.mses_full[i, j] = mse_full
            results.mses_sparse[i, j] = mse_sparse
            results.divergences[i, j] = divergence
            z = m_sparse.Z
            K_tilda = calc_K_tilda(kernel_sparse, data.X_train, z)
            results.traces[i, j] = np.trace(K_tilda)
            _, results.log_determinants[i, j] = np.linalg.slogdet(K_tilda)
            results.logKy_sparse[i, j] = find_logKy(data.X_train, m_sparse)

    pickle.dump(results, open(os.path.join('results', results.fname), "wb"))
