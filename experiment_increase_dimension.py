import pickle
from typing import Type

import GPy
import numpy as np
import tqdm
from GPy.kern import Kern

from compare import diff_marginal_likelihoods, find_mse
from evaluate_experiment_increase_dimension import plot_experiment_results, ExperimentResults
from main import create_full_gp, create_sparse_gp
from simulation import RBFSimulator


def calc_K_tilda(kernel: Type[Kern], X_train: np.array, X_m: np.array):
    """Find K_tilda = Cov(f|fm)."""
    Knn = kernel.K(X_train, X_train)
    Knm = kernel.K(X_train, X_m)
    Kmn = kernel.K(X_m, X_train)
    Kmm = kernel.K(X_m, X_m)
    temp = np.dot(np.dot(Knm, np.linalg.inv(Kmm)), Kmn)
    K_tilda = np.subtract(Knn, temp)
    return K_tilda


def main():
    """Run experiment with changing number of inducing variables."""
    gp_kernel_type = GPy.kern.RBF
    n = 801
    min_dim = 1
    max_dim = 2
    max_num_inducing = n
    dimensions = range(min_dim, max_dim + 1)
    num_inducings = range(1, max_num_inducing + 1, 50)
    simulator = RBFSimulator(n)
    results = ExperimentResults(dimensions, num_inducings)

    # Increase the number of inducing inputs until n==m
    # Note that the runtime of a single iteration increases with num_inducing squared

    for i in tqdm.tqdm(range(len(dimensions))):
        for j in range(len(num_inducings)):
            dim = dimensions[i]
            num_inducing = num_inducings[j]
            data = simulator.simulate(dim)
            kernel_sparse = gp_kernel_type(dim)
            m_sparse = create_sparse_gp(data.X_train, data.y_train, kernel_sparse, num_inducing)
            kernel_full = gp_kernel_type(dim)
            m_full = create_full_gp(data.X_train, data.y_train, kernel_full)
            mse_full = find_mse(m_full, data.X_test, data.y_test)
            mse_sparse = find_mse(m_sparse, data.X_test, data.y_test)
            divergence = diff_marginal_likelihoods(m_sparse, m_full, True)
            results.mses_full[i, j] = mse_full
            results.mses_sparse[i, j] = mse_sparse
            results.divergences[i, j] = divergence
            Z = m_sparse.Z
            K_tilda = calc_K_tilda(kernel_sparse, data.X_train, Z)
            results.traces[i, j] = np.trace(K_tilda)
            _, results.log_determinants[i, j] = np.linalg.slogdet(K_tilda)

    pickle.dump(results, open("results.p", "wb"))
    plot_experiment_results(results)


def calc_metric3(K_tilda):
    """Calculate difference between trace and determinant of K_tilda"""
    trace = np.trace(K_tilda)
    # determinant = np.linalg.det(K_tilda)
    _, log_determinant = np.linalg.slogdet(K_tilda)
    diff = trace - log_determinant
    print(trace, log_determinant, diff)
    return diff


if __name__ == "__main__":
    main()
