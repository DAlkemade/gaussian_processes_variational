import pickle
from collections import namedtuple
from typing import Type

import time
import os
import GPy
import numpy as np
import tqdm
from GPy.kern import Kern

from compare import diff_marginal_likelihoods, find_mse
from evaluate_experiment_increase_dimension import plot_experiment_results, ExperimentResultsDimInd
from main import create_full_gp, create_sparse_gp
from simulation import RBFSimulator, LinearSimulator, FriedMan1Simulator


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
    Experiment = namedtuple('Experiment', ['tag', 'simulator', 'kernel', 'min_dim', 'max_dim'])
    n = 801
    experiments = [
        Experiment('linear', LinearSimulator, GPy.kern.Linear, 1, 20),
        Experiment('rbf', RBFSimulator, GPy.kern.RBF, 1, 20),
        Experiment('friedman', FriedMan1Simulator, GPy.kern.RBF, 5, 20)
    ]
    for experiment in experiments:
        run_single_experiment(experiment.tag, experiment.kernel, experiment.simulator, n, experiment.min_dim, experiment.max_dim)


def run_single_experiment(tag: str, kernel_type, simulator_type, n: int, min_dim: int, max_dim: int):
    """Run experiment with changing number of inducing variables."""
    print(f'Running with kernel {kernel_type} and data simulator {simulator_type}')
    gp_kernel_type = kernel_type
    max_num_inducing = n
    dimensions = range(min_dim, max_dim + 1)
    num_inducings = range(1, max_num_inducing + 1, 50)
    simulator = simulator_type(n)
    results = ExperimentResultsDimInd(dimensions, num_inducings)

    # Increase the number of inducing inputs until n==m
    # Note that the runtime of a single iteration increases with num_inducing squared

    for i in tqdm.tqdm(range(len(dimensions))):
        dim = dimensions[i]
        data = simulator.simulate(dim)
        kernel_full = gp_kernel_type(dim)
        m_full = create_full_gp(data.X_train, data.y_train, kernel_full)
        for j in range(len(num_inducings)):
            num_inducing = num_inducings[j]
            kernel_sparse = gp_kernel_type(dim)
            before = time.time()
            m_sparse = create_sparse_gp(data.X_train, data.y_train, kernel_sparse, num_inducing)
            results.runtime[i, j] = time.time() - before
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

    fname = f"results_{tag}.p"
    pickle.dump(results, open(os.path.join('results', fname), "wb"))
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
