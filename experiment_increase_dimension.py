import pickle
from collections import namedtuple

import time
import os
import GPy
import numpy as np
import tqdm

from compare import diff_marginal_likelihoods, find_mse, calc_K_tilda
from evaluate_experiment_increase_dimension import plot_experiment_results, ExperimentResultsDimInd
from run_single import create_full_gp, create_sparse_gp
from simulation import RBFSimulator, LinearSimulator


def main():
    """Run experiment for different datasets where a grid of number of inducings points and dimensions is explored."""
    Experiment = namedtuple('Experiment', ['tag', 'simulator', 'kernel', 'dimensions', 'num_inducings'])
    n = 801
    inducing_points = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, n]
    dimensions = [1, 2, 3, 4, 5, 10, 15, 20]

    experiments = [
        Experiment('linear', LinearSimulator, GPy.kern.Linear, dimensions, inducing_points),
        Experiment('rbf', RBFSimulator, GPy.kern.RBF, dimensions, inducing_points),
    ]
    for experiment in experiments:
        run_single_experiment(experiment.tag, experiment.kernel, experiment.simulator, n, experiment.dimensions,
                              experiment.num_inducings)


def run_single_experiment(tag: str, kernel_type, simulator_type, n: int, dimensions: list, num_inducings: list,
                          fix_dimension_at: int = None):
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

    pickle.dump(results, open(os.path.join('results', results.fname), "wb"))


if __name__ == "__main__":
    main()
