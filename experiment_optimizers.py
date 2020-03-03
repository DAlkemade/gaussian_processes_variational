import pickle
from collections import namedtuple

import time
import os
import GPy
import numpy as np
import tqdm

from gaussian_processes_variational.compare import diff_marginal_likelihoods, find_mse, calc_K_tilda
from evaluate_experiment_increase_dimension import plot_experiment_results
from gaussian_processes_variational.num_inducing_dimension_experiments import ExperimentResultsDimInd
from run_single import create_full_gp, create_sparse_gp
from gaussian_processes_variational.simulation import RBFSimulator, LinearSimulator, FriedMan1Simulator


def main():
    Experiment = namedtuple('Experiment', ['tag', 'simulator', 'kernel'])
    n = 801
    dim = 20
    num_inducing = 400
    experiments = [
        Experiment('linear', LinearSimulator, GPy.kern.Linear),
        Experiment('rbf', RBFSimulator, GPy.kern.RBF),
        Experiment('friedman', FriedMan1Simulator, GPy.kern.RBF)
    ]
    for experiment in experiments:
        run_single_experiment(experiment.tag, experiment.kernel, experiment.simulator, n, dim, num_inducing)


def run_single_experiment(tag: str, kernel_type, simulator_type, n: int, dim, num_inducing):
    """Run experiment with changing number of inducing variables."""
    print(f'Running with kernel {kernel_type} and data simulator {simulator_type}')
    gp_kernel_type = kernel_type
    max_num_inducing = n
    simulator = simulator_type(n)
    optimizers = ['simplex', 'lbfgsb', 'org-bfgs', 'scg', 'adadelta', 'rprop', 'adam']
    results = ExperimentResultsDimInd(optimizers, [])

    # Increase the number of inducing inputs until n==m
    # Note that the runtime of a single iteration increases with num_inducing squared
    j = 0
    for i in tqdm.tqdm(range(len(optimizers))):
        optimizer = optimizers[i]
        data = simulator.simulate(dim)
        kernel_full = gp_kernel_type(dim)
        m_full = create_full_gp(data.X_train, data.y_train, kernel_full, optimizer=optimizer)

        kernel_sparse = gp_kernel_type(dim)
        before = time.time()
        m_sparse = create_sparse_gp(data.X_train, data.y_train, kernel_sparse, num_inducing, optimizer=optimizer)
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

    fname = f"results_optimizers_{tag}.p"
    pickle.dump(results, open(os.path.join('results', fname), "wb"))
    plot_experiment_results(results)


if __name__ == "__main__":
    main()
