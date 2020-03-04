import matplotlib.pyplot as plt
import numpy as np
import pickle
from decimal import Decimal

from gaussian_processes_variational.num_inducing_dimension_experiments import ExperimentResultsDimInd


def plot_experiment_results(results: ExperimentResultsDimInd):
    """Plot results of dimensions-inducing points experiment."""
    dimensions = results.dimensions
    num_inducings = results.inducing_inputs

    results.plot_diff_mse()

    results.plot_divergence()
    # For linear:
    # plot_heatmap(results.divergences, dimensions, num_inducings, remove_larger_than=20000, fname=f'{tag}_divergence', vmin=1.)

    results.plot_mse_sparse()

    slice_idx = 0
    print(f'Slice inducing points: {num_inducings[slice_idx]}')
    divergences_slice = results.divergences[:, slice_idx]
    plt.plot(dimensions, divergences_slice)
    plt.plot(dimensions, [0] * len(divergences_slice))
    plt.xlabel("Input dimension")
    plt.ylabel("Divergence")
    plt.show()

    plt.plot(dimensions, results.mses_sparse[:, slice_idx], label='sparse')
    plt.plot(dimensions, results.mses_full[:, slice_idx], label='full')
    plt.plot(dimensions, results.mse_bayesian_ridge[:, slice_idx], label='bayesian_ridge')
    plt.yscale('log')

    plt.xlabel("Input dimension")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    plt.plot(dimensions, results.logKy_full[:, slice_idx], label='sparse')
    plt.plot(dimensions, results.logKy_sparse[:, slice_idx], label='full')
    # plt.yscale('log')

    plt.xlabel("Input dimension")
    plt.ylabel("log|Ky|")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    results = pickle.load(open('results/results_linear_high_dim.p', "rb"))
    plot_experiment_results(results)
