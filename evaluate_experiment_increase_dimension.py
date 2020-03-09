import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from gaussian_processes_variational.experiment_data_containers import ExperimentResultsDimInd


def plot_slice(slice_idx: int, x_values: np.array, array_tuples, xlabel: str, ylabel: str, legend=False,
               show_xaxis=False, log=False, fname: str = None):
    for array, label in array_tuples:
        slice = array[:, slice_idx]
        plt.plot(x_values, slice, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_xaxis:
        plt.axhline(y=0, color='k')
    if log:
        plt.yscale('log')
    if legend:
        plt.legend()

    if type(fname) is str:
        print(f'Saving {fname}')
        plt.savefig(os.path.join('images', fname))
    plt.show()


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
    plot_slice(slice_idx, dimensions, [(results.divergences, None)], 'Input dimension', 'Divergence', show_xaxis=True,
               log=True)

    plot_slice(slice_idx, dimensions, [(results.mses_sparse, 'sparse'), (results.mses_full, 'full'),
                                       (results.mse_bayesian_ridge, 'bayesian_ridge')], "Input dimension", "MSE",
               legend=True, log=True)

    plot_slice(slice_idx, dimensions, [(results.logKy_sparse, 'sparse'), (results.logKy_full, 'full')],
               'Input dimension', 'log|Ky|', legend=True)

    plot_slice(slice_idx, dimensions, [(results.traces, None)],
               'Input dimension', 'Tr(Ktilda)', legend=False, log=True)


if __name__ == '__main__':
    results = pickle.load(open('results/results_linear_high_dim.p', "rb"))
    plot_experiment_results(results)
