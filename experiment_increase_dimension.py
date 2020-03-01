from collections import namedtuple

import matplotlib.pyplot as plt
from GPy.kern import Kern
from GPy.models import SparseGPRegression

from compare import KL_divergence, diff_marginal_likelihoods, find_mse
import tqdm
import GPy

from main import create_full_gp, create_sparse_gp
from simulation import RBFSimulator, LinearSimulator, FriedMan1Simulator
import numpy as np
from typing import Type


class ExperimentResults(object):
    def __init__(self, dimensions, num_inducings):
        self.num_inducings = num_inducings
        self.dimensions = dimensions
        self.mses_full = self._init_results_matrix()
        self.mses_sparse = self._init_results_matrix()
        self.divergences = self._init_results_matrix()
        self.traces = self._init_results_matrix()
        self.log_determinants = self._init_results_matrix()

    @property
    def len_dimensions(self):
        return len(self.dimensions)

    @property
    def len_num_inducings_points(self):
        return len(self.num_inducings)

    def _init_results_matrix(self):
        return np.full((self.len_dimensions, self.len_num_inducings_points), -1.)


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

    diff_mse = np.subtract(results.mses_full, results.mses_sparse)
    print(diff_mse)
    diff_mse = np.round(diff_mse, decimals=4)
    print(diff_mse)

    plot_heatmap(diff_mse, dimensions, num_inducings)

    plot_heatmap(results.divergences, dimensions, num_inducings, decimals=2)

    metric3 = np.subtract(results.traces, results.log_determinants)
    plot_heatmap(metric3, dimensions, num_inducings, decimals=4)

    plot_heatmap(results.traces, dimensions, num_inducings, decimals=4)

    slice_idx = 0
    divergences_slice = results.divergences[:, slice_idx]
    plt.plot(dimensions, divergences_slice)
    plt.plot(dimensions, [0] * len(divergences_slice))
    # plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Input dimension")
    plt.ylabel("Divergence")
    plt.show()

    plt.plot(dimensions, results.mses_sparse[:, slice_idx], label='sparse')
    plt.plot(dimensions, results.mses_full[:, slice_idx], label='full')

    # plt.plot(dimensions, [0] * len(dimensions))
    # plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Input dimension")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def calc_metric3(K_tilda):
    """Calculate difference between trace and determinant of K_tilda"""
    trace = np.trace(K_tilda)
    # determinant = np.linalg.det(K_tilda)
    _, log_determinant = np.linalg.slogdet(K_tilda)
    diff = trace - log_determinant
    print(trace, log_determinant, diff)
    return diff


def plot_heatmap(values_matrix, yvalues, xvalues, decimals=None):
    if type(decimals) is int:
        values_matrix = np.round(values_matrix, decimals=decimals)
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(values_matrix)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xvalues)))
    ax.set_yticks(np.arange(len(yvalues)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xvalues)
    ax.set_yticklabels(yvalues)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(yvalues)):
        for j in range(len(xvalues)):
            text = ax.text(j, i, values_matrix[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.xlabel("Number of inducing inputs")
    plt.ylabel("Number of dimensions")
    plt.show()


if __name__ == "__main__":
    main()
