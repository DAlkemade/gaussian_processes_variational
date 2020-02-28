import matplotlib.pyplot as plt

from compare import KL_divergence, diff_marginal_likelihoods, find_mse
import tqdm
import GPy

from main import create_full_gp, create_sparse_gp
from simulation import RBFSimulator, LinearSimulator, FriedMan1Simulator
import numpy as np


def main():
    """Run experiment with changing number of inducing variables."""
    n = 400
    max_dim = 3
    max_num_inducing = 200
    dimensions = range(1, max_dim + 1)
    num_inducings = range(1, max_num_inducing + 1, 50)
    simulator = RBFSimulator(n)
    mses_full = np.full((len(dimensions), len(num_inducings)), -1.)
    mses_sparse = np.full((len(dimensions), len(num_inducings)), -1.)
    divergences = np.full((len(dimensions), len(num_inducings)), -1.)

    # Increase the number of inducing inputs until n==m
    # Note that the runtime of a single iteration increases with num_inducing squared
    print("Start experiment")

    for i in tqdm.tqdm(range(len(dimensions))):
        for j in range(len(num_inducings)):
            dim = dimensions[i]
            num_inducing = num_inducings[j]
            data = simulator.simulate(dim)
            kernel_sparse = GPy.kern.RBF(dim)
            m_sparse = create_sparse_gp(data.X_train, data.y_train, kernel_sparse, num_inducing)
            kernel_full = GPy.kern.RBF(dim)
            m_full = create_full_gp(data.X_train, data.y_train, kernel_full)
            mse_full = find_mse(m_full, data.X_test, data.y_test)
            mse_sparse = find_mse(m_sparse, data.X_test, data.y_test)
            divergence = diff_marginal_likelihoods(m_sparse, m_full, True)
            print(mse_full, mse_sparse)
            mses_full[i, j] = mse_full
            mses_sparse[i, j] = mse_sparse
            divergences[i, j] = divergence

    diff_mse = np.subtract(mses_full, mses_sparse)
    diff_mse = np.round(diff_mse, decimals=3)

    plot_heatmap(diff_mse, dimensions, num_inducings)

    divergences_rounded = np.round(divergences, decimals=1)
    plot_heatmap(divergences_rounded, dimensions, num_inducings)

    slice_idx = 0
    divergences_slice = divergences[:, slice_idx]
    plt.plot(dimensions, divergences_slice)
    plt.plot(dimensions, [0] * len(divergences_slice))
    # plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Input dimension")
    plt.ylabel("Divergence")
    plt.legend()
    plt.show()

    plt.plot(dimensions, mses_sparse[:, slice_idx], label='sparse')
    plt.plot(dimensions, mses_full[:, slice_idx], label='full')

    # plt.plot(dimensions, [0] * len(dimensions))
    # plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Input dimension")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def plot_heatmap(values_matrix, yvalues, xvalues):
    fig, ax = plt.subplots()
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
