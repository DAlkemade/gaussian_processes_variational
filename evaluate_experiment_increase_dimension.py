import matplotlib.pyplot as plt
import numpy as np
import pickle


class ExperimentResults(object):
    """Contains experiment results for the increasing dimension experiment."""
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

    @property
    def diff_mse(self):
        return np.subtract(self.mses_full, self.mses_sparse)

    def _init_results_matrix(self):
        return np.full((self.len_dimensions, self.len_num_inducings_points), -1.)


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


def plot_experiment_results(results: ExperimentResults):

    dimensions = results.dimensions
    num_inducings = results.num_inducings

    plot_heatmap(results.diff_mse, dimensions, num_inducings, decimals=4)

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


if __name__ == '__main__':
    results = pickle.load(open('results.p', "rb"))
    plot_experiment_results(results)
