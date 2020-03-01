import matplotlib.pyplot as plt
import numpy as np
import pickle




class ExperimentResults(object):
    """Contains experiment results for the increasing dimension experiment."""
    def __init__(self, row_labels, column_labels):
        self.column_labels = column_labels
        self.row_labels = row_labels
        self.mses_full = self._init_results_matrix()
        self.mses_sparse = self._init_results_matrix()
        self.divergences = self._init_results_matrix()
        self.traces = self._init_results_matrix()
        self.log_determinants = self._init_results_matrix()
        self.runtime = self._init_results_matrix()

    @property
    def len_row_labels(self):
        return len(self.row_labels)

    @property
    def len_column_labels(self):
        return len(self.column_labels)

    @property
    def diff_mse(self):
        return np.subtract(self.mses_full, self.mses_sparse)

    def _init_results_matrix(self):
        return np.full((self.len_row_labels, self.len_column_labels), -1.)

class ExperimentResultsDimInd(ExperimentResults):
    def __init__(self, row_labels, column_labels):
        super().__init__(row_labels, column_labels)

    @property
    def dimensions(self):
        return self.row_labels

    @property
    def inducing_inputs(self):
        return self.column_labels


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
    fig.colorbar(im)
    plt.show()


def plot_experiment_results(results: ExperimentResultsDimInd):

    dimensions = results.dimensions
    num_inducings = results.inducing_inputs

    plot_heatmap(results.diff_mse, dimensions, num_inducings, decimals=4)

    plot_heatmap(results.divergences, dimensions, num_inducings, decimals=2)

    metric3 = np.subtract(results.traces, results.log_determinants)
    plot_heatmap(metric3, dimensions, num_inducings, decimals=4)

    plot_heatmap(results.traces, dimensions, num_inducings, decimals=4)

    plot_heatmap(results.runtime, dimensions, num_inducings, decimals=4)

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
