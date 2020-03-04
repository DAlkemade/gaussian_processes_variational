import numpy as np

from gaussian_processes_variational.experiment_data_containers import ExperimentResultsDimInd


def test_nan_response():
    """Not a proper test, but test to see whether a heatmap with nan is possible"""
    values = np.full((4,5), 3.)
    values[1,2] = np.nan
    results = ExperimentResultsDimInd(range(0,4), range(0,5), 'test')
    results.mses_sparse = values
    results.plot_mse_sparse(save=False)
