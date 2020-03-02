import numpy as np

from evaluate_experiment_increase_dimension import plot_heatmap


def test_nan_response():
    """Not a proper test, but test to see whether a heatmap with nan is possible"""
    values = np.full((4,5), 3.)
    values[1,2] = np.nan
    plot_heatmap(values, range(0,4), range(0,5))