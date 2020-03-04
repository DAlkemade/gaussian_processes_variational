from gaussian_processes_variational.non_gp_alternatives import bayesian_ridge_regression


def test_bayesian_ridge_regression_simple():
    z, std = bayesian_ridge_regression([[0, 0], [1, 1], [2, 2]], [0, 1, 2], [[1, 1],[1.5,1.5]])
    assert z[1] > z[0]
    assert std [1] > std[0]
