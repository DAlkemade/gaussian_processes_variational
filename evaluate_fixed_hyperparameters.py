import pickle

from evaluate_experiment_increase_dimension import plot_slice
from gaussian_processes_variational.experiment_data_containers import ExperimentResultsDimInd

if __name__ == '__main__':
    results_fixed: ExperimentResultsDimInd = pickle.load(open('results/results_linear_high_dim_fix_covariance.p', "rb"))
    results: ExperimentResultsDimInd = pickle.load(open('results/results_linear_high_dim.p', "rb"))
    slice_idx = 0
    dimensions = results.dimensions
    plot_slice(slice_idx, dimensions, [(results.divergences, 'Unconstrained'), (results_fixed.divergences, 'Fixed')],
               'Input dimension', 'Divergence', show_xaxis=True, fname=f'{results.tag}_fixed_divergence', legend=True)

    plot_slice(slice_idx, dimensions, [(results.mses_sparse, 'sparse_unconstrained'), (results_fixed.mses_full, 'full'),
                                       (results.mse_bayesian_ridge, 'bayesian_ridge'),
                                       (results_fixed.mses_sparse, 'sparse_fixed')], "Input dimension", "MSE",
               legend=True, log=True, fname=f'{results.tag}_fixed_mse')

    plot_slice(slice_idx, dimensions,
               [(results.logKy_sparse, 'sparse_unconstrained'), (results_fixed.logKy_full, 'full'),
                (results_fixed.logKy_sparse, 'sparse_fixed')],
               'Input dimension', 'log|Ky|', legend=True, fname=f'{results.tag}_fixed_logKy')
