import argparse
import pickle

from evaluate_experiment_increase_dimension import plot_slice
from gaussian_processes_variational.experiment_data_containers import ExperimentResultsDimInd


def compare(results1, results2, slice_idx, tag1, tag2 ,tag_global):
    dimensions = results1.dimensions
    plot_slice(slice_idx, dimensions, [(results1.divergences, tag1), (results2.divergences, tag2)],
               'Input dimension', 'Divergence', show_xaxis=True, fname=f'{results1.tag}_{tag_global}_divergence', legend=True,
               log=True)

    plot_slice(slice_idx, dimensions, [(results1.mses_sparse, f'sparse_{tag1}'), (results1.mses_full, f'full_{tag1}'), (results2.mses_full, f'full_{tag2}'),
                                       (results1.mse_bayesian_ridge, 'bayesian_ridge'),
                                       (results2.mses_sparse, f'sparse_{tag2}')], "Input dimension", "MSE",
               legend=True, log=True, fname=f'{results1.tag}_{tag_global}_mse')

    plot_slice(slice_idx, dimensions,
               [(results1.logKy_sparse, f'sparse_{tag1}'), (results1.logKy_full, f'full_{tag1}'), (results2.logKy_full, f'full_{tag2}'),
                (results2.logKy_sparse, f'sparse_{tag2}')],
               'Input dimension', 'log|Ky|', legend=True, fname=f'{results1.tag}_{tag_global}_logKy', log=False)

    plot_slice(slice_idx, dimensions,
               [(results1.runtime, f'{tag1}'), (results2.runtime, f'{tag1}')],
               'Input dimension', 'runtime (s)', legend=True, fname=f'{results1.tag}_{tag_global}_runtime', log=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, default='results/results_rbf_no_ard.p')
    parser.add_argument('--path2', type=str, default='results/results_rbf_ard.p')
    args = parser.parse_args()
    results_normal: ExperimentResultsDimInd = pickle.load(open(args.path1, "rb"))
    results_abnormal: ExperimentResultsDimInd = pickle.load(open(args.path2, "rb"))
    slice_idx = 5
    print(f'Slice inducing points: {results_abnormal.inducing_inputs[slice_idx]}')
    compare(results_normal, results_abnormal, slice_idx, 'no_ard', 'ard', 'ard')

