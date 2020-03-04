from collections import namedtuple

import GPy

from gaussian_processes_variational.num_inducing_dimension_experiments import run_single_experiment
from gaussian_processes_variational.parameter_containers import FixedParameterSettings
from gaussian_processes_variational.simulation import RBFSimulator, LinearSimulator


def main():
    """Run experiment for different datasets where a grid of number of inducings points and dimensions is explored."""
    Experiment = namedtuple('Experiment', ['tag', 'simulator', 'kernel', 'dimensions', 'num_inducings'])
    n = 801
    inducing_points = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, n]
    dimensions = [1, 2, 3, 4, 5, 10, 15, 20]

    experiments = [
        Experiment('rbf_fix_covariance', RBFSimulator, GPy.kern.RBF, dimensions, inducing_points),
        Experiment('linear_fix_covariance', LinearSimulator, GPy.kern.Linear, dimensions, inducing_points),
    ]
    opt_settings = FixedParameterSettings(fix_variance=True, fix_gaussian_noise_variance=True, fix_lengthscale=True)
    for experiment in experiments:
        run_single_experiment(experiment.tag, experiment.kernel, experiment.simulator, n, experiment.dimensions,
                              experiment.num_inducings, opt_settings)


if __name__ == "__main__":
    main()
