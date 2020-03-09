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
    dimensions_high = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + list(range(15, n, 50)) + [891]
    ARD = True

    experiments = [
        # Experiment('rbf_ard', RBFSimulator, GPy.kern.RBF, dimensions, inducing_points),
        Experiment('linear_high_dim_ard', LinearSimulator, GPy.kern.Linear, dimensions_high,
                   [50]),
    ]
    for experiment in experiments:
        run_single_experiment(experiment.tag, experiment.kernel, experiment.simulator, n, experiment.dimensions,
                              experiment.num_inducings, FixedParameterSettings(), ARD=ARD)


if __name__ == "__main__":
    main()
