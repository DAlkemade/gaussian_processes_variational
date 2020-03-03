from collections import namedtuple

import GPy

from gaussian_processes_variational.num_inducing_dimension_experiments import run_single_experiment
from gaussian_processes_variational.simulation import LinearSimulator, FriedMan1Simulator


def main():
    """Run experiment with different dimensionalities and number of inducing inputs for a fixed number of
    informative dimensions.
    """
    Experiment = namedtuple('Experiment',
                            ['tag', 'simulator', 'kernel', 'dimensions', 'num_inducings', 'fix_dimension_at'])
    n = 801
    n_informative = 5
    inducing_points = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, n]
    dimensions = [5, 6, 7, 8, 9, 10, 15, 20]

    experiments = [
        Experiment('linear_fixed_informative', LinearSimulator, GPy.kern.Linear, dimensions, inducing_points
                   , fix_dimension_at=n_informative),
        Experiment('friedman_fixed_informative', FriedMan1Simulator, GPy.kern.RBF, dimensions,
                   inducing_points, fix_dimension_at=None)
    ]
    for experiment in experiments:
        run_single_experiment(experiment.tag, experiment.kernel, experiment.simulator, n, experiment.dimensions,
                              experiment.num_inducings, fix_dimension_at=experiment.fix_dimension_at)


if __name__ == "__main__":
    main()
