from collections import namedtuple

import GPy

from experiment_increase_dimension import run_single_experiment
from simulation import LinearSimulator


def main():
    Experiment = namedtuple('Experiment', ['tag', 'simulator', 'kernel', 'dimensions', 'num_inducings'])
    n = 801
    dimensions = [1, 2, 3, 4, 5, 6,7,8,9,10] + list(range(15, n, 50)) + [891]

    experiments = [
        Experiment('linear_high_dim', LinearSimulator, GPy.kern.Linear, dimensions,
                   [50]),
    ]
    for experiment in experiments:
        run_single_experiment(experiment.tag, experiment.kernel, experiment.simulator, n, experiment.dimensions,
                              experiment.num_inducings)


if __name__ == "__main__":
    main()
