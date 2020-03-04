# The following notebook was used as the starting point: https://github.com/SheffieldML/notebook/blob/master/GPy/sparse_gp_regression.ipynb
import configparser
import os
from argparse import ArgumentParser

import GPy
import numpy as np

from gaussian_processes_variational.compare import find_logKy
from gaussian_processes_variational.non_gp_alternatives import fit_svm, linear_regression
from gaussian_processes_variational.simulation import LinearSimulator, FriedMan1Simulator, RBFSimulator
from gaussian_processes_variational.gp_creation import create_full_gp, create_kernel, create_sparse_gp, \
    evaluate_sparse_gp

np.random.seed(101)

RBF = 'rbf'
LINEAR = 'linear'
GPy.plotting.change_plotting_library('matplotlib')


def main():
    """Run the experiment using a certain config defined in the config file."""
    # Read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='playground.ini')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', args.config))
    input_dim = config['DATA'].getint('input_dim')
    n_samples = config['DATA'].getint('n')
    simulation_function_string = config['DATA']['simulation_function']
    num_inducing = config['SPARSE_GP'].getint('num_inducing')
    if config['GP']['kernel'] == RBF:
        kernel_class = GPy.kern.RBF
    elif config['GP']['kernel'] == LINEAR:
        kernel_class = GPy.kern.Linear
    else:
        raise ValueError("Unknown kernel")
    plot = input_dim <= 1
    # Sample function
    if simulation_function_string == 'make_regression':
        n_informative = config['DATA'].getint('input_dim_informative')
        if n_informative is None:
            n_informative = input_dim
        simulator = LinearSimulator(n_samples)
        data = simulator.simulate(input_dim, n_informative=n_informative)
    elif simulation_function_string == 'make_friedman1':
        simulator = FriedMan1Simulator(n_samples)
        data = simulator.simulate(input_dim)
    elif simulation_function_string == 'rbf':
        simulator = RBFSimulator(n_samples)
        data = simulator.simulate(input_dim)
    else:
        raise ValueError("Unknown simulation function given")

    # Create kernels

    kernel = create_kernel(data.X_train, kernel_class)
    kernel_sparse = create_kernel(data.X_train, kernel_class)

    # Create GPs

    m_full = create_full_gp(data.X_train, data.y_train, kernel, plot=plot)
    m_sparse = create_sparse_gp(data.X_train, data.y_train, kernel_sparse, num_inducing, plot=plot)

    # Evaluate
    evaluate_sparse_gp(data.X_test, data.y_test, m_sparse, m_full)
    print(find_logKy(data.X_train, m_sparse))

    # Test SVM and Linear Regression
    fit_svm(data.X_train, data.y_train, plot=plot)
    linear_regression(data.X_train, data.y_train, plot=plot)


if __name__ == "__main__":
    main()
