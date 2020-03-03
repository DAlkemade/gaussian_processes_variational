import os
import pickle
import time

import GPy
import matplotlib.pyplot as plt
import numpy as np
from GPy.models import SparseGPRegression
from tqdm import trange

from simulation import LinearSimulator, RBFSimulator


class RuntimeResult(object):
    """Stores results of the runtime experiments and saves and plots them."""

    def __init__(self, kernel_time, gp_time, opt_time, dimensions, label: str):
        self.dimensions = dimensions
        self.label = label
        self.opt_time = opt_time
        self.gp_time = gp_time
        self.kernel_time = kernel_time

    def plot(self):
        """Plot results for different parts of the process in one log plot."""
        plt.plot(self.dimensions, self.kernel_time, label='Kernel runtime')
        plt.plot(self.dimensions, self.gp_time, label='GP runtime')
        plt.plot(self.dimensions, self.opt_time, label='Optimization runtime')
        plt.yscale('log')
        plt.xlabel('Dimension')
        plt.ylabel('Runtime (s)')
        plt.legend()
        plt.show()


def main():
    """Run runtime experiment."""
    num_inducing = 50
    n = 1000
    dimensions = range(1, 101, 10)

    linear_result = compute_runtimes(dimensions, n, num_inducing, GPy.kern.Linear, LinearSimulator, 'linear')
    rbf_result = compute_runtimes(dimensions, n, num_inducing, GPy.kern.RBF, RBFSimulator, 'rbf')

    pickle.dump(linear_result, open(os.path.join('results', f'{linear_result.label}_runtime.p'), "wb"))
    pickle.dump(rbf_result, open(os.path.join('results', f'{rbf_result.label}_runtime.p'), "wb"))

    linear_result.plot()
    rbf_result.plot()


def compute_runtimes(dimensions, n, num_inducing, kernel_class, simulator_class, tag: str):
    """Compute runtimes at different dimensionalities for sample_size iterations at every dimension."""
    sample_size = 10
    kernel_creation = []
    gp_creation = []
    gp_optimization = []
    for i in trange(len(dimensions)):
        dim = dimensions[i]
        kernel_creation_array = np.empty(sample_size)
        gp_creation_array = np.empty(sample_size)
        gp_optimization_array = np.empty(sample_size)
        for j in range(sample_size):
            simulator = simulator_class(n, random_state=j)
            data = simulator.simulate(dim, n_informative=n)

            a = time.time()
            kernel = kernel_class(dim)
            b = time.time()
            m = SparseGPRegression(data.X_train, data.y_train, num_inducing=num_inducing,
                                   kernel=kernel)
            c = time.time()
            m.optimize()
            d = time.time()
            kernel_creation_array[j] = b - a
            gp_creation_array[j] = c - b
            gp_optimization_array[j] = d - c

        kernel_creation.append(kernel_creation_array.mean())
        gp_creation.append(gp_creation_array.mean())
        gp_optimization.append(gp_optimization_array.mean())

    return RuntimeResult(kernel_creation, gp_creation, gp_optimization, dimensions, tag)


if __name__ == '__main__':
    main()
