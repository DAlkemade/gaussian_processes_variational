import GPy

import matplotlib.pyplot as plt

from compare import KL_divergence, diff_marginal_likelihoods, find_mse
import tqdm

from main import create_sparse_gp, create_full_gp
from simulation import RBFSimulator, LinearSimulator, FriedMan1Simulator


def main():
    """Run experiment with changing number of inducing variables."""
    simulator = RBFSimulator(500)
    n_input = 5
    data = simulator.simulate(n_input, n_informative=n_input)
    X, y = data.X_train, data.y_train
    kernel_full = GPy.kern.RBF(n_input)
    try:
        m_full = create_full_gp(X, y, kernel_full, plot=False)
    except RuntimeWarning:
        print('Stopping programme, since we cannot rely on it if it has overflows')
        return
    mse_full = find_mse(m_full, data.X_test, data.y_test)
    print(f'MSE full: {mse_full}')
    results = []
    valid_x_values = []
    n = len(X)
    # Increase the number of inducing inputs until n==m
    inducing_inputs_nums = range(1, n+1, 5)
    # Note that the runtime of a single iteration increases with num_inducing squared
    mses = []
    for i in tqdm.tqdm(inducing_inputs_nums):
        num = i
        try:
            kernel_sparse = GPy.kern.RBF(n_input)
            m_sparse = create_sparse_gp(X, y, kernel_sparse, plot=False, num_inducing=num)
        except Exception as error:
            print(f'Error {error} at num={num}')
            continue
        except RuntimeWarning as warning:
            print(f'Warning {warning} at num={num}')
            continue

        try:
            divergence = diff_marginal_likelihoods(m_sparse, m_full, True)
            mse = find_mse(m_sparse, data.X_test, data.y_test)
        except RuntimeWarning as warning:
            print(f'Warning calculating divergence or MSE num={num}, not using this datapoint')
            continue

        mses.append(mse)
        valid_x_values.append(num)
        results.append(divergence)
        # print(f'MSE: {mse} divergence: {divergence}')

    print(results)
    print(mses)
    plt.plot(valid_x_values, results)
    plt.plot(valid_x_values, [0] * len(valid_x_values))
    # plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Number of inducing variables")
    plt.ylabel("KL(p||q)")
    plt.show()

    plt.plot(valid_x_values, mses, label='sparse')
    plt.plot(valid_x_values, [mse_full] * len(valid_x_values), label='full')
    # plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Number of inducing variables")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    valid_x_values = valid_x_values[1:]
    mses = mses[1:]
    plt.plot(valid_x_values, mses, label='sparse')
    plt.plot(valid_x_values, [mse_full] * len(valid_x_values), label='full')
    # plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Number of inducing variables")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
