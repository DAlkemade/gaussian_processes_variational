import matplotlib.pyplot as plt

from main import create_full_gp, create_sparse_gp
from compare import KL_divergence, diff_marginal_likelihoods, find_mse
import tqdm
import sys

from simulation import RBFSimulator, LinearSimulator, FriedMan1Simulator


def main():
    """Run experiment with changing number of inducing variables."""
    simulator = RBFSimulator(100)
    data = simulator.simulate(3, n_informative=3)
    X, y = data.X_train, data.y_train
    try:
        m_full = create_full_gp(X, y, plot=False)
    except RuntimeWarning:
        print('Stopping programme, since we cannot rely on it if it has overflows')
        return
    mse_full = find_mse(m_full, data.X_test, data.y_test)
    print(f'MSE full: {mse_full}')
    results = []
    valid_x_values = []
    n = len(X)
    # Increase the number of inducing inputs until n==m
    inducing_inputs_nums = range(1, n + 1)
    # Note that the runtime of a single iteration increases with num_inducing squared
    mses = []
    print("Start experiment")
    for i in tqdm.trange(len(inducing_inputs_nums)):
        num = inducing_inputs_nums[i]
        try:
            m_sparse = create_sparse_gp(X, y, num_inducing=num, plot=False)
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


if __name__ == "__main__":
    main()
