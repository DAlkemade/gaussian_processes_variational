import matplotlib.pyplot as plt

from main import create_full_gp, create_sparse_gp
from compare import KL_divergence, diff_marginal_likelihoods
from simulation import simulate_data
import tqdm

USE_MARGINAL_LIKELIHOOD_DIFF = True


def main():
    """Run experiment with changing number of inducing variables."""
    data = simulate_data(500, 1)
    X, y = data.X_train, data.y_train
    m_full = create_full_gp(X, y, plot=False)
    results = []
    n = len(X)
    # Increase the number of inducing inputs until n==m
    inducing_inputs_nums = range(1, n + 1)
    for i in tqdm.trange(len(inducing_inputs_nums)):
        num = inducing_inputs_nums[i]
        m_sparse = create_sparse_gp(X, y, num_inducing=num, plot=False)
        if USE_MARGINAL_LIKELIHOOD_DIFF:
            divergence = diff_marginal_likelihoods(m_sparse, m_full, True)
        else:
            samples = X
            divergence = KL_divergence(m_sparse, m_full, samples)
        results.append(divergence)
    plt.plot(inducing_inputs_nums, results)
    plt.plot(inducing_inputs_nums, [0] * n)
    plt.title("KL divergence w.r.t the number of inducing variables")
    plt.xlabel("Number of inducing variables")
    plt.ylabel("KL(p||q)")
    plt.show()


if __name__ == "__main__":
    main()
