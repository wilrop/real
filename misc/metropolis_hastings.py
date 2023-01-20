import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000, help="The maximum number of iterations.")
    parser.add_argument("--burn_in", type=float, default=0.2, help="The fraction of iterations to discard as burn-in.")
    parser.add_argument("--symmetric", action='store_true', help="Whether to use a symmetric proposal distribution.")
    parser.add_argument("--seed", type=int, default=1, help="The seed for random number generation.")
    parser.add_argument("--bins", type=int, default=100, help="The number of bins for the histogram.")
    parser.add_argument("--mean", type=float, default=0, help="The mean of the normal distribution.")
    parser.add_argument("--std", type=float, default=1, help="The standard deviation of the normal distribution.")
    parser.add_argument("--min", type=float, default=0, help="The minimum value to plot the histogram over.")
    parser.add_argument("--max", type=float, default=10, help="The maximum value to plot the histogram over.")
    args = parser.parse_args()
    return args


def symmetric_acceptance_prob(f, proposed, x_prev, std):
    """Compute the acceptance probability for symmetric proposal distributions.

    Args:
        f (function): The function to estimate the density of.
        proposed (float): The proposed sample.
        x_prev (float): The previous sample.
        std (float): The standard deviation of the normal distribution.

    Returns:
        float: The acceptance probability.
    """
    return min(1, f(proposed) / f(x_prev))


def non_symmetric_acceptance_prob(f, proposed, x_prev, std):
    """Compute the acceptance probability for non-symmetric proposal distributions.

    Args:
        f (function): The function to estimate the density of.
        proposed (float): The proposed sample.
        x_prev (float): The previous sample.
        std (float): The standard deviation of the normal distribution.

    Returns:
        float: The acceptance probability.
    """
    return min(1, f(proposed) / f(x_prev) * stats.norm.pdf(x_prev, loc=proposed, scale=std) /
               stats.norm.pdf(proposed, loc=x_prev, scale=std))


def metropolis_hastings(f, iterations=1000, symmetric=True, burn_in=0.2, mean=0, std=1, seed=None):
    """The Metropolis-Hastings algorithm.

    Note:
        This uses the normal distribution as the proposal distribution. The conditional distribution is the normal
        distribution centred at the current sample with standard deviation 1.

    Args:
        f (function): The function to estimate the density of.
        iterations (int): The number of iterations to run the algorithm for.
        symmetric (bool): Whether to use a symmetric proposal distribution.
        burn_in (float): The fraction of iterations to discard as burn-in.
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.
        seed (int): The seed for random number generation.

    Returns:
        List[float]: The samples from the distribution.
    """
    print(f'Running Metropolis-Hastings with {iterations} iterations, {burn_in * 100}% burn-in')
    print(f'Using a {"symmetric" if symmetric else "non-symmetric"} proposal distribution')

    rng = np.random.default_rng(seed)
    accepted = []
    x = rng.normal(loc=mean, scale=std)
    accepted.append(x)

    if symmetric:
        calc_acceptance_prob = symmetric_acceptance_prob
    else:
        calc_acceptance_prob = non_symmetric_acceptance_prob

    for i in range(iterations):
        x_prev = accepted[-1]
        proposed = rng.normal(loc=x_prev, scale=std)
        acceptance_prob = calc_acceptance_prob(f, proposed, x_prev, std)
        if rng.random() <= acceptance_prob:
            accepted.append(proposed)
        else:
            accepted.append(x_prev)
    return accepted[int(burn_in * iterations):]


def exponential_dist(x):
    """The exponential distribution.

    Args:
        x (float): The value to evaluate the distribution at.

    Returns:
        float: The value of the distribution at this point.
    """
    return 0 if x < 0 else np.exp(-x)


def plot_hist(f, samples, min_x=0, max_x=10, bins=100):
    """Plot the histogram of the samples.

    Args:
        f (function): The function to estimate the density of.
        samples (List[float]): The samples to plot the histogram of.
        min_x (float): The minimum value to plot the histogram over.
        max_x (float): The maximum value to plot the histogram over.
        bins (int): The number of bins to use for the histogram.

    """
    fig, ax = plt.subplots()
    x = np.linspace(min_x, max_x, 1000)
    y = [f(x_i) for x_i in x]
    ax.plot(x, y)
    ax.hist(samples, bins=bins, density=True)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    samples = metropolis_hastings(exponential_dist, iterations=args.iterations, symmetric=args.symmetric,
                                  burn_in=args.burn_in, mean=args.mean, std=args.std, seed=args.seed)
    plot_hist(exponential_dist, samples, min_x=args.min, max_x=args.max, bins=args.bins)
