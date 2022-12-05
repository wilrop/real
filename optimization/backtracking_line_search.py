import argparse

import jax
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guess", type=float, nargs='+', default=[0.75, -0.25],
                        help="The current guess for the optimal parameters.")
    parser.add_argument("--direction", type=float, nargs='+', default=[-1.5, 0.5], help="The descent direction.")
    parser.add_argument("--alpha", type=float, default=1., help="The starting step size.")
    parser.add_argument("--rho", type=float, default=0.5, help="The shrink parameter.")
    parser.add_argument("--c", type=float, default=0.5, help="The control parameter.")

    args = parser.parse_args()
    return args


def backtracking_line_search(f, x_k, grad_x_k, p_k, alpha=1., rho=0.5, c=0.5):
    """Backtracking line search to determine the step size.

    Args:
        f (callable): The function to minimise.
        x_k (ndarray): The current estimate.
        grad_x_k (ndarray): The gradient of the current estimate.
        p_k (ndarray): The direction to step towards.
        alpha (float, optional): The beginning step size.
        rho (float, optional): Shrink parameter.
        c (float, optional): Control parameter.

    Returns:
        float: A step size for the search direction.
    """
    func_eval = f(x_k)

    while f(x_k + alpha * p_k) > func_eval + c * alpha * np.dot(grad_x_k, p_k):
        alpha *= rho

    return alpha


def parabola(vec):
    return vec[0] ** 2 + vec[0] * vec[1] + vec[1] ** 2


if __name__ == '__main__':
    args = parse_args()
    f = parabola
    guess = np.array(args.guess)
    direction = np.array(args.direction)
    grad_x_k = jax.grad(f)(guess)
    alpha = backtracking_line_search(f, guess, grad_x_k, direction, alpha=args.alpha, rho=args.rho, c=args.c)

    print(f'Computed step size: {alpha}')
    new_x_k = guess + alpha * direction
    print(f'x_{{k+1}}: {new_x_k}')
    print(f'f(x_{{k+1}}) = {f(new_x_k)}')
