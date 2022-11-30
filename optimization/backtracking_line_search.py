import jax
import numpy as np


def backtracking_line_search(f, x_k, grad_x_k, p_k, alpha=1., rho=0.5, c=0.5):
    """Backtracking line search to determine the step size.

    Args:
        f (callable): The function to minimise.
        x_k (ndarray): The current estimate.
        grad_x_k (ndarray): The gradient of the current estimate.
        p_k (ndarray): The direction to step towards.
        alpha (float): The beginning step size.
        rho (float): Shrink parameter.
        c (float): Control parameter.

    Returns:
        float: A step size for the search direction.
    """
    func_eval = f(x_k)

    while f(x_k + alpha * p_k) > func_eval + c * alpha * np.dot(grad_x_k, p_k):
        alpha *= rho

    return alpha


def parabola(x):
    return x[0] ** 2


def test_backtracking_line_search():
    f = parabola
    x_k = np.array([0.75])
    grad_x_k = jax.grad(f)(x_k)
    p_k = np.array([-3.])
    alpha = backtracking_line_search(f, x_k, grad_x_k, p_k)
    print(f'Computed step size: {alpha}')
    new_x_k = x_k + alpha * p_k
    print(f'x_{{k+1}}: {new_x_k}')
    print(f'f(x_{{k+1}}) = {f(new_x_k)}')


if __name__ == '__main__':
    test_backtracking_line_search()
