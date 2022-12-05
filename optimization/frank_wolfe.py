import argparse

import jax
import numpy as np
from pulp import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=float, nargs='+', default=[-1., -1.],
                        help="The initial guess for the optimal parameters.")
    parser.add_argument("--domain", type=float, nargs='+', default=[-2, 2, -2, 2],
                        help="The domain for each parameter.")
    parser.add_argument("--tolerance", type=float, default=0.000001, help="The tolerance for the Frank-Wolfe gap.")
    parser.add_argument("--lipschitz", type=float, default=2., help="The Lipschitz constant of the function.")
    parser.add_argument("--iterations", type=int, default=100, help="The maximum number of iterations.")

    args = parser.parse_args()
    return args


def f1(vec):
    """The function f(x, y) = x*|y|.

    Note:
        This function has Lipschitz constant L=2. The minimum with -2 <= x, y <= 2 is -4 at (-2, -2) and (-2, 2).

    Args:
        vec (ndarray): An array where the first entry is x and the second entry is y.

    Returns:
        float: The function evaluation at this point.
    """
    return vec[0] * abs(vec[1])


def lin_problem(x_grad, dom):
    """The linear subproblem to solve in the Frank-Wolfe algorithm.

    Args:
        x_grad (ndarray): The gradient at the current estimate.
        dom (List[Tuple[float]]): A list of bounds for each variable.

    Returns:
        ndarray: The solution to the linear minimisation problem.
    """
    problem = LpProblem('mixtureDominance', LpMinimize)
    print(x_grad)

    s = []
    for i, dom_i in enumerate(dom):
        s.append(LpVariable(f's{i}', lowBound=dom_i[0], upBound=dom_i[1]))  # Set the variables in the correct bounds.

    problem += lpDot(list(x_grad), s)  # Minimise the dot product.

    problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    s_t = np.zeros(len(dom))
    for var in problem.variables():  # Get the weight values.
        if var.name.startswith('s'):
            s_t[int(var.name[-1])] = var.value()
    return s_t


def frank_wolfe(f, dom, init_x, delta, L=1, max_t=100):
    """Execute the Frank-Wolfe algorithm for nonlinear constrained optimization.

    Args:
        f (callable): The function to optimise.
        dom (List[tuple]): A list of domains for each variable in f.
        init_x (ndarray): An initial guess for the function.
        delta (float): A tolerance for the decrease between successive estimates.
        L (float, optional): A Lipschitz constant.
        max_t (int, optional): A maximum timestep.

    Returns:
        ndarray: The final estimate for the minimising x.
    """
    x_t = init_x
    grad_f = jax.jit(jax.grad(f))

    for i in range(max_t):
        print(f'Guess {i}: {x_t} - f({x_t}) = {f(x_t)}')
        grad_x_t = grad_f(x_t)
        s_t = lin_problem(grad_x_t, dom)
        d_t = s_t - x_t
        g_t = np.dot(-grad_x_t, d_t)  # Essentially means distance to a stationary point.

        if g_t < delta:
            break

        gamma_t = min(g_t / (L * np.dot(d_t, d_t)), 1)
        x_t += gamma_t * d_t

    return x_t


if __name__ == '__main__':
    args = parse_args()
    domain = [(x_min, x_max) for x_min, x_max in zip(args.domain[::2], args.domain[1::2])]
    init_x = np.array(args.init)
    x_opt = frank_wolfe(f1, domain, init_x, args.tolerance, L=args.lipschitz, max_t=args.iterations)
    print(f'Optimal value: {x_opt}')
