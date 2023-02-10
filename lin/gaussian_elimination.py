import numpy as np


def gaussian_elimination(system):
    """Solution of a system of linear equations using Gaussian elimination.

    Note:
        This function implements the pseudocode from wikipedia which is said to increase numerical stability.

    Args:
        system (ndarray): A system of linear equations.

    Returns:
        ndarray: A matrix in reduced row echelon form.
    """
    # Make a copy of the system.
    solution = np.array(system, dtype=np.float64)

    # Get the number of rows and columns.
    rows, cols = solution.shape

    row_pivot = 0
    col_pivot = 0

    while row_pivot < rows and col_pivot < cols:
        # Find the row with the largest pivot.
        i_max = np.argmax(np.abs(solution[row_pivot:, col_pivot])) + row_pivot
        if solution[i_max, col_pivot] == 0:
            # The column is already in reduced row echelon form.
            col_pivot += 1
        else:
            # Swap the rows.
            solution[[row_pivot, i_max]] = solution[[i_max, row_pivot]]

            # Divide the row by the pivot.
            solution[row_pivot] /= solution[row_pivot, col_pivot]

            # Subtract the row from all other rows.
            for i in range(rows):
                if i != row_pivot:
                    solution[i] -= solution[row_pivot] * solution[i, col_pivot]

            row_pivot += 1
            col_pivot += 1
    return solution


def back_substitution(solution):
    """Perform back substitution on a matrix in reduced row echelon form.

    Args:
        solution (ndarray): A matrix in reduced row echelon form.

    Returns:
        ndarray: The solution to the system of linear equations.
    """
    rows, cols = solution.shape
    sub_sol = np.zeros(rows)

    for row_idx in reversed(range(rows)):
        xi = 1 / solution[row_idx, row_idx] * (
                solution[row_idx, -1] - np.sum(solution[row_idx, row_idx + 1:-1] * sub_sol[row_idx + 1:]))
        sub_sol[row_idx] = xi
    return sub_sol


def compare(system, solution):
    """Compare the solution to the system of linear equations.

    Args:
        system (ndarray): A system of linear equations.
        solution (ndarray): The solution to the system of linear equations.

    Returns:
        bool: Whether the solution is correct.
    """
    correct = np.linalg.solve(system[:, :-1], system[:, -1])
    return np.allclose(correct, solution)


if __name__ == '__main__':
    system = np.array([[9, 3, 4, 7], [4, 3, 4, 8], [1, 1, 1, 3]])
    solution = gaussian_elimination(system)
    sub_sol = back_substitution(solution)
    is_correct = compare(system, sub_sol)
    print(f'Solution: {sub_sol}')
    print(f'Is the solution correct? {is_correct}')
