import numpy as np


def find_grid(N):
    """
    Find the number of rows and columns for a grid of cores such that
    the total number of tiles N can be evenly divided among the cores.
    Each core will have the same integer number of tiles.
    The grid size is limited to a maximum of 2 rows and 8 columns.
    Parameters:
        N (int): Total number of tiles to be distributed.
        Returns:
            tuple: A tuple (rows, cols) representing the grid dimensions.

        Raises:
            AssertionError: If it's not possible to find such a grid configuration.
    """
    max_rows = 8
    max_cols = 8
    max_cores = max_rows * max_cols

    # Find all possible numbers of cores that divide N and are less than or equal to max_cores
    target = 32
    possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
    possible_cores.sort(key=lambda x: abs(x - target))  # Sort by closest to target

    for cores in possible_cores:
        # Try to find a grid configuration with the current number of cores
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols


if __name__ == "__main__":
    X = find_grid(8192 // 32)
    print(X)
