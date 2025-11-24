from time import time

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from tabulate import tabulate


def compute_noisy_or(X: ndarray, p_i: ndarray) -> ndarray:
    """
    TODO: Compute noisy-OR probabilities for each data row:
        P(Y = 1 | X) = 1 - prod_(i = 1)^n (1 - p_i)^(X_i).

    Args:
        X (ndarray): Input data of size T x n
        p_i (ndarray): Array of noisy-OR parameters of length n

    Returns:
        ndarray: Noisy-OR probabilities of size T x 1
    """
    T = X.shape[0]
    n = p_i.shape[0]
    noisy_or = np.zeros(T)
    for t in range(T):
        prod_i = 1
        for i in range(n):
            prob = (1 - p_i[i]) ** (X[t,i])
            prod_i *= prob
        noisy_or[t] = prod_i

    return 1 - noisy_or

def update_p_i(X: ndarray, y: ndarray, p_i: ndarray, noisy_or_prob: ndarray) -> ndarray:
    """
    TODO: Compute updated p_i values:
        p_i <- 1/T_i sum_(t = 1)^T P(Z_i = 1, X_i = 1 | X = x^(t), Y = y^(t)),
    where T_i is the number of rows in which X_i = 1 and
        P(Z_i = 1, X_i = 1 | X = x^(t), Y = y^(t)) = x_i^(t) y^(t) p_i / P(Y = 1 | X)
    is the posterior probability for the t-th row and i-th column.

    Args:
        X (ndarray): Input data of size T x n
        y (ndarray): Output data of size T x 1
        p_i (ndarray): Noisy-OR parameters of length n
        noisy_or_prob (ndarray): Noisy-OR probabilities of size T x 1

    Returns:
        ndarray: Updated p_i values of length n
    """
    T, n = X.shape
    T_i = np.sum(X == 1, axis=0)  # examples of every Xi = 1
    new_p = np.zeros(n)

    for i in range(n):
        numerator_sum = 0.0
        for t in range(T):
            if noisy_or_prob[t] > 0:  # avoid 0
                numerator_sum += (X[t, i] * y[t] * p_i[i]) / noisy_or_prob[t]
        if T_i[i] > 0:
            new_p[i] = numerator_sum / T_i[i]
        else: # get rid of the influence of X_i if it's always 0 in all examples
            new_p[i] = 0

    return new_p

def count_mistakes(y: ndarray, noisy_or_prob: ndarray) -> int:
    """
    TODO: Given the p_i values for a model, compute the number of mistakes (false positives and
    false negatives) made by the model.

    Args:
        y (ndarray): Output data of size T x 1
        noisy_or_prob (ndarray): Noisy-OR probabilities of size T x 1

    Returns:
        int: Number of mistakes made by the model.
    """
    mistakes = 0
    t = y.shape[0]
    for t in range(t):
        if (noisy_or_prob[t] >= 0.5 and y[t] == 0) or (noisy_or_prob[t] < 0.5 and y[t] == 1):
            mistakes += 1

    return mistakes

def compute_log_likelihood(X: ndarray, y: ndarray, p_i: ndarray) -> float:
    """
    TODO: Given the p_i values for a model, compute the normalized log-likelihood:
        L = 1/T sum_(t = 1)^T ln(P(Y = y^(t) | X = x^(t)))

    Args:
        X (ndarray): Input data of size T x n
        y (ndarray): Output data of size T x 1
        p_i (ndarray): Noisy-OR parameters of length n

    Returns:
        float: Normalized log-likelihood.
    """
    T = X.shape[0]
    noisy_or_prob = compute_noisy_or(X, p_i)  # P(Y=1|X), shape (T,)

    total_log = 0.0
    for t in range(T):
        if y[t] == 1:
            prob = noisy_or_prob[t]
        else:
            prob = 1.0 - noisy_or_prob[t]

        if prob <= 0.0:
            prob = 1e-12

        total_log += np.log(prob)

    return total_log / T

def read_data(x_file: str, y_file: str) -> tuple[ndarray, ndarray]:
    """
    Helper function to read input and output data from text files.

    DO NOT MODIFY
    """
    x_data = np.loadtxt(x_file, ndmin=2)
    y_data = np.loadtxt(y_file, ndmin=2)
    return x_data, y_data


def is_power_of_two(n: int) -> bool:
    """
    Helper function to check if a number is a power of two.
    Ref: https://stackoverflow.com/a/57025941

    DO NOT MODIFY
    """
    return n != 0 and (n & (n - 1)) == 0


def run(verbose: bool = False) -> DataFrame:
    """
    Helper function for running the EM algorithm and returning results as a DataFrame.
    By default, we initialize the parameters p_i to 0.05, to match the table in the HW 5 handout.

    DO NOT MODIFY

    Args:
        verbose (bool): Whether to print results of EM algorithm.

    Returns:
        DataFrame: EM algorithm results containing number of mistakes and (normalized)
            log-likelihood at each iteration.
    """
    start = time()
    X, y = read_data("spectX.txt", "spectY.txt")
    p_i = np.array([0.05] * X.shape[1])
    results: list[tuple[int, int, float]] = []

    noisy_or_prob = compute_noisy_or(X, p_i)
    mistakes = count_mistakes(y, noisy_or_prob)
    log_likelihood = compute_log_likelihood(X, y, p_i)
    results.append((0, mistakes, log_likelihood))

    for iter in range(1, 257):
        p_i = update_p_i(X, y, p_i, noisy_or_prob)
        noisy_or_prob = compute_noisy_or(X, p_i)
        if is_power_of_two(iter):
            mistakes = count_mistakes(y, noisy_or_prob)
            log_likelihood = compute_log_likelihood(X, y, p_i)
            results.append((iter, mistakes, log_likelihood))

    end = time()
    results_df = DataFrame(results, columns=["Iteration", "Mistakes", "Log-likelihood"])

    if verbose:
        print(f"EM algorithm completed in {end - start:.4f} secs")
        print(
            tabulate(
                results_df,
                headers="keys",
                # tablefmt="latex_raw",   # Uncomment if you want LaTeX code for this table
                showindex=False,
            )
        )

    return results_df


if __name__ == "__main__":
    run(verbose=True)
