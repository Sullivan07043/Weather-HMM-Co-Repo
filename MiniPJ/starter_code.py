# allowed imports
from numpy.typing import NDArray
from typing import Callable
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import random


def load_data(
    ratings_file: str = "ratings.txt",
    movies_file: str = "movies.txt",
    probZ_init_file: str = "probZ_init.txt",
    probR_init_file: str = "probR_init.txt",
) -> tuple[NDArray, list[str], NDArray, NDArray]:
    """
    Loads all required data files (ratings, movie names, and initial probabilities)
    into the specified NumPy and list structures.

    This function is responsible for reading data from five separate text files
    and converting their contents into the exact Python/NumPy types required
    for the rest of the program.

    :param ratings_file: Path to the file containing user ratings (e.g., "ratings.txt").
    :param movies_file: Path to the file containing movie names (e.g., "movies.txt").
    :param probZ_init_file: Path to the file containing initial probabilities for Z (e.g., "probZ_init.txt").
    :param probR_init_file: Path to the file containing initial probabilities for R|Z (e.g., "probR_init.txt").
    :returns: A tuple containing the four processed data structures in the following order:
        1. ratings (NDArray): A 2D NumPy array where:
           - Rows correspond to users.
           - Columns correspond to movies.
           - Ratings are mapped: "0" -> 0 (Seen, does not recommend), "1" -> 1 (Seen, Recommend), "?" -> -1 (Not seen).
        2. movie_idx_to_name (list[str]): A 1D list of movie names, where the index
           corresponds to the movie's column index in the 'ratings' array.
        3. Z_init (NDArray, Shape=(k,)): A 1D NumPy array of initial probabilities $P(Z=i)$,
           where each value is a float between 0 and 1.
        4. R_init (NDArray, Shape=(M, k)): A 2D NumPy array (matrix) of initial probabilities $P(Rj=1|Z=i)$,
           where each value is a float between 0 and 1.
    """

    # TODO: load data

    # Load ratings
    with open(ratings_file, "r") as f:
        ratings = []
        for line in f:
            items = line.strip().split()
            row = [int(x) if x != "?" else -1 for x in items]
            ratings.append(row)
    ratings = np.array(ratings, dtype=int)

    # Load movie names
    with open(movies_file, "r") as f:
        movie_idx_to_name = [line.strip() for line in f]

    # Load initial P(Z=i)
    Z_init = np.loadtxt(probZ_init_file, dtype=float)
    Z_init = Z_init.reshape(-1)

    # Load initial P(R_j = 1 | Z=i)
    R_init = np.loadtxt(probR_init_file, dtype=float)

    return ratings, movie_idx_to_name, Z_init, R_init


def mean_popularity_rating(
    ratings: NDArray, movie_idx_to_name: list[str]
) -> dict[str, float]:
    """
    Given ratings and movie idx to names, return a dictionary of each movie to
    their associated mean popularity rating.

    `ratings` is a 2D Numpy Array, where:
        ratings[t][j] is the t-th student's rating, for the j-th movie. Both t and
        j are 0-indexed. The corresponding value should be either:
        - 0 (not recommended)
        - 1 (recommended)
        - -1 (no recommendation)
    `movie_idx_to_name` is a list of strings, where:
        movie_idx_to_name[j] is the j-th movie's name, 0-indexed.

    Return:
    `mean_ratings`, a dictionary of mean popularity ratings for each movie.
    """
    mean_ratings: dict[str, float] = {}

    # TODO: complete function mean_popularity_rating

    num_students = ratings.shape[0]
    num_movies = ratings.shape[1]

    for j in range(num_movies):
        watched = 0
        recommended = 0

        for t in range(num_students):
            r = ratings[t][j]
            if r != -1:
                watched += 1
                if r == 1:
                    recommended += 1

        if watched == 0:
            mean_ratings[movie_idx_to_name[j]] = 0.0
        else:
            mean_ratings[movie_idx_to_name[j]] = recommended / watched

    return mean_ratings

def em(
    Z_init: NDArray,
    R_init: NDArray,
    ratings: NDArray,
    iterations: int,
    e_step: Callable,
    evaluate: Callable,
    m_step: Callable,
):
    """
    Engine for running the EM algorithm code.

    - Z_init, R_init, and ratings are from the load_data functions.
    - iterations specifies how many iterations to run for.
    - e_step, evaluate, and m_step are your function definitions later.

    Returns: p_rz, p_z, the estimated CPTs after running EM for a number of
    iterations.
    - p_rz is a numpy array in the same shape as init_R, with the same
        quantities but is repeatedly updated in EM. The same is true with that
        of p_z and init_Z.
    Also returns the log likelihood evaluations.
    """

    # TODO: fill in the ellipses (...).
    p_rz = (
        R_init.copy()
    )
    p_z = Z_init.copy()
    ll_list: list[float] = []
    for it in tqdm(range(iterations )):
        # E-Step
        joints = e_step(
            p_rz, p_z, ratings
        )
        likelihoods = np.sum(joints, axis=0)
        rho = joints / likelihoods[np.newaxis, :]

        ll = evaluate(likelihoods)
        ll_list.append(ll)

        # M-Step
        p_rz, p_z = m_step(p_rz, p_z, rho, ratings)

    return p_rz, p_z, ll_list


def e_step(p_rz: NDArray, p_z: NDArray, ratings: NDArray):
    """
    Calculates P(Z=i) Π P(R_j = r_j^(t) | Z = i) (the numerator of the Written Section :E-Step)
    for one iteration.

    - p_rz and p_z are your current CPT estimates, as specified above
    - ratings is from your data loading function

    Returns the numerator of the P(Z=i|datapoint_t) (i.e.  P(Z=i) Π P(R_j = r_j^(t) | Z = i) aka joints)

    (We know that you can calculate the full probability here which is the true
    value of rho, instead of the joints, but we ask you to follow the procedure here)

    The return value is an array of shape (k, T) which contains
    P(Z=i) Π P(R_j = r_j^(t) | Z = i) at joints[i, t] (shown above)
    """
    T, M = ratings.shape[0], ratings.shape[1]
    k = p_z.shape[0]
    joints = np.ones((k, T), dtype=np.float32)

    # TODO: complete e_step
    for i in range(k):
        for t in range(T):
            prod = 1.0
            for j in range(M):
                r = ratings[t][j]
                if r == -1:
                    continue
                elif r == 1:
                    prod *= p_rz[j][i]
                else:  # r == 0
                    prod *= (1 - p_rz[j][i])
            joints[i][t] = p_z[i] * prod

    return joints


def evaluate(likelihoods: NDArray):
    """
    Calculate the normalized log-likelihood shown above.

    likelihood for each datapoint. Shape = (T,).

    Returns a scalar.
    """

    # TODO: complete the evaluation function
    T = likelihoods.shape[0]
    return np.sum(np.log(likelihoods)) / T

def m_step(p_rz: NDArray, p_z: NDArray, rho: NDArray, ratings: NDArray):
    """
    Makes the updates to the CPTs of the network, preferably not inplace.

    p_rz, p_z are previous CPTs
    rho is from the E step after normalizing (i.e. P(Z=i | datapoint_t) for all i,t)
    ratings is from your data loading function

    Returns new p_rz, p_z in the same format.
    """
    # TODO: complete m_step

    T, M = ratings.shape
    k = p_z.shape[0]

    new_p_z = np.zeros_like(p_z)
    new_p_rz = np.zeros_like(p_rz)

    # Update P(Z=i)
    for i in range(k):
        total = 0.0
        for t in range(T):
            total += rho[i][t]
        new_p_z[i] = total / T

    # Update P(R_j = 1 | Z=i)
    for i in range(k):
        denom = 0.0
        for t in range(T):
            denom += rho[i][t]

        for j in range(M):
            num = 0.0
            for t in range(T):
                r = ratings[t][j]
                if r == -1:
                    # missing rating
                    num += rho[i][t] * p_rz[j][i]
                elif r == 1:
                    num += rho[i][t] * 1
            new_p_rz[j][i] = num / denom

    return new_p_rz, new_p_z


def inference(
    new_ratings: NDArray, p_z: NDArray, p_rz: NDArray, movie_idx_to_name: list[str]
) -> dict[str, float]:
    """
    - new_ratings: np array of shape (M,) where each entry is 0 for not
    recommended, 1 for recommended, and -1 for haven't seen.
    - p_z, p_rz: as defined above
    - movie_idx_to_name: from data loading step

    Calculate expected_ratings and return a dictionary.
    The key should be the movie name (only those not yet watched) and the value should be its expected rating.
    """
    expected_ratings = {}

    # TODO: calculate expected ratings
    # Hint: can you reuse one of the functions from above to simplify your code?
    M = new_ratings.shape[0]
    single_user_ratings = new_ratings.reshape(1, M)
    # reuse func e-step to get joints
    joints = e_step(p_rz, p_z, single_user_ratings)

    rho = joints[:, 0] / np.sum(joints[:, 0])  # shape (k,)

    # Compute expected rating for unseen movies
    for j in range(M):
        if new_ratings[j] == -1:
            exp_rating = 0.0
            for i in range(len(p_z)):
                exp_rating += rho[i] * p_rz[j][i]
            expected_ratings[movie_idx_to_name[j]] = exp_rating

    return expected_ratings


if __name__ == "__main__":
    ratings, movie_idx_to_name, Z_init, R_init = load_data()
    mean_ratings = mean_popularity_rating(ratings, movie_idx_to_name)

    # Run your EM algorithm
    p_rz, p_z, ll_list = em(Z_init, R_init, ratings, 257, e_step, evaluate, m_step)
    for idx in (2**i for i in range(9)):
        print(f"Iteration {idx} has log-likelihood {ll_list[idx]}")

    # Refer to the ratings you provided by looking up your pid
    # Alternatively generate random ratings
    random.seed(0)
    new_ratings = np.array([random.choice([-1, 0, 1]) for _ in range(60)])
    # my_ratings = ratings[15].copy()
    predictions = inference(new_ratings, p_z, p_rz, movie_idx_to_name)

    # Show some recommendations
    sorted_recs = sorted(list(predictions.items()), reverse=True, key=lambda x: x[1])
    print("\n".join((f"{movie}: {score}" for movie, score in sorted_recs[:5])))
