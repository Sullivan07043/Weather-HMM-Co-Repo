from typing import Any
import matplotlib.pyplot as plt
from numpy import ndarray, dtype, float64
from numpy.typing import NDArray
import numpy as np


def load_data_viterbi(
    emission_file: str = "emissionMatrix.txt",
    initial_file: str = "initialStateDistribution.txt",
    obs_file: str = "observations.txt",
    transition_file: str = "transitionMatrix.txt",
) -> tuple[ndarray[tuple[Any, ...], dtype[float64]], ndarray[tuple[Any, ...], dtype[float64]], ndarray[
    tuple[Any, ...], dtype[float64]], ndarray[tuple[Any, ...], dtype[float64]]]:
    """
    Loads the necessary data for the Hidden Markov Model (HMM) Viterbi algorithm
    from specified text files.

    Args:
        emission_file (str, optional): Filepath for the Emission Matrix (B).
            Defaults to "emissionMatrix.txt".
        initial_file (str, optional): Filepath for the Initial State Distribution (Pi).
            Defaults to "initialStateDistribution.txt".
        obs_file (str, optional): Filepath for the Observation Sequence.
            Defaults to "observations.txt".
        transition_file (str, optional): Filepath for the Transition Matrix (A).
            Defaults to "transitionMatrix.txt".

    Returns:
        tuple[NDArray]: A tuple containing four NumPy arrays:
            (emissions, initials, observations, transitions)
            - emissions - shape (S , M): Emission matrix (S states, M possible observations).
            - initials - shape (S,): Initial state distribution vector.
            - observations - shape (T,): Sequence of observations (T time steps).
            - transitions - shape (S , S): Transition matrix.
    """
    emissions = np.loadtxt(emission_file, dtype=float)
    initials = np.loadtxt(initial_file, dtype=float)
    transitions = np.loadtxt(transition_file, dtype=float)
    observations = np.loadtxt(obs_file, dtype=int).ravel()

    emissions = np.array(emissions, dtype=float)
    transitions = np.array(transitions, dtype=float)
    initials = np.ravel(initials).astype(float)

    return emissions, initials, observations, transitions

def viterbi(
    emissions: NDArray, initials: NDArray, observations: NDArray, transitions: NDArray
) -> NDArray:
    """
    Implements the Viterbi algorithm to find the most likely sequence of hidden
    states given a sequence of observations and the HMM parameters.

    Use log-probabilities to avoid underflow issues with small probability values.

    Args:
        emissions (NDArray): The Emission Matrix (B) of shape (S, M).
        initials (NDArray): The Initial State Distribution (Pi) of shape (S,).
        observations (NDArray): The sequence of observations of shape (T,).
        transitions (NDArray): The Transition Matrix (A) of shape (S, S).

    Returns:
        NDArray: The optimal path (most likely sequence of hidden states)
            of shape (T,) as an array of integer indices.
    """
    T = observations.shape[0]
    S = transitions.shape[0]

    logA = np.log(transitions)
    logB = np.log(emissions)
    logPi = np.log(initials)

    l = np.zeros((S, T))

    backptr = np.zeros((S, T), dtype=int)

    for s in range(S):
        l[s, 0] = logPi[s] + logB[s, observations[0]]

    for t in range(1, T):
        for s in range(S):
            scores = l[:, t - 1] + logA[:, s]
            best_previous = np.argmax(scores)
            l[s, t] = scores[best_previous] + logB[s, observations[t]]
            backptr[s, t] = best_previous

    state = np.zeros(T, dtype=int)
    state[T - 1] = np.argmax(l[:, T - 1])

    for t in range(T - 2, -1, -1):
        state[t] = backptr[state[t + 1], t + 1]

    return state

def decoded_answer() -> str:
    """
    Returns a specific hardcoded string - which is the decoded answer you found.

    Returns:
        str: The hardcoded decoded message.
    """

    def num_to_char(n):
        if 0 <= n <= 25:
            return chr(ord('a') + n)
        elif n == 26:
            return ' '
        else:
            raise ValueError(f"Invalid symbol index: {n}")

    emissions, initials, observations, transitions = load_data_viterbi()

    path = viterbi(emissions, initials, observations, transitions)

    collapsed = [path[0]]
    for s in path[1:]:
        if s != collapsed[-1]:
            collapsed.append(s)

    message_lst = [num_to_char(s) for s in collapsed]
    message = "".join(message_lst)

    return message

def plot_viterbi_path(path: NDArray) -> None:

    plt.figure(figsize=(12, 4))
    plt.plot(path, linewidth=0.5)
    plt.xlabel("Timestep")
    plt.ylabel("Hidden State Index (0–26)")
    plt.title("Viterbi Path over Time")
    plt.tight_layout()

    plt.savefig("viterbi_path.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    print(decoded_answer())


    # emissions, initials, observations, transitions = load_data_viterbi()
    # path = viterbi(emissions, initials, observations, transitions)
    # plot_viterbi_path(path)
