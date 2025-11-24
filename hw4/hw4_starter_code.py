import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def load_data(relative_bigram_path: str, relative_unigram_path: str, relative_vocab_path: str) -> Tuple[dict, dict]:
    """
    TODO: Load and process the bigram, unigram, and vocabulary data from files.

    IMPORTANT: Use relative paths only! Do NOT use absolute paths as this may cause 
    the autograder to fail when running tests.

    This function should:
    1. Read bigram data from relative_bigram_path (format: "word1_index word2_index count" per line)
    2. Read unigram counts from relative_unigram_path (format: one count per line)
    3. Read vocabulary from relative_vocab_path (format: one word per line)
    4. Build a unigram_dict mapping word -> count
    5. Build a bigram_dict mapping word1 -> {word2: count}

    Note: In bigram file, indices are 1-based (not 0-based).
    Note: Only include bigrams where the first word actually has successors.

    Parameters
    ----------
    relative_bigram_path : str
        Relative path to bigram counts file (e.g., 'bigram.txt').
    relative_unigram_path : str
        Relative path to unigram counts file (e.g., 'unigram.txt').
    relative_vocab_path : str
        Relative path to vocabulary file (e.g., 'vocab.txt').

    Returns
    -------
    Tuple[dict, dict]
        A tuple containing (bigram_dict, unigram_dict).
        - bigram_dict: dict[str, dict[str, int]] - nested dictionary of bigram counts
        - unigram_dict: dict[str, int] - dictionary of unigram counts
    """
    dic_bigram_dict = {}
    dic_unigram_dict = {}
    # 1. bigram
    with open(relative_bigram_path, "r") as f:
        for line in f:
            word1_index, word2_index, count = map(int, line.strip().split())
            if word1_index not in dic_bigram_dict:
                dic_bigram_dict[word1_index] = {}
            dic_bigram_dict[word1_index][word2_index] = count

    # 2. unigram
    with open(relative_unigram_path, "r") as f:
        for idx, line in enumerate(f, start=1):  # make unigram index 1-based
            dic_unigram_dict[idx] = int(line.strip())

    # 3. vocab
    with open(relative_vocab_path, "r") as f:
        words = [line.strip() for line in f if line.strip()]

    # 4. build word-based dicts
    unigram_dict = {}
    bigram_dict = {}

    for idx, word in enumerate(words, start=1):  # also make to 1-based
        unigram_dict[word] = dic_unigram_dict[idx]

    for w1_index, inner_dic in dic_bigram_dict.items():
        w1_word = words[w1_index - 1]
        bigram_dict[w1_word] = {}
        for w2_index, count in inner_dic.items():
            w2_word = words[w2_index - 1]
            bigram_dict[w1_word][w2_word] = count

    return bigram_dict, unigram_dict

def compute_unigram_probabilities(unigram_dict: dict[str, int]) -> dict[str, float]:
    """
    TODO: Compute maximum likelihood unigram probabilities P(w) for each word.

    Parameters
    ----------
    unigram_dict : dict[str, int]
        Dictionary mapping words to their counts.

    Returns
    -------
    dict[str, float]
        Dictionary mapping words to their unigram probabilities.
    """
    total_count = sum(unigram_dict.values())
    prob_dict = {}
    for word, count in unigram_dict.items():
        prob_dict[word] = count / total_count
    return prob_dict

def compute_bigram_probabilities(bigram_dict: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """
    TODO: Compute maximum likelihood bigram probabilities P(w'|w) for each word pair.

    Parameters
    ----------
    bigram_dict : dict[str, dict[str, int]]
        Nested dictionary of bigram counts.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary mapping word1 -> {word2: probability}.
    """
    prob_dict = {}
    for word1, successors in bigram_dict.items():
        total_count = sum(successors.values())
        prob_dict[word1] = {}
        for word2, count in successors.items():
            prob_dict[word1][word2] = count / total_count
    return prob_dict

def compute_sentence_log_likelihood(
    sentence: str,
    unigram_probabilities: dict[str, float],
    bigram_probabilities: dict[str, dict[str, float]]
) -> Tuple[float, float]:
    """
    TODO: Compute the log-likelihood of a sentence under both unigram and bigram models.

    Note:
    - Use <s> as the sentence start token for the first word's bigram probability
    - If a word is not in vocabulary, replace it with <UNK>
    - If a bigram is not observed, set bigram log-likelihood to -infinity

    Parameters
    ----------
    sentence : str
        Input sentence (space-separated words).
    unigram_probabilities : dict[str, float]
        Unigram probabilities.
    bigram_probabilities : dict[str, dict[str, float]]
        Bigram probabilities.

    Returns
    -------
    Tuple[float, float]
        A tuple containing (unigram_log_likelihood, bigram_log_likelihood).
    """
    # unigram
    sentence_lst_uni = [w if w in unigram_probabilities else "<UNK>" for w in sentence.split()]
    uni_prob_dic = compute_unigram_probabilities(unigram_probabilities)
    unigram_log_likelihood = 0.0
    for word in sentence_lst_uni:
        if word in unigram_probabilities and unigram_probabilities[word]>0:
            unigram_log_likelihood += math.log(uni_prob_dic[word])

    # bigram
    sentence_lst_bi = ["<s>"] + [w if w in unigram_probabilities else "<UNK>" for w in sentence.split()]

    bigram_log_likelihood = 0.0
    for i in range(1, len(sentence_lst_bi)):
        prev, curr = sentence_lst_bi[i - 1], sentence_lst_bi[i]
        if prev in bigram_probabilities and curr in bigram_probabilities[prev]:
            prob = bigram_probabilities[prev][curr]
            if prob > 0:
                bigram_log_likelihood += math.log(prob)
            else:
                return unigram_log_likelihood, float("-inf")
        else:
            return unigram_log_likelihood, float("-inf")

    return unigram_log_likelihood, bigram_log_likelihood

def compute_mixed_likelihood(
    sentence: str,
    lambda_val: float,
    unigram_probabilities: dict[str, float],
    bigram_probabilities: dict[str, dict[str, float]]
) -> float:
    """
    TODO: Compute the log-likelihood under a mixed/interpolated model.

    Note:
    - Use <s> as the previous token for the first word
    - If a word is not in vocabulary, replace it with <UNK>
    - If a bigram is not observed, use 0 for P_b(w'|w)
    - If the mixed probability is 0, return -infinity

    Parameters
    ----------
    sentence : str
        Input sentence (space-separated words).
    lambda_val : float
        Interpolation weight for unigram model (between 0 and 1).
    unigram_probabilities : dict[str, float]
        Unigram probabilities.
    bigram_probabilities : dict[str, dict[str, float]]
        Bigram probabilities.

    Returns
    -------
    float
        Mixed model log-likelihood.
    """
    # Replace unknown words with <UNK>
    tokens = [w if w in unigram_probabilities else "<UNK>" for w in sentence.split()]
    tokens = ["<s>"] + tokens  # prepend <s> as start token

    log_likelihood = 0.0

    for i in range(1, len(tokens)):
        prev, curr = tokens[i - 1], tokens[i]

        # Unigram probability
        p_uni = unigram_probabilities.get(curr, 0.0)

        # Bigram probability (default 0 if unseen)
        p_bi = bigram_probabilities.get(prev, {}).get(curr, 0.0)

        # Mixed probability
        p_mix = lambda_val * p_uni + (1 - lambda_val) * p_bi

        if p_mix > 0:
            log_likelihood += math.log(p_mix)
        else:
            return float("-inf")

    return log_likelihood

def plot_and_get_optimal_lambda(
    sentence: str,
    unigram_probabilities: dict[str, float],
    bigram_probabilities: dict[str, dict[str, float]]
) -> float:
    """
    TODO: Find the optimal lambda value that maximizes the mixed model likelihood.

    This function should:
    1. Try lambda values from 0.00, 0.01, 0.02, ..., 0.99, 1.00
    2. For each lambda, compute the mixed likelihood
    3. Find the lambda that gives the maximum likelihood
    4. Optionally: plot likelihood vs lambda (not required for autograder)

    Parameters
    ----------
    sentence : str
        Input sentence (space-separated words).
    unigram_probabilities : dict[str, float]
        Unigram probabilities.
    bigram_probabilities : dict[str, dict[str, float]]
        Bigram probabilities.

    Returns
    -------
    float
        Optimal lambda value (between 0 and 1).
    """
    lambda_vals = [i / 100.0 for i in range(101)]
    likelihoods = []

    for lam in lambda_vals:
        ll = compute_mixed_likelihood(sentence, lam, unigram_probabilities, bigram_probabilities)
        likelihoods.append(ll)

    # Find the best lambda
    max_idx = max(range(len(lambda_vals)), key=lambda i: likelihoods[i])
    optimal_lambda = lambda_vals[max_idx]

    # Plot (optional, for visualization)
    plt.figure(figsize=(6, 4))
    plt.plot(lambda_vals, likelihoods, marker='o')
    plt.xlabel("Lambda")
    plt.ylabel("Log-Likelihood")
    plt.title(f"Mixed Model Log-Likelihood vs Lambda\nSentence: \"{sentence}\"")
    plt.grid(True)
    plt.savefig("mixed_model_log_likelihood.png")
    plt.show(block=False)

    return optimal_lambda

if __name__ == "__main__":
    # You can test your functions here
    # Example usage:
    relative_bigram_path = 'bigram.txt'
    relative_unigram_path = 'unigram.txt'
    relative_vocab_path = 'vocab.txt'

    # Load data
    bigram_dict, unigram_dict = load_data(relative_bigram_path, relative_unigram_path, relative_vocab_path)

    # Compute probabilities
    unigram_probs = compute_unigram_probabilities(unigram_dict)
    bigram_probs = compute_bigram_probabilities(bigram_dict)

    # Test on example sentence
    sentence = "THE STOCK MARKET FELL BY ONE HUNDRED POINTS LAST WEEK"
    sentence_2 = "THE SIXTEEN OFFICIALS SOLD FIRE INSURANCE"
    u_ll, b_ll = compute_sentence_log_likelihood(sentence_2, unigram_probs, bigram_probs)
    print(f"Unigram log-likelihood: {u_ll}")
    print(f"Bigram log-likelihood: {b_ll}")

    mixed_ll = compute_mixed_likelihood(sentence_2, 0.1, unigram_probs, bigram_probs)
    print(f"Mixed log-likelihood: {mixed_ll}")
    mixed_model = plot_and_get_optimal_lambda(sentence_2, unigram_probs, bigram_probs)
    print(f"Optimal lambda: {mixed_model}")

