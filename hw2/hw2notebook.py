import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tabulate import tabulate
from typing import Tuple

def check_evidence(evidence: Tuple[str, str], w: str) -> bool:
    """
    TODO: This function checks if the word is possible given the evidence. This function should
    return True if its possible and False otherwise.

    For example, suppose you've guessed the letters {a,b,c} and the predicted word state
    after those guesses is: _ _ A _ _. Then words like GRAPE should return true while
    words like DIGIT or CRATE should return false.

    Parameters
    ----------
    evidence : tuple
        A tuple containing two strings.
    w : str
        Word to be checked.

    Returns
    -------
    bool
        True if its possible, False otherwise.
    """
    # raise NotImplementedError("check_evidence function is not implemented.")
    correct,incorrect = evidence
    
    for i in range(len(correct)):
        if correct[i] != '-' and correct[i] != w[i]:
            return False
        if w[i] in incorrect:
            return False
    
    # Get correct letters
    guessed_letters = set(c for c in correct if c != '-')
    
    # Check whether letter only appears in the positions marked in correct
    # exp.: AREAS --> Illegal here given "A---S"
    """
    # method 1: the guessed letter in word 
    # shouldn't be in the unguessed position
    unguessed_letter_position = []
    for i in range(len(correct)):
        if correct[i] == "-":
            unguessed_letter_position.append(i)
    for unguessed_pos in unguessed_letter_position:
        if w[unguessed_pos] in guessed_letters:
            return False
    """
    # method 2: make sure the position of the guessed letter in the word
    # is the same as the correctly guessed letter
    for guessed_letter in guessed_letters:
        for i in range(len(w)):
            if w[i] == guessed_letter and correct[i] != guessed_letter:
                return False

    return True

def compute_prior(word_counts: pd.DataFrame) -> pd.Series:
    """
    TODO: This funciton computes the prior probabilities for all words in the corpus.

    Parameters
    ----------
    word_counts : pd.DataFrame
        DataFrame containing words and their counts.

    Returns
    -------
    pd.Series
        Prior probabilities for all words in the corpus.
    """
    # raise NotImplementedError("compute_prior function is not implemented.")
    return word_counts['Count']/word_counts['Count'].sum()

def get_prior(word,word_counts):
    """
    TODO: Gets the prior probability for a given word from the dataframe.

    Parameters
    ----------
    word : String
        The word to get prior probability for

    word_counts: pd.DataFrame
        DataFrame containing words and their counts

    Returns
    -------
    Float
        Prior probability for a given word..
    """
    # raise NotImplementedError('get_prior function is not implemented.')
    return word_counts[word_counts['Word']==word]['Prior'].values[0]

def compute_posterior_denominator(evidence: Tuple[str, str], word_counts: pd.DataFrame) -> float:
    """
    TODO: This function computes the denominator of the posterior probability.

    Note: this is done separately to speed up computation since this is the same regardless
    of the word.

    Parameters
    ----------
    evidence : tuple
        A tuple containing two strings.
    word_counts : pd.DataFrame
        DataFrame containing words, counts, and prior probabilities.

    Returns
    -------
    float
        Probability of evidence.
    """
    # raise NotImplementedError("compute_posterior_denominator function is not implemented.")
    denominator = 0
    for word in word_counts['Word']:
        if check_evidence(evidence,word):
            denominator += get_prior(word,word_counts)
    return denominator

def compute_posterior(evidence: Tuple[str, str], word: str, word_counts: pd.DataFrame, denominator: float) -> float:
    """
    TODO: This function computes the posterior probability used for determining the most likely
    character to predict from the unguessed pool.

    Note: you should use compute_posterior_denominator to help compute this posterior probability.

    Parameters
    ----------
    evidence : tuple
        A tuple containing two strings.
    word : str
        A given word to compute posterior for.
    word_counts : pd.DataFrame
        DataFrame containing words, counts, and prior probabilities.
    denominator : float
        Denominator of the posterior probability computed earlier.

    Returns
    -------
    float
        Posterior probability.
    """
    # return NotImplementedError("compute_posterior function is not implemented.")
    numerator = get_prior(word,word_counts)
    if not check_evidence(evidence,word):
        numerator = 0

    return numerator/denominator

def predictive_probability(evidence: Tuple[str, str], word_counts: pd.DataFrame, denominator: float) -> list:
    """
    TODO: Computes the probability for each letter being in the word given the evidence.
    This function should return a list of probabilities for each letter.

    Parameters
    ----------
    evidence : tuple
        A tuple containing two strings.
    word_counts : pd.DataFrame
        DataFrame containing words, counts, and prior probabilities.
    denominator : float
        Denominator of posterior probability.

    Returns
    -------
    list
        A list of probabilities for each letter.
    """
    # raise NotImplementedError("predictive_probability function is not implemented.")
    correct, incorrect = evidence

    # Collect unguessed letters (exclude already guessed correct and incorrect letters)
    unguessed_letters = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         if letter not in correct and letter not in incorrect]

    # Identify positions in the word that are still unknown
    unknown_positions = [i for i, c in enumerate(correct) if c == '-']

    # Precompute posterior probabilities for candidate words
    posterior = {}
    for word in word_counts["Word"]:
        if check_evidence(evidence, word):
            posterior[word] = compute_posterior(evidence, word, word_counts, denominator)

    # Compute predictive probability for each unguessed letter
    letter_probs = []
    for letter in unguessed_letters:
        prob = 0
        for word, post in posterior.items():
            if any(word[i] == letter for i in unknown_positions):
                prob += post
        letter_probs.append(prob)

    return letter_probs

def predict_character(evidence: Tuple[str, str], word_counts: pd.DataFrame, denominator: float) -> Tuple[str, float]:
    """
    TODO: Generates the prediction for the next best guess with the associated probability.
    This function should return the predicted chraracter and the associated predctive probability.

    Parameters
    ----------
    evidence : tuple
        A tuple containing two strings.
    word_counts : pd.DataFrame
        DataFrame containing words, counts, and prior probabilities.
    denominator : float
        Denominator of the posterior probability.

    Returns
    -------
    tuple
        Predicted character and associated probability.
    """
    # raise NotImplementedError("predict_character function is not implemented.")
    correct, incorrect = evidence

    # Collect unguessed letters
    unguessed_letters = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         if letter not in correct and letter not in incorrect]

    # Get probability list for unguessed letters
    predicted_probabilities = predictive_probability(evidence, word_counts, denominator)
    """
    # Build dictionary {letter: probability}
    letter_probs = dict(zip(unguessed_letters, predicted_probabilities))

    # Pick the best one
    best_letter, best_prob = max(letter_probs.items(), key=lambda x: x[1])
    """
    best_letter = unguessed_letters[np.argmax(predicted_probabilities)]
    best_prob = max(predicted_probabilities)
    return best_letter, best_prob

def run_hangman(word_counts: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Helper function to run hangman on the given test examples.

    Note: feel free to modify the test cases to more thoroughly
    test your solution.

    Parameters
    ----------
    word_counts : pd.DataFrame
        pandas DataFrame containing corpus words, frequency, and prior probabilities.
    verbose : bool, optional
        Verbose set to True will print out the final dataframe table
        in a clean format. Defaults to True.

    Returns
    -------
    pd.DataFrame
        The final dataframe table.
    """
    empty_word = "-----"
    Evidence = [(empty_word, ""),
               (empty_word, "EA"),
                ("A---S", ""),
                ("A---S","I"),
                ("--O--","AEMNT"),
                (empty_word,"EO"),
                ("D--I-",""),
                ("D--I-","A"),
                ("-U---","AEIOS"),
                ("AREA-", "OP"),
                ("SPOR-", "QM"),
                ("AR-AS", "OPE"),
                ("AR-AS", "OPI"),
                ("-RAVE", "B"),
                ("-RAVE", "G")
               ]

    ### DO NOT MODIFY ###
    output = []
    pbar = tqdm(Evidence)
    for e in (Evidence):
        corr, incorr = e
        incorr = "{" + incorr + "}"
        pbar.set_description(f"Processing Evidence: '{e}'")
        char,prob = predict_character(e,word_counts,compute_posterior_denominator(e,word_counts))
        output += [(corr,incorr,char,prob)]
        pbar.update(1)

    output = pd.DataFrame(output,columns = ["Correctly Guessed", "Incorrectly Guessed","Character", "Probability"])
    if verbose:
        print((tabulate(output, headers='keys', tablefmt='psql')))
    return output

def process_data() -> pd.DataFrame:
    """
    This function processes the required files and return word_counts dataframe
    with three columns: Word, Count, Prior.

    Note: your input files should be in the same working directory as this file
    and of the same name.

    Returns
    -------
    pd.DataFrame
        The word_counts data table with columns ('Word', 'Count', 'Prior'), where each row
        represents a different word, its count, and the (computer) prior probability P(W=w)
        of that word.
    """
    ### DO NOT MODIFY ###
    cwd = os.getcwd()
    word_counts_path = os.path.join(cwd, 'hw1_word_counts.txt')

    assert os.path.exists(word_counts_path), f"File not found: {word_counts_path}"

    word_counts = pd.read_csv(word_counts_path,header=None,sep = ' ')
    word_counts = word_counts.rename(columns={0:'Word',1:'Count'})

    word_counts['Prior'] = compute_prior(word_counts)
    return word_counts

def run() -> Tuple[set, set, pd.DataFrame]:
    """
    TODO: This function should run all of hangman and return the required deliverables for parts A and B.

    Note: This function should return three different objects: 2 sets containing the 15 most and 14 least frequent
    5-letter words in the corpus (Part A) and a pandas DataFrame containing the next best guess l and the associated
    predictive probability for each test case.

    DO NOT MODIFY the return types of this function, as this is what the autograder will use to grade your homework!

    Returns
    -------
    Tuple[set, set, pd.DataFrame]
        The 15 most frequent 5-letter words (as a set), the 14 least frequent 5-letter words (as a set), and
        a pandas DataFrame containing the next best guess l and the associated predictive probability for
        each test case.
    """
    word_counts = process_data()

    # Part A: Get 15 most frequent and 14 least frequent 5-letter words
    most_frequent_15 = word_counts.nlargest(15, 'Count')['Word'].tolist()
    least_frequent_14 = word_counts.nsmallest(14, 'Count')['Word'].tolist()
    
    # Part B: Run hangman and get the output
    output = run_hangman(word_counts, verbose=True)
    print(set(most_frequent_15), set(least_frequent_14))
    return set(most_frequent_15), set(least_frequent_14), output

if __name__ == "__main__":
    """
    # debug: if there is any other letter other than alphabetic characters 
    words = process_data()["Word"]
    chars = [[letter for letter in word] for word in words]
    lst = []
    for i in chars:
        lst += i
    set_lst = set(lst)
    for _ in set_lst:
        if _ not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            print("not a letter")
    print("all good")
    """
    run()