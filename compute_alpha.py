import math
import numpy as np
from WarAndPeaceNgramCounts import log_digram_matrix as English_log_digram_matrix
from WarAndPeaceNgramCounts import log_digram_dict
from WarAndPeaceNgramCounts import digram_freqs as P2
from WarAndPeaceNgramCounts import trigram_freqs as P3
from WarAndPeaceDictionary import two_letter_words as VALID_TWO_LETTER_WORDS
from WarAndPeaceDictionary import three_letter_words as VALID_THREE_LETTER_WORDS
from decimal import *
from decimal import Decimal
from helpers import transpose_letters

getcontext().prec = 64

# The English alphabet, in alphabetical order
alphabet = list('abcdefghijklmnopqrstuvwxyz ')
# ndarray indices are ints, this is needed to convert letter to index based on alphabetized order
letter_to_index = {letter: alphabet.index(letter) for letter in alphabet}


def compute_digram_score(string: str):
    """Compute the digram score of given string. We compute the score as a sum of logs, rather than
    a log of a product, in order to avoid overflow/precision issues."""
    score = Decimal(0)
    str_letters = list(string)
    for i in range(0, len(str_letters) - 1):
        digram_score = Decimal(0)
        digram = f'{str_letters[i]}{str_letters[i + 1]}'
        if P2[digram] != 0:
            digram_score = Decimal(math.log(P2[digram]))
        score = Decimal(score + digram_score)
    return score


def compute_trigram_score(string: str):
    score = Decimal(1)
    str_letters = list(string)
    for i in range(0, len(str_letters) - 2):
        trigram_score = Decimal(1)
        trigram = f'{str_letters[i]}{str_letters[i + 1]}{str_letters[i + 2]}'
        if P3[trigram] != 0:
            trigram_score = Decimal(P3[trigram])
        score = Decimal(score * trigram_score)
    return Decimal(score)


def get_digram_counts(string: str):
    """Returns a dictionary of digram counts for all English digrams, for given string"""
    digram_counts = {}
    for a in alphabet:
        for b in alphabet:
            digram_counts[f'{a}{b}'] = 0
    for i in range(0, len(string) - 1):
        digram_counts[f'{string[i]}{string[i + 1]}'] += 1

    # Check that we counted all digrams
    assert sum(digram_counts[digram] for digram in digram_counts) == (len(string) - 1)
    return digram_counts


def compute_character_transition_matrix(s: str):
    """
    For a given string s, returns an 27x27 matrix whose (i,j)-th entry is the number of
    times the digram i-j appears in s.
    """
    # Initialize character_transition_matrix to all zeros
    T = np.zeros((27, 27))

    for i in range(len(s) - 1):
        T[letter_to_index[s[i]], letter_to_index[s[i + 1]]] += 1

    return T


def compute_alpha_1(txt: str, l1: str, l2: str):
    """Computes alpha using equation (72)
    :param txt: str
    :param l1: str, letter to transpose
    :param l2: str, other letter to transpose
    :return float
    """
    curr_score = compute_digram_score(txt)
    proposed_txt = transpose_letters(txt, l1, l2)
    new_score = compute_digram_score(proposed_txt)
    return Decimal(new_score / curr_score)


def compute_alpha_2(txt: str, l1: str, l2: str):
    """Computes alpha using equation (73)
    :param txt: str
    :param l1: str, letter to transpose
    :param l2: str, other letter to transpose
    :return float
    """
    proposed_txt = transpose_letters(txt, l1, l2)
    digram_counts_in_txt = get_digram_counts(txt)
    digram_counts_in_proposed_txt = get_digram_counts(proposed_txt)

    # Compute (73)
    num = 1
    denom = 1
    for a in alphabet:
        for b in alphabet:
            if P2[f'{a}{b}'] > 0:
                num = num * pow(P2[f'{a}{b}'], digram_counts_in_proposed_txt[f'{a}{b}'])
                denom = denom * pow(P2[f'{a}{b}'], digram_counts_in_txt[f'{a}{b}'])

    return Decimal(num / denom)


def compute_alpha_3(txt: str, l1: str, l2: str):
    """Computes alpha using equation (74)
      :param txt: str
      :param l1: str, letter to transpose
      :param l2: str, other letter to transpose
      :return float
      """
    proposed_txt = transpose_letters(txt, l1, l2)
    digram_counts_in_txt = get_digram_counts(txt)
    digram_counts_in_proposed_txt = get_digram_counts(proposed_txt)

    # Compute (74)
    num = 1
    denom = 1
    for a in alphabet:
        p2xi = P2[f'{a}{l1}']
        p2xj = P2[f'{a}{l2}']
        p2ix = P2[f'{l1}{a}']
        p2jx = P2[f'{l2}{a}']

        # TODO - there is double counting for digrams ij and ji

        if p2xi > 0:
            num = Decimal(num * Decimal(pow(p2xi, digram_counts_in_proposed_txt[f'{a}{l1}'])))
            denom = denom * pow(p2xi, digram_counts_in_txt[f'{a}{l1}'])
        if p2xj > 0:
            num = Decimal(num * Decimal(pow(p2xj, digram_counts_in_proposed_txt[f'{a}{l2}'])))
            denom = denom * pow(p2xj, digram_counts_in_txt[f'{a}{l2}'])
        if p2ix > 0:
            num = Decimal(num * Decimal(pow(p2ix, digram_counts_in_proposed_txt[f'{l1}{a}'])))
            denom = denom * pow(p2ix, digram_counts_in_txt[f'{l1}{a}'])
        if p2jx > 0:
            num = Decimal(num * Decimal(pow(p2jx, digram_counts_in_proposed_txt[f'{l2}{a}'])))
            denom = denom * pow(p2jx, digram_counts_in_txt[f'{l2}{a}'])

    return Decimal(num / denom)


def compute_alpha_4(txt: str, l1: str, l2: str):
    """Computes alpha using equation (75)
        :param txt: str
        :param l1: str, letter to transpose
        :param l2: str, other letter to transpose
        :return float
        """
    proposed_txt = transpose_letters(txt, l1, l2)
    digram_counts_in_txt = get_digram_counts(txt)
    digram_counts_in_proposed_txt = get_digram_counts(proposed_txt)

    # Compute (75)
    log_alpha = Decimal(0)
    for a in alphabet:
        p2xi = P2[f'{a}{l1}']
        p2xj = P2[f'{a}{l2}']
        p2ix = P2[f'{l1}{a}']
        p2jx = P2[f'{l2}{a}']

        i1 = alphabet.index(l1)
        i2 = alphabet.index(l1)
        ia = alphabet.index(a)

        if p2xi > 0:
            # log_p2xi = English_log_digram_matrix[ia, i1]
            log_p2xi = Decimal(math.log(p2xi))
            log_alpha = Decimal(log_alpha + Decimal(((digram_counts_in_proposed_txt[f'{a}{l1}'] -
                                                      digram_counts_in_txt[
                                                          f'{a}{l1}']) * log_p2xi)))
        if p2xj > 0:
            # log_p2xj = English_log_digram_matrix[ia, i2]
            log_p2xj = Decimal(math.log(p2xj))
            log_alpha = Decimal(log_alpha + Decimal(((digram_counts_in_proposed_txt[f'{a}{l2}'] -
                                                      digram_counts_in_txt[
                                                          f'{a}{l2}']) * log_p2xj)))
        if p2ix > 0:
            # log_p2ix = English_log_digram_matrix[i1, ia]
            log_p2ix = Decimal(math.log(p2ix))
            log_alpha = Decimal(log_alpha + Decimal(((digram_counts_in_proposed_txt[f'{l1}{a}'] -
                                                      digram_counts_in_txt[
                                                          f'{l1}{a}']) * log_p2ix)))
        if p2jx > 0:
            # log_p2jx = English_log_digram_matrix[i2, ia]
            log_p2jx = Decimal(math.log(p2jx))
            log_alpha = Decimal(log_alpha + Decimal(((digram_counts_in_proposed_txt[f'{l2}{a}'] -
                                                      digram_counts_in_txt[
                                                          f'{l2}{a}']) * log_p2jx)))

    return Decimal(math.exp(log_alpha))


def compute_alpha_5(txt: str, l1: str, l2: str):
    curr_trans_matrix = compute_character_transition_matrix(txt)
    proposed_txt = transpose_letters(txt, l1, l2)
    proposed_transition_matrix = compute_character_transition_matrix(proposed_txt)
    diff_trans_matr = proposed_transition_matrix - curr_trans_matrix

    log_alpha = Decimal(0)
    for a in alphabet:
        for b in alphabet:
            p2 = P2[f'{a}{b}']
            if p2 > 0:
                log_p2 = Decimal(math.log(p2))
                i1 = letter_to_index[a]
                i2 = letter_to_index[b]
                log_alpha = log_alpha + (Decimal(diff_trans_matr[i1, i2]) * log_p2)

    return Decimal(math.exp(log_alpha))


def compute_alpha(transition_matrix: np.array, l1: str, l2: str) -> (Decimal, np.array):
    proposed_transition_matrix = transition_matrix.copy()
    i1 = letter_to_index[l1]
    i2 = letter_to_index[l2]
    # Swap the rows corresponding to the letters to transpose
    proposed_transition_matrix[[i1, i2]] = proposed_transition_matrix[[i2, i1]]
    # Then swap the columns corresponding to the letters (by swapping rows of the transpose)
    proposed_transition_matrix = proposed_transition_matrix.T
    proposed_transition_matrix[[i1, i2]] = proposed_transition_matrix[[i2, i1]]
    proposed_transition_matrix = proposed_transition_matrix.T

    log_alpha = np.multiply(proposed_transition_matrix - transition_matrix,
                            English_log_digram_matrix).sum()
    try:
        return Decimal(math.exp(log_alpha)), proposed_transition_matrix
    except OverflowError:
        if log_alpha > math.exp(1):
            return Decimal(1), proposed_transition_matrix
        else:
            print('Unhandled OverflowError', log_alpha)


def compute_dictionary_multiplier(s: str, l1: str, l2: str, factor: float = 0.05) -> Decimal:
    """Compute dictionary multiplier to be applied to Metropolis step alpha, corresponding to a
    transposition of l1 and l2 from string s.

    The penalty is computed as
        MULTIPLIER = (1 +/- (factor)) ^ [|n_incorrect_words(curr) - n_incorrect_words(proposed)|]

    Thus if the proposed string has fewer incorrect words, the multiplier will be > 1, which
    increases alpha (and the likelihood of transitioning to the proposed string).

    factor defaults to 0.05 but can be specified as an arg
    """

    def _is_invalid_two_letter_word(w: str) -> bool:
        if len(w) == 2 and w not in VALID_TWO_LETTER_WORDS:
            return True
        return False

    def _is_invalid_three_letter_word(w: str) -> bool:
        if len(w) == 3 and w not in VALID_THREE_LETTER_WORDS:
            return True
        return False

    multiplier = Decimal(1)  # to avoid possibility of MULTIPLIER = 0
    factor = Decimal(factor)
    proposed = transpose_letters(s, l1, l2)
    n_incorrect_words_curr = 0
    n_incorrect_words_proposed = 0
    for word in s.split(' '):
        if _is_invalid_two_letter_word(word) or _is_invalid_three_letter_word(word):
            n_incorrect_words_curr += 1
    for word in proposed.split(' '):
        if _is_invalid_two_letter_word(word) or _is_invalid_three_letter_word(word):
            n_incorrect_words_proposed += 1
    diff = n_incorrect_words_curr - n_incorrect_words_proposed
    if diff == 0:
        return Decimal(1)
    if diff > 0:
        return Decimal((multiplier + factor) ** diff)
    return Decimal((multiplier - factor) ** (-diff))
