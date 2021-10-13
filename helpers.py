import numpy as np
import random

# The English alphabet, in alphabetical order
ENGLISH_ALPHABET = list(''.join('abcdefghijklmnopqrstuvwxyz '))
# ndarray indices are ints, this is needed to convert letter to index based on alphabetized order
letter_to_index = {letter: ENGLISH_ALPHABET.index(letter) for letter in ENGLISH_ALPHABET}
# String of English symbols (26 letters and space) in decreasing order of frequency
english_freq_string = ' etaonihsrdlumcwfgypbvkxjzq'


def compute_character_transition_matrix(s: str) -> np.ndarray:
    """
    For a given string s, returns an 27x27 matrix whose (i,j)-th entry is the number of
    times the digram i-j appears in s.
    """
    # Initialize character_transition_matrix to all zeros
    T = np.zeros((27, 27))

    for i in range(len(s) - 1):
        T[letter_to_index[s[i]], letter_to_index[s[i + 1]]] += 1

    return T


def compute_frequency_string(s: str) -> str:
    """
    For a given string s containing N unique letters, returns a string of length N whose first
    character is the letter appearing most frequently in s, etc.

    Ties are handled by index in English language frequency string i.e. the letter that occurs more
    often empirically in English text wins
    """
    letter_counts = get_letter_counts(s)
    # letter counts in descending order
    sorted_letter_counts = sorted(letter_counts, key=letter_counts.get, reverse=True)
    # TODO - HANDLE TIES!!!!

    return ''.join(sorted_letter_counts)


def get_correctness(str1: str, str2: str):
    # Count the number of symbols from the alphabet were decoded correctly
    correct_symbol_count = 0
    # Count the number of letters in the plaintext that were decoded correctly
    correct_letter_count = 0
    # The correctly decoded letters
    correct_letter_list = []
    for l in list(set(str1)):
        if str1.find(l) == str2.find(l) and str1.find(l) >= 0:
            correct_symbol_count += 1
            correct_letter_count += str1.count(l)
            correct_letter_list.append(l)

    correct_letter_list.sort()

    return correct_symbol_count, correct_letter_count, correct_letter_list


def get_letter_counts(s: str) -> dict:
    """
    For given string s, returns a dictionary of letter counts appearing in s. The dictionary keys
    are the letters appearing in s, and the values are the counts
    """
    unique_letters = set(s)
    letter_counts = {letter: 0 for letter in unique_letters}
    for char in s:
        letter_counts[char] += 1
    return letter_counts


def permute_string(s: str, permutation: dict):
    """
    Applies the given permutation to the letters in the string s
    """
    new_string = ''

    for i in range(len(s)):
        new_string += permutation[s[i]]

    return new_string


def get_random_string(n: int, s: str):
    """
    Get a random string of length n from s
    """
    string_start_index = random.randint(1, len(s) - n)
    return s[string_start_index: string_start_index + n]


def transpose_letters(string: str, l1: str, l2: str):
    """Swap all occurrences of l1 and l2 in string"""
    ret = ''
    for char in string:
        if char == l1:
            ret += l2
        elif char == l2:
            ret += l1
        else:
            ret += char
    return ret


def update_decryption_map(decryption_map: dict, l1: str, l2: str):
    """Updates given decryption map to handle transposition of symbols l1 and l2"""
    letter_mapping_to_l1 = None
    letter_mapping_to_l2 = None
    for d in decryption_map:
        if decryption_map[d] == l1:
            letter_mapping_to_l1 = d
        elif decryption_map[d] == l2:
            letter_mapping_to_l2 = d
    if letter_mapping_to_l1 and letter_mapping_to_l2:
        decryption_map[letter_mapping_to_l1] = l2
        decryption_map[letter_mapping_to_l2] = l1
