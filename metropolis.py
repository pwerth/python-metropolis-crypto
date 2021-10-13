from termcolor import colored
from collections import OrderedDict
import json
import math
import numpy as np
import pandas as pd
import random
import time
import sanitizer
import WarAndPeaceNgramCounts
from helpers import get_random_string, get_correctness, permute_string, \
    compute_character_transition_matrix, get_letter_counts, update_decryption_map
from compute_alpha import compute_digram_score, compute_alpha, compute_dictionary_multiplier
from decimal import Decimal, getcontext, InvalidOperation
getcontext().prec = 64

np.set_printoptions(linewidth=1000, precision=2)

English_log_digram_matrix = WarAndPeaceNgramCounts.log_digram_matrix

war_and_peace = open('./sources/WarAndPeace.txt').read()
sanitized_text = sanitizer.sanitize(war_and_peace)

# Length of random string to sample
STR_LENGTHS = [500]
# Number of times to perform the experiment
EXPERIMENT_LIMIT = 100
# Cutoff number of steps when experiment ends
STEP_LIMITS = [2000, 10000]
# Set to True to enable step by step logging
VERBOSE = False
# Set to True to enable dictionary penalty during scoring
USE_DICTIONARY_MULTIPLIER = True
# Use this to override default value of 0.05
DICTIONARY_MULTIPLIER = 0.05
# Number of steps after which dictionary penalty kicks in
DICTIONARY_MULTIPLIER_STEP_THRESHOLD = {
    2000: 1000,
    10000: 5000,
    30000: 10000
}

# Scoring function to use (e.g. digram, trigram)
score = compute_digram_score

# The English alphabet, in alphabetical order
alphabet = list(''.join('abcdefghijklmnopqrstuvwxyz '))
# ndarray indices are ints, this is needed to convert letter to index based on alphabetized order
letter_to_index = {letter: alphabet.index(letter) for letter in alphabet}

# Letters in order of frequency
sorted_war_and_peace_letters = sorted(WarAndPeaceNgramCounts.letter_counts,
                                      key=WarAndPeaceNgramCounts.letter_counts.get,
                                      reverse=True)

# List of averaged experiment results for each parameter set
all_experiment_results = []

for L in STR_LENGTHS:
    for S in STEP_LIMITS:
        print(f'About to perform {EXPERIMENT_LIMIT} experiments with L = {L}, S = {S}')
        experiment_count = 0
        experiment_results = []
        start_time = time.time()

        # Run experiments for these parameters
        while experiment_count < EXPERIMENT_LIMIT:

            # Reset experiment results
            initial_guess_correct_letters = 0
            ciphertext_correct_letters = 0

            if experiment_count > 0:
                print(f'Completed {experiment_count}/{EXPERIMENT_LIMIT} experiments in {round(time.time() - start_time)}s', end='\r')

            # Set plaintext to a random text sample of given length
            plaintext = get_random_string(L, sanitized_text)
            unique_plaintext_letters = list(set(plaintext))

            # Encrypt by applying a randomly chosen permutation from S_27
            encryption_permutation = random.sample(alphabet, len(alphabet))
            encryption_map = {alphabet[i]: encryption_permutation[i] for i in range(0, len(alphabet))}

            ciphertext = permute_string(plaintext, encryption_map)
            ciphertext_correct_letters = get_correctness(plaintext, ciphertext)[0]

            # Initialize decryption map to identity
            decryption_map = {letter: letter for letter in alphabet}

            # Compute frequencies of letters in the ciphertext
            ciphertext_letter_counts = get_letter_counts(ciphertext)
            # Letters in order of frequency in ciphertext (descending)
            sorted_ciphertext_letters = sorted(ciphertext_letter_counts, key=ciphertext_letter_counts.get,
                                               reverse=True)

            # Make initial guess based on relative frequencies of individual letters
            for i in range(0, len(unique_plaintext_letters)):
                decryption_map[sorted_ciphertext_letters[i]] = sorted_war_and_peace_letters[i]

            # Record correctness of initial guess
            initial_guess_str = permute_string(ciphertext, decryption_map)
            curr_ciphertext = initial_guess_str
            transition_matrix = compute_character_transition_matrix(curr_ciphertext)
            initial_guess_correct_letters = get_correctness(plaintext, initial_guess_str)[0]

            step_count = 0
            experiment_count += 1
            accepted_transposition_count = 0
            n_accepted_transpositions_before_max = 0
            n_plaintext_was_best = 0
            plaintext_visited = 0  # bool represented as int
            plaintext_score = Decimal(score(plaintext))
            top_ten_scores = [-math.inf for _ in range(10)]
            max_score = -math.inf
            max_ciphertext = permute_string(ciphertext, decryption_map)
            max_step = 0

            # Perform the Metropolis Algorithm
            while step_count < S:

                # If using dictionary method, reset to the best scoring permutation
                if USE_DICTIONARY_MULTIPLIER and step_count == DICTIONARY_MULTIPLIER_STEP_THRESHOLD[S]:
                    curr_ciphertext = max_ciphertext
                    transition_matrix = compute_character_transition_matrix(curr_ciphertext)

                # Generate two uniformly randomly chosen letters to transpose
                l1 = np.random.choice(unique_plaintext_letters, 1)[0]
                l2 = np.random.choice(unique_plaintext_letters, 1)[0]

                # If proposed transposition is the identity, do nothing
                if l1 == l2:
                    continue

                # Note: we only increment step_count if a non-identity transposition is proposed
                step_count += 1

                # Compute alpha
                alpha, proposed_transition_matrix = compute_alpha(transition_matrix, l1, l2)
                r = random.uniform(0, 1)

                if USE_DICTIONARY_MULTIPLIER and step_count >= DICTIONARY_MULTIPLIER_STEP_THRESHOLD[S]:
                    multiplier = compute_dictionary_multiplier(curr_ciphertext, l1, l2, DICTIONARY_MULTIPLIER)
                    alpha = alpha * multiplier

                if r < alpha:
                    accepted_transposition_count += 1
                    update_decryption_map(decryption_map, l1, l2)

                    curr_ciphertext = permute_string(ciphertext, decryption_map)
                    if curr_ciphertext == plaintext:
                        plaintext_visited = 1

                    if VERBOSE:
                        try:
                            print(colored(f'[{step_count}] ({l1} <> {l2})', 'green'), f'{curr_ciphertext}',
                                  f' | alpha = {round(alpha, 2)}')
                        except InvalidOperation as ex:
                            pass

                    current_score = Decimal(score(curr_ciphertext))
                    transition_matrix = proposed_transition_matrix
                    # Record score if in top 10
                    if current_score > top_ten_scores[0]:
                        # Update stats if score is new top score
                        if current_score > max_score:
                            max_score = current_score
                            max_ciphertext = curr_ciphertext
                            max_step = step_count
                            n_accepted_transpositions_before_max = accepted_transposition_count
                        top_ten_scores.append(current_score)
                        top_ten_scores = sorted(top_ten_scores)[-10:]
                elif VERBOSE:
                    try:
                        print(colored(f'[{step_count}] ({l1} <> {l2})', 'red'), f'{curr_ciphertext}',
                              f' | alpha = {round(alpha, 2)}')
                    except InvalidOperation as ex:
                        pass

            n_correct_letters, n_correct_symbols, all_correct_letters = get_correctness(plaintext,
                                                                                        max_ciphertext)

            # Compute relative rank of plaintext score
            plaintext_score_rank = None
            if plaintext_score <= top_ten_scores[0]:
                plaintext_score_rank = '10+'
            elif plaintext_score >= top_ten_scores[-1]:
                plaintext_score_rank = '1'
            else:
                for i in range(9):
                    if top_ten_scores[i] <= plaintext_score < top_ten_scores[i + 1]:
                        plaintext_score_rank = str(10 - i)
            if not plaintext_score_rank:
                print(f'Unable to compute rank', plaintext_score, top_ten_scores)

            # Record the results of the experiment
            experiment_results.append(OrderedDict({
                'n_unique_letters_plaintext': len(unique_plaintext_letters),
                'ciphertext_correct_letters': ciphertext_correct_letters,
                'initial_guess_correct_letters': initial_guess_correct_letters,
                'n_correct_letters': n_correct_letters,
                'n_correct_symbols': n_correct_symbols,
                'plaintext_score': round(plaintext_score, 2),
                'best_score': round(max_score, 2),
                'best_step': max_step,
                'correct_letters': ''.join(all_correct_letters),
                'n_accepted_transpositions': accepted_transposition_count,
                'n_accepted_transpositions_before_max': n_accepted_transpositions_before_max,
                'plaintext_visited': plaintext_visited,
                'plaintext_was_best': max_ciphertext == plaintext,
                'plaintext_rank': plaintext_score_rank
            }))

        total_time = time.time() - start_time
        print('\nAll experiments completed after', round(total_time, 2), 's')
        print(f'Average time of experiment (for L = {L}, S = {S}):', round(total_time / EXPERIMENT_LIMIT, 2), 's')

        # Use Pandas DataFrame for utility functions
        experiment_results_df = pd.DataFrame(experiment_results)
        # Record the experiment results in master table
        all_experiment_results.append(OrderedDict({
            'L': L,
            'Step Limit': S,
            'Unique Letters': experiment_results_df['n_unique_letters_plaintext'].mean(),
            'Decoded Letters': experiment_results_df['n_correct_letters'].mean(),
            'Decoded Symbols': experiment_results_df['n_correct_symbols'].mean(),
            'Plaintext Visited': experiment_results_df['plaintext_visited'].sum(),
            'Plaintext Score': experiment_results_df['plaintext_score'].mean(),
            'Best Visited Score': experiment_results_df['best_score'].mean(),
            'Best Step': experiment_results_df['best_step'].mean(),
            'State Changes': experiment_results_df['n_accepted_transpositions'].mean(),
            'Plaintext Relative Score': json.dumps(experiment_results_df['plaintext_rank'].value_counts().to_dict())
        }))
        # Print CSV of all experiment results (at each loop, to save progress)
        all_experiment_results_df = pd.DataFrame(all_experiment_results)
        all_experiment_results_df.to_csv('experiment_results.csv', index=False)
