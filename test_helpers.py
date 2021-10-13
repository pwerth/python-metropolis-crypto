import numpy as np
from decimal import Decimal, getcontext
getcontext().prec = 64

from helpers import get_correctness, permute_string, get_random_string, compute_frequency_string,\
    get_letter_counts, transpose_letters, update_decryption_map
from compute_alpha import compute_dictionary_multiplier


def test_get_correctness():
    assert get_correctness('abc', 'bca') == (0, 0, [])
    assert get_correctness('abc', 'acb') == (1, 1, ['a'])
    assert get_correctness('abc', 'abd') == (2, 2, ['a', 'b'])
    assert get_correctness('abc', 'abc') == (3, 3, ['a', 'b', 'c'])
    assert get_correctness('aabbcc', 'aabbcc') == (3, 6, ['a', 'b', 'c'])


def test_permute_string():
    perm = {
        'a': 'b',
        'b': 'd',
        'c': 'c',
        'd': 'a'
    }
    assert permute_string('aaaa', perm) == 'bbbb'
    assert permute_string('abcd', perm) == 'bdca'
    assert permute_string('ccc', perm) == 'ccc'
    assert permute_string('abcdbca', perm) == 'bdcadcb'


def test_get_random_string():
    text = 'this is a random fake string of nonsense'
    for i in range(1, 10):
        rand_str = get_random_string(i, text)
        assert len(rand_str) == i
        assert get_random_string(i, text) in text


def test_compute_frequency_string():
    s = 'abbbabba'
    assert compute_frequency_string(s) == 'ba'

    s = '   iiii lli lleetttr zzzzzzz   '
    assert compute_frequency_string(s) == ' zilter'


def test_get_letter_counts():
    s = 'aaabbccd'
    assert get_letter_counts(s) == {'a': 3, 'b': 2, 'c': 2, 'd': 1}

    s = 'eee   eee'
    assert get_letter_counts(s) == {'e': 6, ' ': 3}


def test_transpose_letters():
    assert transpose_letters('string', 's', 't') == 'tsring'
    assert transpose_letters('abcabcabc', 'a', 'c') == 'cbacbacba'
    assert transpose_letters('string', 'e', 'f') == 'string'


def test_update_decryption_map():
    decryption_map = {
        'a': 'b',
        'b': 'c',
        'c': 'a'
    }

    # Test that nothing happens if provided symbols aren't in map
    update_decryption_map(decryption_map, 'd', 'e')
    assert decryption_map == {
        'a': 'b',
        'b': 'c',
        'c': 'a'
    }

    # Test that map is updated for symbols that are in map
    update_decryption_map(decryption_map, 'a', 'b')
    assert decryption_map == {
        'a': 'a',
        'b': 'c',
        'c': 'b'
    }

# TODO - use np close here
def test_compute_dictionary_multiplier():
    s = 'it is an be the and'  # 0 invalid two or three letter words
    # Swapping 'a' and 'x' gives 'it is xn be the xnd' which has 1 invalid two letter word, and
    # one invalid three letter word
    l1 = 'a'
    l2 = 'x'
    multiplier = compute_dictionary_multiplier(s, l1, l2, 0.05)
    assert multiplier == Decimal(0.95 ** 2)

    # Now test the reverse situation
    s = 'it is xn be the xnd'
    multiplier = compute_dictionary_multiplier(s, l1, l2, 0.05)
    assert multiplier == Decimal(1.05 ** 2)
