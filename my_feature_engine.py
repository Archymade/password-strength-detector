''' Generate features for password. '''

import string
import re


def count_characters(s, upper=False, alpha=True):
    """ Count the characters in the string:
        upper alphabets, lower alphabets, or numeric characters. """

    if alpha:
        if upper:
            num = len([l for l in s if l in string.ascii_uppercase])

        else:
            num = len([l for l in s if l in string.ascii_lowercase])

    else:
        num = len([n for n in s if n.isnumeric()])

    return num


#####################################################################################################################################

def get_consecutive_numbers(x):
    ''' Count character repetitions. '''

    ix = -1
    num_consec = 0

    for s in x:
        ### Account for possible multiple character occurences
        if x.count(s) > 1:
            ix += 1

        else:
            ix = x.index(s)

        ### Watch out for end of string
        if ix == len(x) - 1:
            break

        else:
            ### Finally, count
            if s.isnumeric() & x[ix + 1].isnumeric():
                num_consec += 1

    return num_consec


#####################################################################################################################################


def generate_patterns(p):
    """ Generate regular expression patterns. """

    p1 = p
    p2 = p[: : -1].replace('(', ')').replace(')', '(')
    p2 = '.*' + p2[2:-3] + ')' + '.*'

    return p1, p2


#####################################################################################################################################

def match_re(s: str, p1: str, p2: str) -> bool:
    ''' Match patterns p1 and p2 to string, s. Return Bool value. '''
    p1, p2 = re.compile(p1), re.compile(p2)

    return True if (re.match(p1, s.lower()) or re.match(p2, s.lower())) else False


#####################################################################################################################################

def count_punct(s):
    """ Count punctuation occurrences. """

    num_punct = 0

    for l in s:
        if l in string.punctuation:
            num_punct += 1

    return num_punct


#####################################################################################################################################

def count_whitespaces(s: str) -> int:
    """ Count whitespace occurences. """

    num_space = 0

    for l in s:
        if l in string.whitespace:
            num_space += 1

    return num_space


#####################################################################################################################################

def get_consecutive_punct(x: str) -> int:
    ''' Count consecutive punctuation pairs. '''

    ix = -1
    num_consec = 0

    for s in x:
        if x.count(s) > 1:
            ix += 1

        else:
            ix = x.index(s)

        if ix == len(x) - 1:
            break

        else:
            if (s in string.punctuation) & (x[ix + 1] in string.punctuation):
                num_consec += 1

    return num_consec


#####################################################################################################################################

def get_consecutive_whitespaces(x: str) -> int:
    ''' Count consecutive punctuation pairs. '''

    ix = -1
    num_consec = 0

    for s in x:
        if x.count(s) > 1:
            ix += 1
        else:
            ix = x.index(s)

        if ix == len(x) - 1:
            break
        else:
            if (s in string.whitespace) & (x[ix + 1] in string.whitespace):
                num_consec += 1

    return num_consec

#####################################################################################################################################
