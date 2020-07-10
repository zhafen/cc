from collections import Counter
import nltk
from nltk.metrics import edit_distance
import numpy as np

import augment
import verdict

########################################################################

def uniquify_concepts( a, max_edit_distance=2 ):
    '''Unique key concepts, as simplified using nltk tools.
    Steps to retrieve unique key concepts:
    1. Union of the same stems.
    2. Concepts with a sufficiently low edit distance
       (accounts for mispellings)

    Args:
        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two concepts for them
            to count as the same concept.
    '''

    # First pass through with a stemmer
    s = nltk.stem.SnowballStemmer( 'english' )
    ucs = []
    for c in a:
        words = nltk.word_tokenize( c )
        stemmed_words = [ s.stem( w ) for w in words ]
        ucs.append( ' '.join( stemmed_words ) )
    ucs = set( ucs )

    # Look for concepts with a sufficiently low Levenshtein edit-distance
    ucs_arr = np.array( list( ucs ) )
    ucs_ed = []
    for c in ucs_arr:
        # Find matches
        edit_distances = np.array([
            edit_distance( c, c_check, )
            for c_check
            in ucs_arr
        ])
        matches = ucs_arr[edit_distances<=max_edit_distance]
        # Count the matches and represent by maximum count
        count = Counter( matches )
        true_c = verdict.Dict( count ).keymax()[0]
        
        # Store
        ucs_ed.append( true_c )
    ucs = set( ucs_ed )

    return ucs
