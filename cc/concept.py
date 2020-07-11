from collections import Counter
import nltk
from nltk.metrics import edit_distance
import numpy as np

import augment
import verdict

########################################################################

def uniquify_concepts( a, max_edit_distance=2, min_len_ed=5 ):
    '''Unique key concepts, as simplified using nltk tools.
    Steps to retrieve unique key concepts:
    1. Union of the same stems.
    2. Concepts with a sufficiently low edit distance
       (accounts for mispellings)

    Args:
        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two concepts for them
            to count as the same concept.

        min_len_ed (int):
            Words below this length will not be considered the same even if
            they are within the given edit distance.
    '''

    # First pass through with a stemmer
    s = nltk.stem.SnowballStemmer( 'english' )
    ucs = []
    for c in a:
        words = nltk.word_tokenize( c )
        stemmed_words = [ s.stem( w ) for w in words ]
        ucs.append( ' '.join( stemmed_words ) )
    ucs = set( ucs )

    # Find word lengths
    word_lens = np.array([ len( uc ) for uc in ucs ])
    satisfies_min_len_ed = word_lens >= min_len_ed

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

        # Matches if below minimum edit distance and not too short, or
        # if no edits are required
        is_matching = (
            ( edit_distances <= max_edit_distance ) &
            satisfies_min_len_ed
        )
        is_matching = np.logical_or( is_matching, edit_distances == 0 )
        matches = ucs_arr[is_matching]

        # Count the matches and represent by maximum count
        count = Counter( matches )
        true_c = verdict.Dict( count ).keymax()[0]
        
        # Store
        ucs_ed.append( true_c )
    ucs = set( ucs_ed )

    return ucs
