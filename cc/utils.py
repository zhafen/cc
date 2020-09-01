from collections import Counter 
import nltk
from nltk.metrics import edit_distance
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import verdict

########################################################################

def uniquify_words( a, **kwargs ):
    '''Find the unique words in a list
    This involves...
    1. Stemming the words in the lists
    2. Matching words with a sufficiently low edit distance
       (accounts for mispellings)

    Args:
        a (list of strs):
            The list to uniquify

    Kwargs:
        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two words for them
            to count as the same word.

        min_len_ed (int):
            Words below this length will not be considered the same even if
            they are within the given edit distance.

        stemmed (bool):
            If True the words are already stemmed.
    '''

    return match_words( a, a, **kwargs )

########################################################################

def match_words( a, b, max_edit_distance=2, min_len_ed=5, stemmed=False ):
    '''Find the matching words in two lists.
    This involves...
    1. Stemming the words in the lists
    2. Matching words with a sufficiently low edit distance
       (accounts for mispellings)

    Args:
        a (list of strs):
            The first to look for matching words in.

        b (list of strs):
            The second to look for matching words in.

        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two concepts for them
            to count as the same concept.

        min_len_ed (int):
            Words below this length will not be considered the same even if
            they are within the given edit distance.

        stemmed (bool):
            If True the words are already stemmed.

    Returns:
        result (set):
            Set of matching words between the two lists.
    '''

    # Short-cut
    if ( len( a ) == 0 ) or ( len( b ) == 0 ):
        return set( [] )

    # Stem the lists first
    if not stemmed:
        sa, sb = stem( a ), stem( b )
    else:
        sa, sb = a, b

    # When not including edit distance
    if max_edit_distance is None:
        result = []
        for c in sa:
            if c in sb:
                result.append( c )
        result = set( result )
        return result

    # The words in at least one list (list b) have to be sufficiently long
    # to avoid warping one short word into another.
    word_lens = np.array([ len( _ ) for _ in sb ])
    satisfies_min_len_ed = word_lens >= min_len_ed

    # Look for concepts with a sufficiently low Levenshtein edit-distance
    result = []
    for c in sa:

        # Find matches
        edit_distances = np.array([
            edit_distance( c, c_check, )
            for c_check
            in sb
        ])

        # Matches if below minimum edit distance and not too short, or
        # if no edits are required
        is_matching = (
            ( edit_distances <= max_edit_distance ) &
            satisfies_min_len_ed
        )
        is_matching = np.logical_or( is_matching, edit_distances == 0 )
        matches = sb[is_matching]

        # Skip when there's no matches
        if len( matches ) == 0:
            continue

        # Count the matches and represent by maximum count
        count = Counter( matches )
        true_c = verdict.Dict( count ).keymax()[0]
        
        # Store
        result.append( true_c )

    result = set( result )

    return result

########################################################################

def stem( l ):
    '''Stem the words in a list of words.
    
    Args:
        l (list of strs):
            The words to stem.

    Returns:
        sl (list of strs):
            List of stemmed words.
    '''
    s = nltk.stem.SnowballStemmer( 'english' )
    sl = []
    for c in l:
        words = nltk.word_tokenize( c )
        stemmed_words = [ s.stem( w ) for w in words ]
        sl.append( ' '.join( stemmed_words ) )
    sl = np.array( list( set( sl ) ) )
    return sl

########################################################################

def multicolor_text(
    x,
    y,
    strings,
    colors,
    fontweights = None,
    ax = None,
    fontsize = 24,
    annotated = False,
    annote_fontsize = 12,
    spacing = 1.1,
    **kwargs
):

    if ax is None:
        figure = plt.figure( figsize=(16,0.1), facecolor='w' )
        ax = plt.gca()

    n_lines = 0
    t = ax.transData
    for i, s in enumerate( strings ):

        c = colors[i]

        if fontweights is not None:
            fontweight = fontweights[i]
        else:
            fontweight = None

        text = ax.text(
            x,
            y,
            r'' + s + " ",
            color = c,
            transform = t,
            fontweight = fontweight,
            fontsize = fontsize,
            **kwargs
        )

        # Need to draw to update the text position.
        text.draw( ax.figure.canvas.get_renderer() )
        ex = text.get_window_extent()

        if annotated:
            annot_t = transforms.offset_copy(
                t,
                y = ex.height,
                units = 'dots',
            )
            ax.text(
                x,
                y,
                str( i ),
                transform = annot_t,
                color = '0.7',
                fontsize = annote_fontsize,
                **kwargs
            )

        # Wrap
        out_of_bounds = ex.transformed( ax.transAxes.inverted() ).x1 > 1.
        if not out_of_bounds:

            # Normal offset
            t = transforms.offset_copy(
                t,
                x = ex.width,
                units = 'dots',
            )

        else:

            # Wrap reset
            n_lines += 1
            t = ax.transData
            t = transforms.offset_copy(
                t,
                y = -n_lines * ex.height * ( 1. + annotated ) * spacing,
                units = 'dots',
            )

