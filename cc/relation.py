import numba
from numba.typed import List
import numpy as np

from . import utils

########################################################################

def vectorize( words, feature_names=None ):
    '''Given a list of words, project into conckey_concept space.

    Args:
        words (list-like of strs):
            Words to project into conckey_concept space.

        feature_names (list-like of strs):
            Existing features to project onto.

    Returns:
        vector (list-like of floats):
            Components in new space.
       
        feature_names
            Concepts for the space.
    '''

    # Project for non-zero features
    flattened = np.hstack( words )
    nonzero_features, values = np.unique( flattened, return_counts=True )

    # Combine with existing component features
    if feature_names is not None:
        vector, feature_names = project_onto_existing(
            original_feature_names = feature_names,
            added_vector = values,
            added_feature_names = nonzero_features,
        )
        
    else:
        vector = values
        feature_names = nonzero_features

    return vector, feature_names

########################################################################

def project_onto_existing(
    original_feature_names,
    added_vector,
    added_feature_names,
):
    '''Project onto an existing list of features.

    Args:
        original_feature_names (list-like of strs):
            Original features to project onto.

        added_vector (list-like of floats or ints):
            Weights for the new words.

        added_feature_names (list-like of strs):
            Concepts for the new vector. Some may already be part of
            original_feature_names

    Returns:
        vector (list-like of floats):
            Components in new space.
       
        feature_names
            Concepts for the space.
    '''

    @numba.njit
    def numba_fn(
        original_feature_names,
        added_vector,
        added_feature_names,
    ):

        # Store the features shared with other publications
        dup_inds = List()
        vector = List()
        feature_names = List( original_feature_names )
        for i, ci in enumerate( original_feature_names ):
            no_match = True
            for j, cj in enumerate( added_feature_names ):
                # If a match is found
                if ci == cj:
                    vector.append( added_vector[j] )
                    dup_inds.append( j )
                    no_match = False
                    break
            # If made to the end of the loop with no match
            if no_match:
                vector.append( 0 )

        # Finish combining
        for i in range( len( added_feature_names ) ):
            if i in dup_inds:
                continue
            vector.append( added_vector[i] )
            feature_names.append( added_feature_names[i] )

        return vector, feature_names

    vector, feature_names = numba_fn(
        List( original_feature_names ),
        List( added_vector ),
        List( added_feature_names ),
    )

    return np.array( vector ), np.array( feature_names )

########################################################################

def inner_product( a, b, word_per_concept=True, **kwargs ):
    '''The inner product between two relations.
    Fiducially defined as the number of shared key features.

    Args:
        a (str):
            The first relation to calculate the inner product of.

        b (str):
            The second relation to calculate the inner product of.

        word_per_concept (bool):
            If True, break each conckey_concept into its composite words.

    Kwargs:
        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two features for them
            to count as the same concept.

    Returns:
        result (int):
            The number of shared key features between two relations.
    '''

    a_kcs = parse_relation_for_key_concepts(
        a,
        word_per_concept = word_per_concept
    )
    b_kcs = parse_relation_for_key_concepts(
        b,
        word_per_concept = word_per_concept
    )

    a_kcs = utils.uniquify_words( a_kcs, **kwargs )
    b_kcs = utils.uniquify_words( b_kcs, **kwargs )

    result = 0
    for a_kc in a_kcs:
        for b_kc in b_kcs:
            if a_kc == b_kc:
                result += 1

    return result

########################################################################

def parse_relation_for_key_concepts( a, word_per_concept=True ):
    '''Parse a formatted relation for key_concepts, including nested brackets.

    Args:
        a (str):
            The written relation you want to parse.

        word_per_concept (bool):
            If True, limit features to one word per concept, and break down
            multi-word features (including nested features) into individuals.

    Returns:
        list of strs:
            The key features contained in the relation.
    '''

    # Parse key features, including nested brackets
    key_concepts = []
    stack = []
    nesting = 0
    for i, char in enumerate( a ):
        if char == '[':
            nesting += 1
            stack.append( i )
        elif char == ']' and stack:
            nesting -= 1
            start = stack.pop()
            key_concept = a[start+1:i]
            key_concept = key_concept.replace( '[', '' )
            key_concept = key_concept.replace( ']', '' )

            if word_per_concept:
                # Only include top level, which is broken up.
                if nesting == 0:
                    kcs = key_concept.split( ' ' )
                    [ key_concepts.append( kc ) for kc in kcs ]

            else:
                key_concepts.append( key_concept )

    return key_concepts
