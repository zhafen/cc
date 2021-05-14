import numba
from numba.typed import List
import numpy as np

from . import utils

########################################################################

def concept_projection( words, component_concepts=None ):
    '''Given a list of words, project into concept space.

    Args:
        words (list-like of strs):
            Words to project into concept space.

        component_concepts (list-like of strs):
            Existing concepts to project onto.

    Returns:
        components (list-like of floats):
            Components in new space.
       
        component_concepts
            Concepts for the space.
    '''

    # Project for non-zero concepts
    flattened = np.hstack( words )
    nonzero_concepts, values = np.unique( flattened, return_counts=True )

    # Combine with existing component concepts
    if component_concepts is not None:
        components, component_concepts = project_onto_existing(
            original_concepts = component_concepts,
            added_components = values,
            added_concepts = nonzero_concepts,
        )
        
    else:
        components = values
        component_concepts = nonzero_concepts

    return components, component_concepts

########################################################################

def project_onto_existing(
    original_concepts,
    added_components,
    added_concepts,
):
    '''Project onto an existing list of concepts.

    Args:
        original_concepts (list-like of strs):
            Original concepts to project onto.

        added_components (list-like of floats or ints):
            Weights for the new words.

        added_concepts (list-like of strs):
            Concepts for the new vector. Some may already be part of
            original_concepts

    Returns:
        components (list-like of floats):
            Components in new space.
       
        component_concepts
            Concepts for the space.
    '''

    @numba.njit
    def numba_fn(
        original_concepts,
        added_components,
        added_concepts,
    ):

        # Store the concepts shared with other publications
        dup_inds = List()
        components = List()
        component_concepts = List( original_concepts )
        for i, ci in enumerate( original_concepts ):
            no_match = True
            for j, cj in enumerate( added_concepts ):
                # If a match is found
                if ci == cj:
                    components.append( added_components[j] )
                    dup_inds.append( j )
                    no_match = False
                    break
            # If made to the end of the loop with no match
            if no_match:
                components.append( 0 )

        # Finish combining
        for i in range( len( added_concepts ) ):
            if i in dup_inds:
                continue
            components.append( added_components[i] )
            component_concepts.append( added_concepts[i] )

        return components, component_concepts

    components, component_concepts = numba_fn(
        List( original_concepts ),
        List( added_components ),
        List( added_concepts ),
    )

    return np.array( components ), np.array( component_concepts )

########################################################################

def inner_product( a, b, word_per_concept=True, **kwargs ):
    '''The inner product between two relations.
    Fiducially defined as the number of shared key concepts.

    Args:
        a (str):
            The first relation to calculate the inner product of.

        b (str):
            The second relation to calculate the inner product of.

        word_per_concept (bool):
            If True, break each concept into its composite words.

    Kwargs:
        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two concepts for them
            to count as the same concept.

    Returns:
        result (int):
            The number of shared key concepts between two relations.
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
            If True, limit concepts to one word per concept, and break down
            multi-word concepts (including nested concepts) into individuals.

    Returns:
        list of strs:
            The key concepts contained in the relation.
    '''

    # Parse key concepts, including nested brackets
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
