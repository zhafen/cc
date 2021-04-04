import augment

from . import utils

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
