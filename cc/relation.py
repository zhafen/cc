import augment

########################################################################

def relation_inner_product( a, b ):
    '''The inner product between two relations.
    '''

    pass

########################################################################

def parse_relation_for_key_concepts( point ):
    '''Parse a formatted relation for key_concepts, including nested brackets.

    Args:
        point (str):
            The written relation you want to parse.

    Returns:
        list of strs:
            The key concepts contained in the relation.
    '''

    # Parse key concepts, including nested brackets
    key_concepts = []
    stack = []
    for i, char in enumerate( point ):
        if char == '[':
            stack.append( i )
        elif char == ']' and stack:
            start = stack.pop()
            key_concept = point[start+1:i]
            key_concept = key_concept.replace( '[', '' )
            key_concept = key_concept.replace( ']', '' )
            key_concepts.append( key_concept )

    return key_concepts
