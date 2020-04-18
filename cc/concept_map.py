import itertools

import augment

########################################################################

class ConceptMap( object ):

    @augment.store_parameters
    def __init__( self, concepts ):

        pass

    ########################################################################

    def start_evaluation( self ):
        '''Yields the concepts and relations that need to be evaluated for the
        concept map to be complete.

        Returns:
            requested_concepts (list of strs):
                Concepts missing information needed to complete the map.

            requested_relations (list of strs):
                Relations missing information needed to complete the map.
        '''

        # Evaluate relations to request
        concept_products = list(
            itertools.product( self.concepts, self.concepts )
        )
        requested_relations = []
        for concept_product in concept_products:
            if concept_product[0] == concept_product[1]:
                continue
            elif concept_product[::-1] in requested_relations:
                continue
            else:
                requested_relations.append( concept_product )

        return self.concepts, requested_relations

    ########################################################################

    def complete_manual_evaluation( self ):

        pass
