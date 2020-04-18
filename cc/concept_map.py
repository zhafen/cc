import itertools
import numpy as np

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

    def finish_evaluation( self, weights, relations ):
        '''Store the evaluated concept map in a useful format.

        Args:
            weights (dict of floats):
                Weights of individual concepts.

            relations (dict of floats):
                Relations between concepts. A value of 1.0 means the concepts
                are identical. A value of 0.0 means the concepts are completely
                independent.

        Modifies:
            self.weights (dict of floats):
                Weights of individual concepts.

            self.relation_matrix (np.ndarray, (n_concepts,n_concepts)):
                Matrix expressing the relationship between concepts.
        '''

        # Store weights
        self.weights = weights

        n = len( self.concepts )
        self.relation_matrix = np.full( ( n, n ), -1.0 )
        for i, c_i in enumerate( self.concepts ):
            for j, c_j in enumerate( self.concepts ):

                # Diagonals
                if i == j:
                    self.relation_matrix[i,j] = 1.
                    continue

                # Retrieve data
                try:
                    value = relations[(c_i,c_j)]
                except KeyError:
                    value = relations[(c_j,c_i)]

                self.relation_matrix[i,j] = value

