import itertools
import numpy as np

import augment
import verdict

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
        self.weights = np.array([ weights[c] for c in self.concepts ])

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

    ########################################################################

    def save( self, filepath ):
        '''Save the concept map to a .hdf5 file.

        Args:
            filepath (str): Location to save the file.
        '''

        # Prep
        data = verdict.Dict( {} )
        for attr in [ 'concepts', 'weights', 'relation_matrix' ]:
            data[attr] = getattr( self, attr )

        # Save
        data.to_hdf5( filepath )

    ########################################################################

    @classmethod
    def load( cls, filepath ):
        '''Load a concept map from a .hdf5 file.

        Args:
            filepath (str): Where to load the file from.
        '''

        data = verdict.Dict.from_hdf5( filepath )

        result = ConceptMap( list( data['concepts'] ) )
        result.weights = data['weights']
        result.relation_matrix = data['relation_matrix']

        return result
