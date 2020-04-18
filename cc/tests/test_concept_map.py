from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.concept_map

########################################################################

class TestManualEvaluation( unittest.TestCase ):

    def setUp( self ):

        concepts = [ 'red', 'blue', 'dog' ]
        self.cm = cc.concept_map.ConceptMap( concepts )

    ########################################################################
    
    def test_start_evaluation( self ):

        expected_weights = [
            'red',
            'blue',
            'dog',
        ]
        expected_relations = [
            ( 'red', 'blue' ),
            ( 'red', 'dog' ),
            ( 'blue', 'dog' ),
        ]

        requested_weights, requested_relations = self.cm.start_evaluation()

        assert expected_weights == requested_weights
        assert expected_relations == requested_relations

    ########################################################################

    def test_finish_evaluation( self ):

        weights = {
            'red': 0.5,
            'blue': 0.5,
            'dog': 1.0,
        }
        relations = {
            ( 'red', 'blue' ): 0.9,
            ( 'red', 'dog' ): 0.3,
            ( 'blue', 'dog' ): 0.2,
        }

        self.cm.finish_evaluation( weights, relations )

        expected_relation_matrix = np.array([
            [ 1.0, 0.9, 0.3, ],
            [ 0.9, 1.0, 0.2, ],
            [ 0.3, 0.2, 1.0, ],
        ])

        assert weights == self.cm.weights
        npt.assert_allclose( expected_relation_matrix, self.cm.relation_matrix )
