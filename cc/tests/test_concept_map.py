from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.concept_map

########################################################################

class TestConceptMap( unittest.TestCase ):

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

        expected_weights = np.array([ weights[c] for c in self.cm.concepts ])
        expected_relation_matrix = np.array([
            [ 1.0, 0.9, 0.3, ],
            [ 0.9, 1.0, 0.2, ],
            [ 0.3, 0.2, 1.0, ],
        ])

        npt.assert_allclose( expected_weights, self.cm.weights )
        npt.assert_allclose( expected_relation_matrix, self.cm.relation_matrix )

    ########################################################################

    def test_finish_evaluation_more_concepts( self ):

        # Initial setup
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

        # W/ additional concepts
        concepts = [ 'red', 'blue', 'dog', 'cat' ]
        weights = {
            'red': 0.5,
            'blue': 0.5,
            'dog': 1.0,
            'cat': 1.5,
        }
        relations = {
            ( 'cat', 'red' ): 0.1,
            ( 'cat', 'blue' ): 0.2,
            ( 'cat', 'dog' ): 0.7,
        }
        self.cm.finish_evaluation( weights, relations, concepts )

        expected_weights = np.array([ 0.5, 0.5, 1.0, 1.5 ])
        expected_relation_matrix = np.array([
            [ 1.0, 0.9, 0.3, 0.1, ],
            [ 0.9, 1.0, 0.2, 0.2, ],
            [ 0.3, 0.2, 1.0, 0.7, ],
            [ 0.1, 0.2, 0.7, 1.0, ],
        ])

        npt.assert_allclose( expected_weights, self.cm.weights )
        npt.assert_allclose( expected_relation_matrix, self.cm.relation_matrix )

    ########################################################################

    def test_save_and_load( self ):

        # Setup
        fp = './tests/data/example_concept_map.hdf5'
        self.cm.weights = np.array([ 0.5, 0.5, 1.0 ])
        self.cm.relation_matrix = np.array([
            [ 1.0, 0.9, 0.3, ],
            [ 0.9, 1.0, 0.2, ],
            [ 0.3, 0.2, 1.0, ],
        ])

        # Test
        self.cm.save( fp )
        cm = cc.concept_map.ConceptMap.load( fp )

        assert cm.concepts == self.cm.concepts
        npt.assert_allclose( cm.weights, self.cm.weights )
        npt.assert_allclose( cm.relation_matrix, self.cm.relation_matrix )
    
