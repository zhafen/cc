from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.concept_map

########################################################################

class TestManualEvaluation( unittest.TestCase ):

    def test_start_manual_evaluation( self ):

        concepts = [ 'red', 'blue', 'dog' ]
        cm = cc.concept_map.ConceptMap( concepts )

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

        requested_weights, requested_relations = cm.start_evaluation()

        assert expected_weights == requested_weights
        assert expected_relations == requested_relations

