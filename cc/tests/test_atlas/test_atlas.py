from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.atlas.atlas as atlas

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestRetrieveData( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_bibtex( self ):

        bibtex_fp = './tests/data/example.bib'

        self.a.import_bibtex( bibtex_fp )

        assert self.a.data['Hafen2019'].citation['eprint'] == '1811.11753'

########################################################################

class TestKeyConcepts( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_unique_key_concepts( self ):

        self.a._all_key_concepts = [
            'dog',
            'dogs',
            'kitty cat',
        ]

        expected = set( [
            'dog',
            'kitti cat',
        ] )
        assert self.a.get_unique_key_concepts() == expected

    ########################################################################

    def test_unique_key_concepts_forgotten_space( self ):

        self.a._all_key_concepts = [
            'dog',
            'dogs',
            'kittycat',
            'kitty cat',
        ]

        expected = set( [
            'dog',
            'kitti cat',
        ] )
        assert self.a.get_unique_key_concepts() == expected
