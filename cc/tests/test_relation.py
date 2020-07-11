from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.relation as relation

########################################################################

class TestInnerProduct( unittest.TestCase ):

    def test_inner_product( self ):

        a = '[Dogs] and [cats] are [animals].'
        b = '[Cats] and [dogs] are not [people].'

        actual = relation.inner_product( a, b )
        expected = 2

        assert actual == expected

    ########################################################################

    def test_inner_product_nested( self ):

        a = '[Big [dogs]] and [cats] are [animals].'
        b = '[Cats] and [dogs] are not [people].'

        actual = relation.inner_product( a, b )
        expected = 2

        assert actual == expected

    ########################################################################

    def test_inner_product_two_word_concept( self ):

        a = '[Big dogs] and [cats] are [animals].'
        b = '[Cats] and [dogs] are not [people].'

        actual = relation.inner_product( a, b )
        expected = 2

        assert actual == expected

    ########################################################################

    def test_inner_product_self( self ):

        a = 'Uses a [particle-tracking] analysis applied to the [FIRE-2 simulations] to study the [origins of the [CGM]], including [IGM accretion], [galactic wind], and [satellite wind].'
        actual = relation.inner_product( a, a )
        expected = 13

        assert actual == expected

########################################################################

class TestParse( unittest.TestCase ):

    def test_parse_relation_for_kcs_word_per_concept( self ):

        a = '[Big dogs] and [cats] are [animals].'
        actual = relation.parse_relation_for_key_concepts(
            a,
            word_per_concept = True,
        )
        expected = [ 'Big', 'dogs', 'cats', 'animals' ]
        assert actual == expected

        a = '[Big [dogs]] and [cats] are [animals].'
        actual = relation.parse_relation_for_key_concepts(
            a,
            word_per_concept = True,
        )
        expected = [ 'Big', 'dogs', 'cats', 'animals' ]
        assert actual == expected
