import copy
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import shutil
import unittest

import cc.cartography as cartography

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestInnerProduct( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_inner_product_atlas_atlas( self ):

        np.random.seed( 1234 )

        expected = ( self.c.components.sum( axis=0 )**2. ).sum()

        actual = self.c.inner_product( 'atlas', 'atlas' )

        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication_atlas( self ):

        np.random.seed( 1234 )

        expected = ( self.c.components * self.c.components[8,:] ).sum()
        actual = self.c.inner_product(
            'Hafen2019',
            'atlas',
        )
        npt.assert_allclose( actual, expected, rtol=0.05 )

        ip_atlas_atlas = self.c.inner_product( 'atlas', 'atlas' )
        ip_pub_pub = self.c.inner_product( 'Hafen2019', 'Hafen2019' )
        actual = actual / np.sqrt( ip_atlas_atlas * ip_pub_pub )
        comp_norm = self.c.components / self.c.norms[:,np.newaxis]
        expected = (
            self.c.components / np.sqrt( ip_atlas_atlas ) * comp_norm[8,:]
        ).sum()
        assert actual < 1.
        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################
    
    def test_inner_product_all_atlas( self ):

        np.random.seed( 1234 )

        expected = ( self.c.components * self.c.components[8,:] ).sum()

        actual = self.c.inner_product(
            'all',
            'atlas',
        )

        npt.assert_allclose( actual[8], expected, rtol=0.05 )

    ########################################################################
    
    def test_inner_product_all_all( self ):

        np.random.seed( 1234 )

        expected = self.c.norms**2.

        actual = self.c.inner_product(
            'all',
            'all',
        )

        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication_publication( self ):

        np.random.seed( 1234 )

        expected = ( self.c.components[0,:] * self.c.components[8,:] ).sum()

        actual = self.c.inner_product(
            'Hafen2019',
            'VandeVoort2018a',
        )
        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication_all( self ):

        np.random.seed( 1234 )

        expected = ( self.c.components[0,:] * self.c.components[8,:] ).sum()

        actual = self.c.inner_product(
            'Hafen2019',
            'all',
        )
        npt.assert_allclose( actual[0], expected, rtol=0.05 )
