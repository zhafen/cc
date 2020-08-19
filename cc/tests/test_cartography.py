import copy
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import shutil
import unittest

import cc.atlas as atlas
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

########################################################################

class TestExplore( unittest.TestCase ):

    def setUp( self ):

        # We want to start fresh for these tests
        ads_bib_fp = './tests/data/example_atlas/cc_ads.bib'
        if os.path.isfile( ads_bib_fp ):
            os.remove( ads_bib_fp )

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )
        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_explore( self ):

        previous_size = len( self.a.data )

        # Build expected keys
        expected_keys = np.union1d(
            list( self.a.data.keys() ),
            self.a['Hafen2019'].citations
        )
        expected_keys = np.union1d(
            expected_keys,
            self.a['Hafen2019'].references
        )
        expected_keys = sorted( list( expected_keys ) )

        new_a = self.c.explore( 'Hafen2019', self.a, n=1 )
        actual_keys = sorted( list( new_a.data.keys() ) )

        n_duplicates = 7 # Found manually
        assert len( expected_keys ) - n_duplicates == len( actual_keys )

########################################################################

class TestAsymmetryEstimator( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_general( self ):

        result = self.c.asymmetry_estimator()
        assert result.shape == self.c.publications.shape

    ########################################################################

    def test_avoid_nans( self ):

        # Replace the first row with zeros to test if handled
        self.c.components[0] = np.zeros( self.c.components[0].size )

        result = self.c.asymmetry_estimator( date_type='publication_dates' )

        # There should be exactly two well-understood nans
        assert np.isnan( result[1:] ).sum() == 2

    ########################################################################

    def test_avoid_zeros( self ):

        # Replace the first row with zeros to test if handled
        self.c.components[0] = np.zeros( self.c.components[0].size )

        result = self.c.asymmetry_estimator()

        assert not np.any( np.isclose( result, 0. ) )

    ########################################################################

    def test_constant_estimator( self ):

        # Try for some publication
        dir, actual = self.c.constant_asymmetry_estimator( 3, date_type='publication_dates' )
        assert dir.shape == self.c.component_concepts.shape
        assert not np.isnan( actual )

        # Try for a file with a nan publication date.
        dir, actual = self.c.constant_asymmetry_estimator( 0, date_type='publication_dates' )
        assert dir.shape == self.c.component_concepts.shape
        assert np.isnan( actual )
