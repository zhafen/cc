import copy
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pytest
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

class TestDistance( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_distance( self ):

        np.random.seed( 1234 )

        expected = np.sqrt( (
            ( self.c.components_normed[0,:] - self.c.components_normed[8,:] )**2.
        ).sum() )

        actual = self.c.distance(
            'Hafen2019',
            'VandeVoort2018a',
        )
        npt.assert_allclose( actual, expected, rtol=0.05 )

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

    @pytest.mark.slow
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

        missing_from_actual = '2016MNRAS.463.4533V'
        new_a = self.c.explore( 'Hafen2019', self.a, n=1 )
        actual_keys = sorted( list( new_a.data.keys() ) )

        n_duplicates = 7 # Found manually
        assert len( expected_keys ) - n_duplicates == len( actual_keys )

    ########################################################################

    @pytest.mark.slow
    def test_survey( self ):

        np.random.seed( 1234 )

        previous_size = len( self.a.data )

        # Build expected keys
        cite_key = 'Howk2017'
        expected_keys = np.union1d(
            list( self.a.data.keys() ),
            self.a[cite_key].citations
        )
        expected_keys = np.union1d(
            expected_keys,
            self.a[cite_key].references
        )
        expected_keys = sorted( list( expected_keys ) )

        # Calculation
        new_a = self.c.survey( 'Hafen2019', self.a, 0.5 )
        actual_keys = sorted( list( new_a.data.keys() ) )

        # Check that we have the expected length
        n_duplicates = 3 # Found manually
        assert len( expected_keys ) - n_duplicates == len( actual_keys )

########################################################################

class TestAsymmetryEstimator( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_general( self ):

        result = self.c.citation_estimator()
        assert result.shape == self.c.publications.shape

    ########################################################################

    def test_avoid_nans( self ):

        # Replace the first row with zeros to test if handled
        self.c.components[0] = np.zeros( self.c.components[0].size )

        result = self.c.citation_estimator( date_type='publication_dates' )

        # There should be exactly two well-understood nans
        assert np.isnan( result[1:] ).sum() == 2

    ########################################################################

    def test_avoid_zeros( self ):

        # Replace the first row with zeros to test if handled
        self.c.components[0] = np.zeros( self.c.components[0].size )

        result = self.c.citation_estimator()

        assert not np.any( np.isclose( result, 0. ) )

    ########################################################################

    def test_constant_asymmetry_estimator( self ):

        # Try for some publication
        actual = self.c.citation_estimator(
            [ 3, ],
            estimator = 'constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert not np.isnan( actual[0] )

        # Try for a file with a nan publication date.
        actual = self.c.citation_estimator(
            [ 0, ],
            estimator = 'constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert np.isnan( actual[0] )

    ########################################################################

    def test_kernel_constant_asymmetry_estimator( self ):

        # Try for some publication
        actual = self.c.citation_estimator(
            [ 3, ],
            estimator = 'kernel_constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert not np.isnan( actual[0] )

        # Try for a file with a nan publication date.
        actual = self.c.citation_estimator(
            [ 0, ],
            estimator = 'kernel_constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert np.isnan( actual[0] )

    ########################################################################

    def test_density_estimator( self ):

        # Try for some publication
        actual = self.c.citation_estimator(
            [ 3, ],
            estimator = 'density',
            date_type = 'publication_dates'
        )
        assert not np.isnan( actual[0] )

        # Try for a file with a nan publication date.
        actual = self.c.citation_estimator(
            [ 0, ],
            estimator = 'density',
            date_type = 'publication_dates'
        )
        assert np.isnan( actual[0] )

        
