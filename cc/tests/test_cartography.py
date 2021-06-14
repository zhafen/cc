import copy
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pytest
import shutil
import unittest
import verdict
import warnings

import cc.atlas as atlas
import cc.cartography as cartography

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

########################################################################

class TestCartographer( unittest.TestCase ):

    def test_init( self ):

        fp = './tests/data/example_atlas/projection.h5'

        data = verdict.Dict.from_hdf5( fp, sparse=True )
        data['components'] = data['components'].toarray()

        # Time that normally breaks things
        data['publication_dates'][0] = '1677-01-01 00:00:00'

        return cartography.Cartographer( **data )

########################################################################

class TestInnerProduct( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_inner_product_publication_publication( self ):

        # Identify the right publication
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        np.random.seed( 1234 )

        expected = ( self.c.components[ind_v,:] * self.c.components[ind_h,:] ).sum()

        actual = self.c.inner_product(
            'Hafen2019',
            'VandeVoort2018a',
        )
        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication_all( self ):

        # Identify the right publication
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        np.random.seed( 1234 )

        expected = ( self.c.components[ind_v,:] * self.c.components[ind_h,:] ).sum()

        actual = self.c.inner_product(
            'Hafen2019',
            'all',
        )
        npt.assert_allclose( actual[ind_v], expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_matrix( self ):

        # Identify the right publication
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        np.random.seed( 1234 )

        expected = ( self.c.components[ind_v,:] * self.c.components[ind_h,:] ).sum()

        actual = self.c.inner_product_matrix
        npt.assert_allclose( actual[ind_h][ind_v], expected, rtol=0.05 )

########################################################################

class TestInnerProductPython( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp, backend='python' )

    ########################################################################

    def test_inner_product_atlas_atlas( self ):

        np.random.seed( 1234 )

        expected = ( self.c.components.sum( axis=0 )**2. ).sum()

        actual = self.c.inner_product( 'atlas', 'atlas' )

        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication_atlas( self ):

        np.random.seed( 1234 )

        # Identify the right publication
        ind = np.argmax( self.c.publications == 'Hafen2019' )

        expected = ( self.c.components * self.c.components[ind,:] ).sum()
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
            self.c.components / np.sqrt( ip_atlas_atlas ) * comp_norm[ind,:]
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

        # Identify the right publication
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        np.random.seed( 1234 )

        expected = ( self.c.components[ind_v,:] * self.c.components[ind_h,:] ).sum()

        actual = self.c.inner_product(
            'Hafen2019',
            'VandeVoort2018a',
        )
        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication_all( self ):

        # Identify the right publication
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        np.random.seed( 1234 )

        expected = ( self.c.components[ind_v,:] * self.c.components[ind_h,:] ).sum()

        actual = self.c.inner_product(
            'Hafen2019',
            'all',
        )

    ########################################################################

    def test_pairwise_inner_product( self ):

        result = self.c.pairwise( 'inner_product' )
        n_pubs = self.c.publications.size
        assert result.size == n_pubs * ( n_pubs - 1 ) / 2

########################################################################

class TestTextOverlap( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_publication_all( self ):

        # Identify the right publications
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        h = self.c.components[ind_h,:]
        v = self.c.components[ind_v,:]
        shared = 0
        for i, h_i in enumerate( h ):
            if h_i > 0 and v[i] > 0:
                shared += min( h_i, v[i] )

        # Order one
        expected = shared / h.sum()
        actual = self.c.text_overlap(
            'Hafen2019',
            'all',
        )
        npt.assert_allclose( actual[ind_v], expected )

        # Order two
        expected = shared / v.sum()
        actual = self.c.text_overlap(
            'all',
            'Hafen2019',
        )
        npt.assert_allclose( actual[ind_v], expected )

        # Symmetric (geometric mean of lengths)
        expected = shared / np.sqrt( v.sum() * h.sum() )
        actual = self.c.symmetric_text_overlap(
            'Hafen2019',
            'all',
        )
        npt.assert_allclose( actual[ind_v], expected )

    ########################################################################

    def test_publication_publication( self ):

        # Identify the right publications
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        h = self.c.components[ind_h,:]
        v = self.c.components[ind_v,:]
        shared = 0
        for i, h_i in enumerate( h ):
            if h_i > 0 and v[i] > 0:
                shared += min( h_i, v[i] )

        # Order one
        expected = shared / h.sum()
        actual = self.c.text_overlap(
            'Hafen2019',
            'VandeVoort2018a',
        )
        npt.assert_allclose( actual, expected )

        # Order two
        expected = shared / v.sum()
        actual = self.c.text_overlap(
            'VandeVoort2018a',
            'Hafen2019',
        )
        npt.assert_allclose( actual, expected )

        # Symmetric (geometric mean of lengths)
        expected = shared / np.sqrt( v.sum() * h.sum() )
        actual = self.c.symmetric_text_overlap(
            'Hafen2019',
            'VandeVoort2018a',
        )
        npt.assert_allclose( actual, expected )
    
    ########################################################################

    def test_pairwise( self ):

        # Pairwise call
        result = self.c.pairwise( 'symmetric_text_overlap' )
        n_pubs = self.c.publications.size
        assert result.size == n_pubs * ( n_pubs - 1 ) / 2

        # Pairwise, duplicates allowed
        result = self.c.pairwise( 'symmetric_text_overlap', trim_and_reshape=False )
        assert result.size == n_pubs**2

        # Alt call
        alt_call_result = self.c.symmetric_text_overlap( 'all', 'all' )
        npt.assert_allclose( result, alt_call_result )
        npt.assert_allclose( np.diag(result), np.ones(n_pubs) )

########################################################################

class TestDistance( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_distance( self ):

        # Identify the right publication
        ind_h = np.argmax( self.c.publications == 'Hafen2019' )
        ind_v = np.argmax( self.c.publications == 'VandeVoort2018a' )

        np.random.seed( 1234 )

        expected = np.sqrt( (
            ( self.c.components_normed[ind_v,:] - self.c.components_normed[ind_h,:] )**2.
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
        self.a = atlas.Atlas( './tests/data/example_atlas', atlas_data_format='hdf5' )

    def tearDown(self):

        # We want to start fresh for these tests
        ads_bib_fp = './tests/data/example_atlas/cc_ads.bib'
        if os.path.isfile( ads_bib_fp ):
            os.remove( ads_bib_fp )

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
        new_a = self.c.explore( 'Hafen2019', self.a, n=1, identifier='arxiv' )
        actual_keys = sorted( list( new_a.data.keys() ) )

        n_duplicates = 7 # Found manually
        assert len( expected_keys ) - n_duplicates == len( actual_keys )

    ########################################################################

    @pytest.mark.slow
    def test_survey( self ):

        np.random.seed( 1234 )

        # Build expected keys
        cite_key = 'Hafen2019a'
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
        new_a = self.c.survey( 'Hafen2019', self.a, 0.6, identifier='arxiv' )
        actual_keys = sorted( list( new_a.data.keys() ) )

        # Check that we have the expected length
        n_duplicates = 8 # Found manually
        assert len( expected_keys ) - n_duplicates == len( actual_keys )

    ########################################################################

    def test_expand( self ):

        n_downloaded = len( self.a['Hafen2019'].references ) + len( self.a['Hafen2019a'].references )
        new_a = self.c.expand( self.a, center='Hafen2019', n_pubs_max=n_downloaded+1 )

        # Check that the new atlas has the old data
        for key, item in self.a.data.items():
            assert new_a.data[key].abstract_str() != ''

        # This is a publication that cites Hafen2020,
        # the most similar publication to Hafen2019
        assert '2020MNRAS.498.1668W' in new_a.data
        # This is a publication cited by VandeVoort2018
        # but not by more similar publications.
        assert '2015PhRvD..92l3526C' not in new_a.data

    ########################################################################

    @patch( 'ads.ExportQuery' )
    def test_expand_check_call( self, mock ):

        # Modify citations to control expected results
        self.a.data['Hafen2019'].citations = []
        self.a.data['Hafen2019a'].citations = []

        # Dummy publication that should still exist, but should not be dual-retrieved
        dummy_key = '2017MNRAS.470.4698A'
        self.a.data[dummy_key] = self.a.data['Howk2017']

        # Build the expected call
        expected_call = (
            list( self.a['Hafen2019'].references ) +
            list( self.a['Hafen2019a'].references )
            # list( self.a['VandeVoort2016'].references )
        )
        expected_call = sorted( list( set( expected_call ) ) )
        expected_call.remove( dummy_key )

        self.c.expand( self.a, center='Hafen2019', n_pubs_max=len( expected_call ) )
        actual_call = mock.call_args[0][0] 

        # Check
        is_not_in = np.invert( np.in1d( actual_call, expected_call ) )
        assert is_not_in.sum() == 0
        for i, key in enumerate( sorted( actual_call ) ):
            assert key == expected_call[i]

    ########################################################################

    @pytest.mark.slow
    def test_expand_no_center( self ):

        new_a = self.c.expand( self.a, n_pubs_max=2000 )

        # Check that the new atlas has the old data
        for key, item in self.a.data.items():
            assert new_a.data[key].abstract_str() != ''

        # Check that the new atlas consists of new references
        expected_keys = list( self.a.data.keys() )
        for key, item in self.a.data.items():
            expected_keys = np.union1d(
                expected_keys,
                self.a[key].citations
            )
            expected_keys = np.union1d(
                expected_keys,
                self.a[key].references
            )
        expected_keys = sorted( list( expected_keys ) )
        actual_keys = sorted( list( new_a.data.keys() ) )

        missing_from_actual = [ _ not in actual_keys for _ in expected_keys ]
        missing_from_actual = np.array( expected_keys )[missing_from_actual]
        missing_from_expected = [ _ not in expected_keys for _ in actual_keys ]
        missing_from_expected = np.array( actual_keys )[missing_from_expected]

        # Any missing ones should only be because of papers that previously weren't
        # published, but now are published
        for key in missing_from_actual:
            assert 'arXiv' in key or 'tmp' in key

    ########################################################################

    def test_record_update_history( self ):

        input = [
            [ 'Hafen2019', 'Hafen2019a' ],
            [ 'Hafen2019', 'Hafen2019a', 'Stern2018', 'Howk2017' ],
            [ 'Hafen2019', 'Hafen2019a', 'Stern2018', 'Howk2017', 'VandeVoort2018a' ],
        ]

        self.c.record_update_history( input )

        expected = np.array([ 2, 1, 1, -2, -2, -2, -2, -2, 0, 0 ])
        npt.assert_allclose( expected, self.c.update_history )

    ########################################################################

    def test_converged_kernel_size( self ):

        # Setup mock data
        self.c.update_history = np.array([ 2, 1, 1, 3, 3, 4, 1, 1, 0, 0 ])

        actual, actual_cospsis = self.c.converged_kernel_size( 'Hafen2019' )
        expected = np.array([ 1, 2, 2, 2, ])
        npt.assert_allclose( expected, actual )
        expected_cospsis = np.array([ 0.6257132252113373, 0.53010468, 0.53010468, 0.53010468, ])
        npt.assert_allclose( expected_cospsis, actual_cospsis, rtol=1e-3 )

    ########################################################################

    def test_converged_kernel_size_all( self ):

        # Setup mock data
        self.c.update_history = np.array([ 2, 1, 1, 3, 3, 4, 1, 1, 0, 0 ])

        actual, actual_cospsis = self.c.converged_kernel_size( 'all' )
        expected = np.array([ 1, 2, 2, 2 ])
        npt.assert_allclose( expected, actual[self.c.publications=='Hafen2019'][0] )
        expected_cospsis = np.array([ 0.6257132252113373, 0.53010468, 0.53010468, 0.53010468, ])
        npt.assert_allclose( expected_cospsis, actual_cospsis[self.c.publications=='Hafen2019'][0], rtol=1e-3 )

        expected = np.array([ -1, -1, 2, 2, ])
        npt.assert_allclose( expected, actual[self.c.publications=='VandeVoort2018a'][0] )

    ########################################################################

    def test_converged_kernel_size_python( self ):

        # Setup mock data
        self.c.update_history = np.array([ 2, 1, 1, 3, 3, 4, 1, 1, 0, 0 ])

        actual, actual_cospsis = self.c.converged_kernel_size( 'Hafen2019', python=True )
        expected = np.array([ 1, 2, 2, 2, ])
        npt.assert_allclose( expected, actual )
        expected_cospsis = np.array([ 0.6257132252113373, 0.53010468, 0.53010468, 0.53010468, ])
        npt.assert_allclose( expected_cospsis, actual_cospsis, rtol=1e-3 )

    ########################################################################

    def test_converged_kernel_size_all_python( self ):

        # Setup mock data
        self.c.update_history = np.array([ 2, 1, 1, 3, 3, 4, 1, 1, 0, 0 ])

        actual, actual_cospsis = self.c.converged_kernel_size( 'all', python=True )
        expected = np.array([ 1, 2, 2, 2 ])
        npt.assert_allclose( expected, actual[self.c.publications=='Hafen2019'][0] )
        expected_cospsis = np.array([ 0.6257132252113373, 0.53010468, 0.53010468, 0.53010468, ])
        npt.assert_allclose( expected_cospsis, actual_cospsis[self.c.publications=='Hafen2019'][0], rtol=1e-3 )

        expected = np.array([ -1, -1, 2, 2, ])
        npt.assert_allclose( expected, actual[self.c.publications=='VandeVoort2018a'][0] )

    ########################################################################

    def test_converged_kernel_size_random_subset_python( self ):

        # Setup mock data
        self.c.update_history = np.array([ 2, 1, 1, 3, 3, 4, 1, 1, 0, 0 ])

        actual, actual_cospsis = self.c.converged_kernel_size( 5, python=True )
        assert actual.shape == ( 5, 4 )
        assert actual_cospsis.shape == ( 5, 4 )

########################################################################

class TestAsymmetryMetric( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

    ########################################################################

    def test_general( self ):

        result = self.c.topography_metric()
        assert result.shape == self.c.publications.shape

    ########################################################################

    def test_avoid_nans( self ):

        # Replace the first row with zeros to test if handled
        self.c.components[0] = np.zeros( self.c.components[0].size )

        result = self.c.topography_metric( date_type='publication_dates' )

        # There should be exactly two well-understood nans
        assert np.isnan( result[1:] ).sum() == 2

    ########################################################################

    def test_avoid_zeros( self ):

        # Replace the first row with zeros to test if handled
        self.c.components[0] = np.zeros( self.c.components[0].size )

        result = self.c.topography_metric()

        assert not np.any( np.isclose( result, 0. ) )

    ########################################################################

    def test_constant_asymmetry_metric( self ):

        # Try for some publication
        actual = self.c.topography_metric(
            [ 3, ],
            metric = 'constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert not np.isnan( actual[0] )

        # Try for a file with a nan publication date.
        actual = self.c.topography_metric(
            [ 0, ],
            metric = 'constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert np.isnan( actual[0] )

    ########################################################################

    def test_kernel_constant_asymmetry_metric( self ):

        # Try for some publication
        actual = self.c.topography_metric(
            [ 3, ],
            metric = 'kernel_constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert not np.isnan( actual[0] )

        # Try for a file with a nan publication date.
        actual = self.c.topography_metric(
            [ 0, ],
            metric = 'kernel_constant_asymmetry',
            date_type = 'publication_dates'
        )
        assert np.isnan( actual[0] )

    ########################################################################

    def test_density_metric( self ):

        with warnings.catch_warnings(record=True) as w:

            # Try for some publication
            actual = self.c.topography_metric(
                [ 3, ],
                metric = 'density',
                date_type = 'publication_dates'
            )
            assert not np.isnan( actual[0] )

            # Try for a file with a nan publication date.
            actual = self.c.topography_metric(
                [ 0, ],
                metric = 'density',
                date_type = 'publication_dates'
            )
            assert np.isnan( actual[0] )

            assert len( w ) == 2

    ########################################################################

    def test_smoothing_length_metric( self ):

        # Try for some publication
        actual = self.c.topography_metric(
            [ 3, ],
            metric = 'smoothing_length',
            date_type = 'publication_dates'
        )
        assert not np.isnan( actual[0] )

        # Try for a file with a nan publication date.
        actual = self.c.topography_metric(
            [ 0, ],
            metric = 'smoothing_length',
            date_type = 'publication_dates'
        )
        assert np.isnan( actual[0] )

########################################################################

class TestSimilarityMetric( unittest.TestCase ):

    def setUp( self ):

        fp = './tests/data/example_atlas/projection.h5'
        self.c = cartography.Cartographer.from_hdf5( fp )

        self.similarity_metrics = [
            'inner_product',
            'cospsi',
            'psi',
            'text_overlap',
            'symmetric_text_overlap',
            'distance'
        ]

    ########################################################################

    def test_consistent_lengths( self ):

        metric_values = {}
        for metric in self.similarity_metrics:
            print( metric )
            metric_values[metric] = self.c.pairwise( metric, )

        n_pubs = self.c.publications.size
        expected_size = n_pubs * ( n_pubs - 1 ) / 2

        for key, item in metric_values.items():
            assert item.size == expected_size