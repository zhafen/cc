import copy
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import shutil
import unittest

import cc.atlas as atlas

import verdict

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestBibTexData( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_bibtex( self ):

        self.a.import_bibtex( './tests/data/example_atlas/example.bib' )

        assert self.a.data['Hafen2019'].citation['eprint'] == '1811.11753'

    ########################################################################

    def test_process_bibtex_anotations( self ):

        self.a.data.process_bibtex_annotations()

        before = copy.deepcopy( self.a.data['Hafen2019'].notes )

        # Make sure that it caches
        self.a.data.process_bibtex_annotations()

        after = self.a.data['Hafen2019'].notes

        assert before == after

########################################################################

class TestFromBibcodes( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

        self.bibtex_fp = './tests/data/example_atlas/cc_ads.bib' 

    def tearDown( self ):
        if os.path.isfile( self.bibtex_fp ):
            os.remove( self.bibtex_fp )

    ########################################################################

    def test_from_bibcodes( self ):

        bibcodes = [
            '2019MNRAS.488.1248H',
            '2020MNRAS.494.3581H',
        ]

        a = atlas.Atlas.from_bibcodes(
            self.a.atlas_dir,
            bibcodes,
        )

        # Saved in the right spot
        assert a.bibtex_fp == self.bibtex_fp

        # Expected values for entries
        for key in [ 'title', 'year', 'arxivid' ]:
            assert a.data['2019MNRAS.488.1248H'].citation[key] == self.a.data['Hafen2019'].citation[key]

    ########################################################################

    def test_from_bibcodes_existing_bib( self ):

        # Copy the duplicate file so there's already a bib there
        shutil.copyfile(
            './tests/data/duplicate.bib',
            self.bibtex_fp,
        )

        bibcodes = [
            '2020MNRAS.494.3581H',
        ]

        a = atlas.Atlas.from_bibcodes(
            self.a.atlas_dir,
            bibcodes,
        )

        # Saved in the right spot
        assert a.bibtex_fp == self.bibtex_fp

        # Expected values for entries
        for key in [ 'title', 'year', 'arxivid' ]:
            assert a.data['2019MNRAS.488.1248H'].citation[key] == self.a.data['Hafen2019'].citation[key]

    ########################################################################

    def test_import_bibcodes( self ):

        bibcodes = [
            '2019MNRAS.488.1248H',
            '2020MNRAS.491.6102B',
        ]

        self.a.import_bibcodes( bibcodes )

        # Check that the values exist
        for key in [ 'title', 'year', 'arxivid' ]:
            self.a.data['2020MNRAS.491.6102B'].citation[key]

        # This should already exist, so yeah, make sure it doesn't
        assert '2019MNRAS.488.1248H' not in self.a.data.keys()

########################################################################

class TestAtlasData( unittest.TestCase ):

    def setUp( self ):

        # Delete and recreate
        self.empty_dir = './tests/data/empty_atlas'
        try:
            shutil.rmtree( self.empty_dir )
        except FileNotFoundError:
            pass
        os.makedirs( self.empty_dir )
        shutil.copyfile(
            './tests/data/example_atlas/example.bib',
            os.path.join( self.empty_dir, 'example.bib' ),
        )

        self.a = atlas.Atlas( self.empty_dir )

    ########################################################################

    def test_load_data_no_data( self ):
        '''We just don't want it to fail.'''

        self.a.load_data( fp=self.empty_dir )

    ########################################################################

    def test_load_data( self ):

        # Create test data
        d = verdict.Dict( {} )
        for key, item in self.a.data.items():
            d[key] = {}
            d[key]['abstract'] = 'Fake abstract for {}'.format( key )
        d.to_hdf5( os.path.join( self.empty_dir, 'atlas_data.h5' ) )

        self.a.load_data()

        # Test
        for key, item in self.a.data.items():
            assert d[key]['abstract'] == self.a[key].abstract

    ########################################################################

    def test_save_data( self ):

        # Create some fake attributes
        for key, item in self.a.data.items():
            item.test_attr = key

        # Function itself
        self.a.save_data(
            attrs_to_save = [ 'test_attr', ],
        )

        # Load saved data
        d = verdict.Dict.from_hdf5( 
            './tests/data/empty_atlas/atlas_data.h5',
        )
        
        for key, item in self.a.data.items():
            assert d[key]['test_attr'] == self.a[key].test_attr

    ########################################################################

    def test_save_data_ads_abstract( self ):

        # Get the data
        self.a.data.process_abstract()

        # Function itself
        self.a.save_data()

        # Load saved data
        d = verdict.Dict.from_hdf5( 
            './tests/data/empty_atlas/atlas_data.h5',
        )
                
        for key, item in self.a.data.items():
            abstract = item.abstract['nltk']
            for ikey, iitem in abstract.items():
                for i, v_i in enumerate( iitem ):
                    for j, v_j in enumerate( v_i ):
                        for k, v_k in enumerate( v_j ):
                            assert v_k == d[key]['abstract']['nltk'][ikey][i][j][k]

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

        actual = self.a.get_unique_key_concepts()
        try:
            expected = set( [
                'dog',
                'kitti cat',
            ] )
            assert actual == expected
        except AssertionError:
            expected = set( [
                'kittycat',
                'dog',
            ] )
            assert actual == expected

    ########################################################################

    def test_unique_key_concepts_forgotten_space( self ):

        np.random.seed( 1234 )

        self.a._all_key_concepts = [
            'dog',
            'dogs',
            'kittycat',
            'kitty cat',
        ]

        actual = self.a.get_unique_key_concepts()
        try:
            expected = set( [
                'dog',
                'kitti cat',
            ] )
            assert actual == expected
        except AssertionError:
            expected = set( [
                'kittycat',
                'dog',
            ] )
            assert actual == expected

########################################################################

class TestSearchPublicationsKeyConcepts( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_default( self ):

        actual = self.a.concept_search(
            'origins of the CGM',
            return_paragraph = False
        )

        h19_kps = self.a['Hafen2019'].notes['key_points']
        h19a_kps = self.a['Hafen2019a'].notes['key_points']
        expected = {
            'Hafen2019': [ h19_kps[0], ],
            'Hafen2019a': [ h19a_kps[-1], ],
        }

        for key, item in expected.items():
            assert item == actual[key]

########################################################################

class TestConceptProjection( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

        # Ensure we have necessary data available
        self.a['Hafen2019'].abstract['nltk']['primary_stemmed']

        fp = './tests/data/example_atlas/concept_projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

    ########################################################################

    def test_concept_projection( self ):

        # Test
        cp = self.a.concept_projection()

        # The dimensions of the concept projection
        expected_dim = (
            len( self.a.data ),
            len( cp['component_concepts'] )
        )
        assert cp['components'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == cp['projected_publications'][i]

        # Normalized check
        norm = np.sqrt( ( cp['components']**2. ).sum( axis=1 ) )
        npt.assert_allclose(
            cp['components_normed'][0,0], cp['components'][0,0]/norm[0]
        )

########################################################################

class TestComparison( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )
        self.a.import_bibtex( './tests/data/example_atlas/example.bib' )
        self.a.data.process_bibtex_annotations( word_per_concept=True )
        self.a.data.identify_unique_key_concepts()

    ########################################################################

    def test_inner_product_atlas_atlas( self ):

        np.random.seed( 1234 )

        # When optimized this shouldn't cost extra to call the line below
        cp = self.a.concept_projection()

        expected = ( cp['components']**2. ).sum()

        actual = self.a.inner_product( 'atlas', 'atlas' )

        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication_atlas( self ):

        np.random.seed( 1234 )

        # When optimized this shouldn't cost extra to call the line below
        cp = self.a.concept_projection()

        expected = ( cp['components'] * cp['components'][8,:] ).sum()

        actual = self.a.inner_product(
            'Hafen2019',
            'atlas',
        )

        npt.assert_allclose( actual, expected, rtol=0.05 )

    ########################################################################

    def test_inner_product_custom_self( self ):

        np.random.seed( 1234 )

        w_aa = self.a.inner_product_custom( self.a, )

        npt.assert_allclose( w_aa, 4266, rtol=0.05 )

    ########################################################################

    def test_inner_product_custom_publication( self ):

        np.random.seed( 1234 )

        w_pa = self.a.inner_product_custom(
            self.a.data['Hafen2019'],
        )

        npt.assert_allclose( w_pa, 529, rtol=0.05 )

