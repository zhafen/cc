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

class TestComparison( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )
        self.a.import_bibtex( './tests/data/example_atlas/example.bib' )
        self.a.data.process_bibtex_annotations( word_per_concept=True )
        self.a.data.identify_unique_key_concepts()

    ########################################################################

    def test_inner_product_self( self ):

        np.random.seed( 1234 )

        w_aa = self.a.inner_product( self.a, )

        npt.assert_allclose( w_aa, 4266, rtol=0.05 )

    ########################################################################

    def test_inner_product_publication( self ):

        np.random.seed( 1234 )

        w_pa = self.a.inner_product(
            self.a.data['Hafen2019'],
        )

        npt.assert_allclose( w_pa, 529, rtol=0.05 )

