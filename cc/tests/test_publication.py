from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.publication

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestRetrieveMetadata( unittest.TestCase ):

    def test_get_ads_data( self ):

        # Load
        p = cc.publication.Publication( 'Hafen2019' )

        p.get_ads_data( arxiv='1811.11753', fl=[ 'title', ] )

        assert p.ads_data['title'] == ['The origins of the circumgalactic medium in the FIRE simulations']

    ########################################################################

    def test_read_citation( self ):

        # Load
        p = cc.publication.Publication( 'Hafen2019' )

        bibtex_fp = './tests/data/example_atlas/example.bib'

        p.read_citation( bibtex_fp )

        assert p.citation['eprint'] == '1811.11753'

    ########################################################################

    def test_citations_per_year( self ):

        # Load
        p = cc.publication.Publication( 'Hafen2019' )
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p.read_citation( bibtex_fp )
        p.get_ads_data( arxiv='1811.11753', fl=[ 'citation', 'entry_date'] )
        p.citations = p.ads_data['citation']
        p.entry_date = p.ads_data['entry_date']

        assert p.citations_per_year() > 0.

########################################################################

class TestTex( unittest.TestCase ):

    def test_load_full_tex( self ):

        # Load
        p = cc.publication.Publication( 'Hafen2019' )

        p.load_full_tex( filepath )

        assert p.full_text[0] == '% mnras_template.tex \n'

########################################################################

class TestPublicationAnalysis( unittest.TestCase ):

    def test_process_bibtex_annotations( self ):

        # Run the function
        p = cc.publication.Publication( 'Hafen2019' )
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p.process_bibtex_annotations( bibtex_fp )

        # Check that we got the key points right
        expected_key_points = [
            r'Uses a [particle-tracking] analysis applied to the [FIRE-2 simulations] to study the [origins of the [CGM]], including [IGM accretion], [galactic wind], and [satellite wind].',
            r'Strong [stellar feedback] means only [L* halos] retain {\textgreater}{\~{}}50{\%} of their [baryon budget].',
            r'{\textgreater}{\~{}}60{\%} of the [metal budget] is retained by halos.',
            r'{\textgreater}{\~{}}60{\%} of the [CGM] originates as [IGM accretion].',
            r'{\~{}}20-40{\%} of the [CGM] originates as [galactic wind].',
            r'In [L* halos] {\~{}}1/4 of the [CGM] originates as [satellite wind].',
            r'The [lifetime] for gas in the [CGM] is billions of years, during which time it forms a well-mixed [hot halo].',
            r'For [low-redshift] [L* halos] [cool CGM gas] (T {\textless} 1e4.7 K) is distributed on average preferentially along the [galaxy plane], but with strong [halo-to-halo variability].',
            r'The [metallicity] of [IGM accretion] is systematically lower than the [metallicity] of [winds] (typically by {\textgreater}{\~{}}1 dex), but metallicities depend significantly on the treatment of [subgrid metal diffusion].',
        ]
        assert expected_key_points == p.notes['key_points']

        # Check that we got the key concepts right
        expected_key_concepts = [
            [
                'particle-tracking',
                'FIRE-2 simulations',
                'origins of the CGM',
                'CGM',
                'IGM accretion',
                'galactic wind',
                'satellite wind',
            ],
            [
                'stellar feedback',
                'L* halos',
                'baryon budget',
            ],
            [
                'metal budget',
            ],
            [
                'CGM',
                'IGM accretion',
            ],
            [
                'CGM',
                'galactic wind',
            ],
            [
                'L* halos',
                'CGM',
                'satellite wind',
            ],
            [
                'lifetime',
                'CGM',
                'hot halo',
            ],
            [
                'low-redshift',
                'cool CGM gas',
                'galaxy plane',
                'halo-to-halo variability',
                'L* halos',
            ],
            [
                'metallicity',
                'IGM accretion',
                'winds',
                'subgrid metal diffusion',
            ],
        ]
        for i, e_kc in enumerate( expected_key_concepts ):
            assert sorted( e_kc ) == sorted( p.notes['key_concepts'][i] )

        assert p.notes['read'] == 'author'

        assert p.notes['uncategorized'] == [
            r"Test junk I'm leaving here....",
        ]

    ########################################################################

    def test_process_bibtex_annotations_cached( self ):

        # Run the function
        p = cc.publication.Publication( 'Hafen2019' )
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p.process_bibtex_annotations( bibtex_fp )
        p.process_bibtex_annotations( bibtex_fp )

        assert len( p.notes['key_points'] ) == 9

        assert p.notes['uncategorized'] == [
            r"Test junk I'm leaving here....",
        ]

    ########################################################################

    def test_process_annotation_default( self ):

        # Setup
        p = cc.publication.Publication( 'Hafen2019' )
        point = r'Uses a [particle-tracking] analysis applied to the [FIRE-2 simulations] to study the [origins of the [CGM]], including [IGM accretion], [galactic wind], and [satellite wind] ([extra [brackets [here] for testing]]).'

        # Run
        actual = p.process_annotation_line( point )
        expected = {
            'key_points': [ point, ],
            'key_concepts': [ [
                'particle-tracking',
                'FIRE-2 simulations',
                'CGM',
                'origins of the CGM',
                'IGM accretion',
                'galactic wind',
                'satellite wind',
                'here',
                'brackets here for testing',
                'extra brackets here for testing',
            ], ],
        }

        # Check
        assert sorted( actual['key_points'] ) == sorted( expected['key_points'] )
        assert sorted( actual['key_concepts'][0] ) == sorted( expected['key_concepts'][0] )

    ########################################################################

    def test_process_abstract( self ):

        p = cc.publication.Publication( 'Hafen2019' )
        p.process_bibtex_annotations( './tests/data/example_atlas/example.bib' )

        p.process_abstract()

        assert p.abstract['nltk']['all'][0][3] == ('particle', 'NN')

        assert len( p.abstract['nltk']['uncategorized'] ) == 0

        assert 'separ' in p.abstract['nltk']['primary_stemmed'][0]

    ########################################################################

    def test_process_abstract_no_arxivid( self ):

        p = cc.publication.Publication( 'Hafen2019' )
        p.process_bibtex_annotations( './tests/data/example_atlas/example.bib' )

        del p.citation['arxivid']

        p.process_abstract()

        assert p.abstract['nltk']['all'][0][3] == ('particle', 'NN')

        assert len( p.abstract['nltk']['uncategorized'] ) == 0

        assert 'separ' in p.abstract['nltk']['primary_stemmed'][0]

    ########################################################################

    def test_process_abstract_no_arxivid_no_doi( self ):

        p = cc.publication.Publication( 'Hafen2019' )
        p.process_bibtex_annotations( './tests/data/example_atlas/example.bib' )

        del p.citation['arxivid']
        del p.citation['doi']

        p.process_abstract()

        assert p.abstract['nltk']['all'] == []

    ########################################################################

    def test_process_abstract_bad_arxivid( self ):

        p = cc.publication.Publication( 'Hafen2019' )
        p.process_bibtex_annotations( './tests/data/example_atlas/example.bib' )

        p.citation['arxivid'] = 'bad'

        p.process_abstract()

        assert p.abstract['nltk']['all'][0][3] == ('particle', 'NN')

        assert len( p.abstract['nltk']['uncategorized'] ) == 0

        assert 'separ' in p.abstract['nltk']['primary_stemmed'][0]

    ########################################################################

    def test_process_abstract_bad_arxivid_bad_doi( self ):

        p = cc.publication.Publication( 'Hafen2019' )
        p.process_bibtex_annotations( './tests/data/example_atlas/example.bib' )

        p.citation['arxivid'] = 'bad'
        p.citation['doi'] = 'bad'

        p.process_abstract()

        assert p.abstract['nltk']['all'] == []

########################################################################

class ComponentProjection( unittest.TestCase ):

    def setUp( self ):

        self.p = cc.publication.Publication( 'Hafen2019' )
        bibtex_fp = './tests/data/example_atlas/example.bib'
        self.p.process_bibtex_annotations( bibtex_fp )

    ########################################################################

    def test_concept_projection( self ):

        values, comp_concepts = self.p.concept_projection()

        # Should be no 0s
        assert values.min() > 0

        # Two spot checks
        assert 'wind' in comp_concepts
        assert 'cohes' in comp_concepts

    ########################################################################

    def test_concept_projection_existing_vector( self ):

        comp_concepts = [ 'accret', 'dog' ]
        values, comp_concepts = self.p.concept_projection(
            comp_concepts
        )

        # Should be one 0
        assert values[-1] == 0
        assert comp_concepts[-1] == 'dog'

        # Spot checks
        assert 'wind' in comp_concepts
        assert 'cohes' in comp_concepts
        assert 'accret' in comp_concepts
        assert 'dog' in comp_concepts

########################################################################

class TestComparison( unittest.TestCase ):

    def test_inner_product_custom_self( self ):

        # Load test data
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p1 = cc.publication.Publication( 'Hafen2019' )
        p2 = cc.publication.Publication( 'Hafen2019a' )
        p1.process_bibtex_annotations( bibtex_fp )
        p2.process_bibtex_annotations( bibtex_fp )

        # Calculate inner products
        w_11 = p1.inner_product_custom( p1, method='key-point concepts' )
        w_12 = p1.inner_product_custom( p2, method='key-point concepts' )
        w_21 = p2.inner_product_custom( p1, method='key-point concepts' )

        # Check expected relations
        assert w_12 == w_21
        assert w_11 > w_12

    ########################################################################

    def test_consistency( self ):

        # Load test data
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p = cc.publication.Publication( 'Hafen2019' )
        p.process_bibtex_annotations( bibtex_fp, word_per_concept=True )
        p.identify_unique_key_concepts()

        # Check expected relations
        w1 = p.inner_product_custom( p, method='key-point concepts' )
        w2 = p.inner_product_custom( p, method='cached key-point concepts' )

        npt.assert_allclose( w1, w2, rtol=0.1 )
        
    ########################################################################

    def test_abstract_similarity( self ):

        # Load test data
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p1 = cc.publication.Publication( 'Hafen2019' )
        p2 = cc.publication.Publication( 'Hafen2019a' )
        p1.process_bibtex_annotations( bibtex_fp )
        p2.process_bibtex_annotations( bibtex_fp )

        # Calculate inner products
        w_11 = p1.inner_product_custom( p1, method='abstract similarity' )
        w_12 = p1.inner_product_custom( p2, method='abstract similarity' )
        w_21 = p2.inner_product_custom( p1, method='abstract similarity' )

        # Check expected relations
        assert w_12 == w_21
        assert w_11 > w_12

    ########################################################################

    def test_inner_product_custom_self_edit_distance( self ):

        # Load test data
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p1 = cc.publication.Publication( 'Hafen2019' )
        p2 = cc.publication.Publication( 'Hafen2019a' )
        p1.process_bibtex_annotations( bibtex_fp )
        p2.process_bibtex_annotations( bibtex_fp )

        # Calculate inner products
        kwargs = {
            'method': 'key-point concepts',
            'max_edit_distance': 2,
        }
        w_11 = p1.inner_product_custom( p1, **kwargs )
        w_12 = p1.inner_product_custom( p2, **kwargs )
        w_21 = p2.inner_product_custom( p1, **kwargs )

        # Check expected relations
        assert w_12 == w_21
        assert w_11 > w_12

########################################################################

class TestCitationEstimator( unittest.TestCase ):

    def setUp( self ):

        bibtex_fp = './tests/data/example_atlas/example.bib'
        self.p = cc.publication.Publication( 'Hafen2019' )
        self.p.process_bibtex_annotations( bibtex_fp )
