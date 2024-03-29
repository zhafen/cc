from mock import patch
import nltk
import numpy as np
import numpy.testing as npt
import pytest
import unittest

import cc.publication

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestRetrieveMetadata( unittest.TestCase ):
    ## API_extension::get_data_via_api

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

@pytest.mark.onhold
class TestTex( unittest.TestCase ):

    def test_load_full_tex( self ):

        # Load
        p = cc.publication.Publication( 'Hafen2019' )

        p.load_full_tex( filepath )

        assert p.full_tex.string[:22] == '% mnras_template.tex \n'

        # Make sure \include is handled correctly
        assert p.full_tex.string.splitlines()[121] == r'\newcommand{\evolutionMmaxsubMWq}{5\times10^{9}}'

        # Check that the abstract was loaded and parsed correctly.
        assert p.tex['Abstract'].string[:8] == '\n\nWe use'
        assert p.tex['Abstract'].string[-10:] == 'picture.\n\n'

        # Check that the sections were loaded and parsed correctly
        expected = [
            'Abstract',
            'Introduction',
            'Methods',
            'Results',
            'Discussion',
            'Conclusions',
            'Appendix',
        ]
        actual = list( p.tex.keys() )
        assert actual == expected

        # Check that we have the appendix correct.
        expected = [ 'Phases of CGM Gas', 'Supplementary Material' ]
        actual = list( p.tex['Appendix'].keys() )
        assert actual == expected

########################################################################

class TestPublicationAnalysis( unittest.TestCase ):

    def test_process_bibtex_annotations( self ):

        # Run the function
        p = cc.publication.Publication( 'Hafen2019' )
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p.process_bibtex_annotations( bibtex_fp, word_per_concept=False )

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
            r"This word should show up in the vectorized text: garglflinx.",
            r"Test junk I'm leaving here....",
        ]

        # Check we got the points right, which is an amalgation of the
        # abstract, the key points, and the uncategorized notes
        expected_points = [
            r'Uses a [particle-tracking] analysis applied to the [FIRE-2 simulations] to study the [origins of the [CGM]], including [IGM accretion], [galactic wind], and [satellite wind].',
            r'Strong [stellar feedback] means only [L* halos] retain {\textgreater}{\~{}}50{\%} of their [baryon budget].',
            r'{\textgreater}{\~{}}60{\%} of the [metal budget] is retained by halos.',
            r'{\textgreater}{\~{}}60{\%} of the [CGM] originates as [IGM accretion].',
            r'{\~{}}20-40{\%} of the [CGM] originates as [galactic wind].',
            r'In [L* halos] {\~{}}1/4 of the [CGM] originates as [satellite wind].',
            r'The [lifetime] for gas in the [CGM] is billions of years, during which time it forms a well-mixed [hot halo].',
            r'For [low-redshift] [L* halos] [cool CGM gas] (T {\textless} 1e4.7 K) is distributed on average preferentially along the [galaxy plane], but with strong [halo-to-halo variability].',
            r'The [metallicity] of [IGM accretion] is systematically lower than the [metallicity] of [winds] (typically by {\textgreater}{\~{}}1 dex), but metallicities depend significantly on the treatment of [subgrid metal diffusion].',
            r"This word should show up in the vectorized text: garglflinx.",
            r"Test junk I'm leaving here....",
        ]
        expected_points += nltk.sent_tokenize( p.citation['abstract'] )
        assert expected_points == p.points()

    ########################################################################

    def test_process_bibtex_annotations_cached( self ):

        # Run the function
        p = cc.publication.Publication( 'Hafen2019' )
        bibtex_fp = './tests/data/example_atlas/example.bib'
        p.process_bibtex_annotations( bibtex_fp )
        p.process_bibtex_annotations( bibtex_fp )

        assert len( p.notes['key_points'] ) == 9

        assert p.notes['uncategorized'] == [
            r"This word should show up in the vectorized text: garglflinx.",
            r"Test junk I'm leaving here....",
        ]

    ########################################################################

    def test_process_annotation_default( self ):

        # Setup
        p = cc.publication.Publication( 'Hafen2019' )
        point = r'Uses a [particle-tracking] analysis applied to the [FIRE-2 simulations] to study the [origins of the [CGM]], including [IGM accretion], [galactic wind], and [satellite wind] ([extra [brackets [here] for testing]]).'

        # Run
        actual = p.process_annotation_line( point, word_per_concept=False )
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
        del p.citation['abstract']

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

        del p.citation['abstract']
        p.citation['arxivid'] = 'bad'
        p.citation['doi'] = 'bad'

        p.process_abstract()

        assert p.abstract['nltk']['all'] == []

########################################################################

class Vectorize( unittest.TestCase ):

    def setUp( self ):

        self.p = cc.publication.Publication( 'Hafen2019' )
        bibtex_fp = './tests/data/example_atlas/example.bib'
        self.p.process_bibtex_annotations( bibtex_fp )

    ########################################################################

    def test_vectorize( self ):

        values, feature_names = self.p.vectorize()

        # Should be no 0s
        assert values.min() > 0

        # Two spot checks
        assert 'wind' in feature_names
        assert 'cohes' in feature_names
        assert 'garglflinx' in feature_names

    ########################################################################

    def test_vectorize_existing_vector( self ):

        feature_names_orig = [ 'accret', 'dog' ]
        values, feature_names = self.p.vectorize(
            feature_names_orig
        )

        # Should be one 0
        assert values[1] == 0

        # Should match with the formatting of the original vector
        for i, feature_name in enumerate( feature_names_orig ):
            assert feature_name == feature_names[i]

        assert len( values ) == len( feature_names )

        # Spot checks
        assert 'wind' in feature_names
        assert 'cohes' in feature_names
        assert 'accret' in feature_names
        assert 'dog' in feature_names

    ########################################################################

    def test_vectorize_notes_not_included( self ):

        values, feature_names = self.p.vectorize( include_notes=False )

        # Should be no 0s
        assert values.min() > 0

        # Two spot checks
        assert 'wind' in feature_names
        assert 'cohes' in feature_names
        assert 'garglflinx' not in feature_names

    ########################################################################
    
    def test_vectorize_no_viable_words( self ):

        # Modify to provide one w/o nouns
        self.p.citation['abstract'] = '(Without )'

        feature_names_orig = [ 'accret', 'dog' ]
        values, feature_names = self.p.vectorize(
            feature_names_orig,
            include_notes = False
        )

        # Should be all zeros
        npt.assert_allclose( values, np.array([0., 0.]) )

        # Should match with the formatting of the original vector
        for i, feature_name in enumerate( feature_names ):
            assert feature_name == feature_names_orig[i]

        assert len( values ) == len( feature_names )

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
