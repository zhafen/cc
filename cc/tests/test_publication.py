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

        p.get_ads_data( arxiv='1811.11753' )

        assert p.ads_data.title == ['The origins of the circumgalactic medium in the FIRE simulations']

    ########################################################################

    def test_read_citation( self ):

        # Load
        p = cc.publication.Publication( 'Hafen2019' )

        bibtex_fp = './tests/data/example.bib'

        p.read_citation( bibtex_fp )

        assert p.citation['eprint'] == '1811.11753'

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
        bibtex_fp = './tests/data/example.bib'
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
            'particle-tracking',
            'FIRE-2 simulations',
            'origins of the CGM',
            'CGM',
            'IGM accretion',
            'galactic wind',
            'satellite wind',
            'stellar feedback',
            'L* halos',
            'baryon budget',
            'metal budget',
            'lifetime',
            'hot halo',
            'low-redshift',
            'cool CGM gas',
            'galaxy plane',
            'halo-to-halo variability',
            'metallicity',
            'winds',
            'subgrid metal diffusion',
        ]
        assert sorted( expected_key_concepts ) == p.notes['key_concepts']

        assert p.notes['read'] == 'author'

        assert p.notes['uncategorized'] == [
            r"Test junk I'm leaving here....",
        ]

    ########################################################################

    def test_process_annotation_default( self ):

        # Setup
        p = cc.publication.Publication( 'Hafen2019' )
        point = r'Uses a [particle-tracking] analysis applied to the [FIRE-2 simulations] to study the [origins of the [CGM]], including [IGM accretion], [galactic wind], and [satellite wind] ([extra [brackets [here] for testing]]).'

        # Run
        actual = p.process_annotation( point )
        expected = {
            'key_points': [ point, ],
            'key_concepts': [
                'particle-tracking',
                'FIRE-2 simulations',
                'origins of the CGM',
                'CGM',
                'IGM accretion',
                'galactic wind',
                'satellite wind',
            ],
        }

        # Check
        for key, item in expected.items():
            assert actual[key] == item
