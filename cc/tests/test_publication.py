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

# class TestPublicationAnalysis( unittest.TestCase ):
# 
#     def test_generate_abstract_points( self ):
# 
#         assert False
