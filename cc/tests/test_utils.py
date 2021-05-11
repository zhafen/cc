from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.utils as utils

########################################################################

class TestCitationToADS( unittest.TestCase ):

    def setUp( self ):

        self.citation = {
            'ENTRYTYPE': 'article',
            'ID': 'VandeVoort2012a',
            'abstract': 'We study the propert... galaxies.',
            'author': 'Van de Voort, Freeke...haye, Joop',
            'date': '2012',
            'doi': '10.1111/j.1365-2966.2012.20949.x',
            'eprint': '1111.5039v1',
            'eprinttype': 'arxiv',
            'issn': '00358711',
            'journaltitle': 'Monthly Notices of t...al Society',
            'keywords': 'Absorption, Cosmolog...um, {OWLS}',
            'pages': '2991--3010',
            'title': 'Properties of gas in...axy haloes',
            'volume': '423'
        }

    ########################################################################

    def test_basic( self ):

        q = utils.citation_to_ads_call( self.citation )

        pubs = utils.ads_query( q )
        assert len( pubs ) == 1
        p = pubs[0]

        assert '10.1111/j.1365-2966.2012.20949.x' in p.identifier

    ########################################################################

    def test_arxiv( self ):

        del self.citation['doi']
        self.citation['eprint'] = '1111.5039'

        q = utils.citation_to_ads_call( self.citation )

        pubs = utils.ads_query( q )
        assert len( pubs ) == 1
        p = pubs[0]

        assert '10.1111/j.1365-2966.2012.20949.x' in p.identifier

    ########################################################################

    def test_arxiv_vnonzero( self ):

        del self.citation['doi']

        q = utils.citation_to_ads_call( self.citation )

        pubs = utils.ads_query( q )
        assert len( pubs ) == 1
        p = pubs[0]

        assert '10.1111/j.1365-2966.2012.20949.x' in p.identifier