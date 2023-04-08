from mock import patch
import numpy as np
import numpy.testing as npt
import unittest

import cc.utils as utils

########################################################################

class TestCitationToADS( unittest.TestCase ):
    ## API_extension::process_data

    def setUp( self ):

        self.citation = {
            'ENTRYTYPE': 'article',
            'ID': 'VandeVoort2012a',
            'abstract': 'We study the propert... galaxies.',
            'author': 'Van de Voort, Freeke and Schaye, Joop',
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

        q, ident, id = utils.citation_to_api_call( self.citation )

        pubs = utils.api_query( q )
        assert len( pubs ) == 1
        p = pubs[0]

        assert '10.1111/j.1365-2966.2012.20949.x' in p.identifier

    ########################################################################

    def test_arxiv( self ):

        del self.citation['doi']
        self.citation['eprint'] = '1111.5039'

        q, ident, id = utils.citation_to_api_call( self.citation )

        pubs = utils.api_query( q )
        assert len( pubs ) == 1
        p = pubs[0]

        assert '10.1111/j.1365-2966.2012.20949.x' in p.identifier

    ########################################################################

    def test_arxiv_vnonzero( self ):

        del self.citation['doi']

        q, ident, id = utils.citation_to_api_call( self.citation )

        pubs = utils.api_query( q )
        assert len( pubs ) == 1
        p = pubs[0]

        assert '10.1111/j.1365-2966.2012.20949.x' in p.identifier

    ########################################################################

    def test_noid( self ):

        del self.citation['doi']
        del self.citation['eprint']
        del self.citation['eprinttype']

        q, ident, id = utils.citation_to_api_call( self.citation )

        pubs = utils.api_query( q )
        assert len( pubs ) == 1
        p = pubs[0]

        assert '10.1111/j.1365-2966.2012.20949.x' in p.identifier