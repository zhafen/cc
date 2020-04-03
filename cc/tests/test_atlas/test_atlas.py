from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.atlas.atlas as atlas

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestRetrieveData( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_bibtex( self ):

        bibtex_fp = './tests/data/example.bib'

        self.a.import_bibtex( bibtex_fp )

        assert self.a.data['Hafen2019'].citation['eprint'] == '1811.11753'

