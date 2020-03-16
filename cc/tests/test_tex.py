from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.tex

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestTex( unittest.TestCase ):

    def test_init( self ):

        # Load
        p = cc.tex.Paper( filepath )

        assert p.full_text[0] == r'% mnras_template.tex \n'
