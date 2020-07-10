from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import cc.relation as relation

########################################################################

class TestFunctions( unittest.TestCase ):

    def test_point_inner_product( self ):

        a = '[Dogs] and [cats] are [animals].'
        b = '[Cats] and [dogs] are not [people].'

        actual = relation.inner_product( a, b )
        expected = 2

        assert actual == expected

    ########################################################################

