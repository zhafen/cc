import copy
import h5py
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pytest
import scipy.sparse as ss
import shutil
import unittest
import verdict
import warnings

import cc.atlas as atlas
import cc.cartography as cartography
import cc.api as api

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

########################################################################


# setup

atlas_dir = 'tests/data/example_atlas'
bibtex_fp = os.path.join(atlas_dir, api.DEFAULT_BIB_NAME)

fp = 'tests/data/example_atlas/projection.h5'
c = cartography.Cartographer.from_hdf5( fp )
a = atlas.Atlas( 'tests/data/example_atlas', atlas_data_format='hdf5' )

ax, ( coords, inds, pairs ) = c.plot_map()
