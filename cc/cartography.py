import copy
import ctypes
import glob
import inspect
import numba
import numpy as np
import os
import pandas as pd
import scipy.sparse as ss
import scipy.spatial
from scipy.spatial.distance import cdist
import sklearn.feature_extraction.text as skl_text_features
import warnings

# Import tqdm but change default options
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, ncols=79)

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly
import plotly.graph_objects as go
import palettable

import augment
import verdict

from . import atlas
from . import utils
from . import api
from . import publication

########################################################################

class Cartographer( object ):
    '''Class for analyzing and exploring projected data.

        Args:
            backend (str):
                What code to use for calculations?
                Options are 'python', 'c/c++'.
    '''

    def __init__( self, backend='c/c++', transform=None, **kwargs ):

        self.backend = backend

        if self.backend == 'c/c++':
            ## Get the c executable
            cc_dir = os.path.dirname( os.path.dirname( __file__ ) )
            lib_glob = os.path.join( cc_dir, 'build', '*/cartography*.so' )
            lib_fp = glob.glob( lib_glob )[0]
            self.c_cartography = ctypes.CDLL( lib_fp )

        self.update_data( **kwargs )

        if transform is not None:
            self.apply_transform( transform )

    @augment.store_parameters
    def update_data(
        self,
        vectors,
        norms,
        feature_names,
        publications,
        publication_dates,
        entry_dates,
        prune_zeros = True,
        prune_duplicates = True,
    ):
        '''Update the data used for calculations.
        '''

        # Convert date to a more useable array
        self.publication_dates = pd.to_datetime( publication_dates, errors='coerce' )
        self.entry_dates = pd.to_datetime( entry_dates, errors='coerce' )

        # Ensure that vectors has, if compressed, sorted indices
        if ss.issparse( vectors ):
            if not vectors.has_sorted_indices:
                warnings.warn( 'vectors are compressed but unsorted. Sorting...' )
                vectors.sort_indices()

        if prune_zeros:
            self.prune_zero_entries()

        if prune_duplicates:
            self.prune_duplicate_entries()

    ########################################################################
    # Core methods
    ########################################################################

    @property
    def inds( self ):

        if not hasattr( self, '_inds' ):

            self._inds = np.arange( self.publications.size )

        return self._inds

    ########################################################################

    @property
    def vectors_notsp( self ):

        if not hasattr( self, '_vectors_notsp' ):

            try:
                self._vectors_notsp = self.vectors.toarray()
            except AttributeError:
                self._vectors_notsp = self.vectors

        return self._vectors_notsp

    ########################################################################

    @property
    def vectors_notsp_normed( self ):
        '''Components normalized such that <P|P>=1 .
        '''

        if not hasattr( self, '_vectors_notsp_normed' ):

            # Divide by NaN is unimportant and handled
            with np.errstate(divide='ignore',invalid='ignore'):

                self._vectors_notsp_normed = self.vectors_notsp / self.norms[:,np.newaxis]

        return self._vectors_notsp_normed

    ########################################################################

    def prune_zero_entries( self ):
        '''Toss out any entries which have no components,
        usually due to no abstract.
        '''

        is_nonzero = np.invert( np.isclose( self.norms, 0. ) )
        valid_inds = np.arange( self.publications.size )[is_nonzero]
        self.prune( valid_inds )

    ########################################################################

    def prune_duplicate_entries( self ):
        '''Toss out duplicate entries.'''

        unique_inds = np.unique( self.vectors_notsp, axis=0, return_index=True )[1]
        valid_inds = np.sort( unique_inds )
        self.prune( valid_inds )

    ########################################################################

    def prune( self, valid_inds ):
        '''Toss out entries not in valid_inds.'''

        for attr in [ 'vectors', 'norms', 'publications', 'publication_dates', 'entry_dates' ]:
            value = getattr( self, attr )[valid_inds]
            setattr( self, attr, value )

        # Clear out calculated properties to start fresh
        for attr in dir( self ):
            if isinstance( getattr( type( self ), attr, None), property):
                stored_attr = '_' + attr
                if hasattr( self, stored_attr ):
                    delattr( self, stored_attr )

    ########################################################################

    def get_age( self, date_type ):

        time_elapsed = (                                                        
            pd.to_datetime( 'now', ) -                                          
            getattr( self, date_type).tz_localize(None)              
        )                                                                       
        time_elapsed_years = time_elapsed.total_seconds() / 3.154e7 

        return time_elapsed_years

    @property
    def age_years( self ):

        if not hasattr( self, '_age_years' ):

            self._age_years = self.get_age( 'entry_dates' )

        return self._age_years

    @property
    def publication_age_years( self ):

        if not hasattr( self, '_publication_age_years' ):

            self._publication_age_years = self.get_age( 'publication_dates' )

        return self._publication_age_years

    ########################################################################

    @classmethod
    def from_hdf5( self, fp, sparse=True, backend='c/c++', transform=None ):
        '''Load the cartographer from a saved file.

        Args:
            fp (str):
                Filepath to the projected data.

            sparse (int):
                Whether or not the components are saved as a sparse matrix.
                Will convert to a sparse matrix after loading.

            backend (str):
                What code to use for calculations?
                Options are 'python', 'c/c++'.

        Returns:
            Cartographer instance
        '''

        data = verdict.Dict.from_hdf5( fp, sparse=sparse )

        # If old format compatbility
        if 'components' in data.keys():
            data['vectors'] = copy.copy( data['components'] )
            del data['components']
        if 'component_concepts' in data.keys():
            data['feature_names'] = copy.copy( data['component_concepts'] )
            del data['component_concepts']

        # Convert
        if not sparse:
            vectors_notsp = copy.copy( data['vectors'] )
            data['vectors'] = ss.csr_matrix( data['vectors'] )

        c = Cartographer( backend=backend, transform=transform, **data )

        if not sparse:
            c._vectors_notsp = vectors_notsp

        return c

    ########################################################################
    # Core data manipulation
    ########################################################################

    def apply_transform( self, transform='tf-idf', **kwargs ):
        '''Apply a transformation to the vectors.
        This is useful for downweighting common words, for example.

        Args:
            transform:
                The transform to apply. Currently only a tf-idf transform is available.

        Keyword Args:
            Extra arguments passed to the sklearn transformer object.

        Modifies:
            self.vector
            self.norms
        '''

        if transform == 'tf-idf':

            # Apply to vectors
            transformer = skl_text_features.TfidfTransformer( norm=None, **kwargs )
            self.vectors = transformer.fit_transform( self.vectors )

            # Recalc normalization
            norm_squared_unformatted = self.vectors.multiply( self.vectors ).sum( axis=1 )
            self.norms = np.sqrt( np.array( norm_squared_unformatted ).flatten() )
        else:
            raise NameError( 'Unrecognized transformation, transform={}'.format( transform ) )

    ########################################################################
    # Basic Analysis
    ########################################################################

    def inner_product( self, key_a, key_b, backend=None, **kwargs ):
        '''Calculate the inner product between a and b, using the
        pre-generated vector projection.

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    'atlas':
                        Inner product with the full atlas.
                    'all':
                        Array of inner product with each publication.
                    key from self.data:
                        Inner product for a particular publication.

            key_b (str):
                Reference to the second object, same options as key_a.

            backend (str):
                What code to use to calculate the inner product?
                If None then falls back to self.backend,
                set during initialization.

        Keyword Args:
            Passed to self.vectorize

        Returns:
            The inner product of a and b
        '''

        # Swap a and b since it doesn't matter anyways when one is all
        if key_b != 'all' and key_a == 'all':
            key_a, key_b = key_b, key_a

        # When a==b we can use the norms
        if key_a == key_b:
            if key_a == 'atlas':
                return ( self.vectors_notsp.sum( axis=0 )**2. ).sum()
            elif key_a == 'all':
                return self.norms**2.
            else:
                is_p = self.publications == key_a
                return self.norms[is_p]**2.

        if backend is None:
            backend = self.backend

        if backend == 'python':

            # Find the objects the keys refer to
            def interpret_key( key ):
                # A single publication
                if key in self.publications:
                    is_p = self.publications == key
                    return self.vectors_notsp[is_p][0]
                # The entire atlas
                elif key == 'atlas' or key == 'all':
                    return self.vectors_notsp
                else:
                    raise KeyError( 'Unhandled key, {}'.format( key ) )

            a = interpret_key( key_a )
            b = interpret_key( key_b )

            # When we're doing the inner product with the atlas for all pubs
            if sorted([ key_a, key_b ]) == [ 'all', 'atlas' ]:
                result = np.array([ np.dot( a, row ).sum() for row in b ])
                return result

            # Dot product
            try:
                try:
                    result = np.dot( a, b )
                except TypeError:
                    # DEBUG
                    import pdb; pdb.set_trace()
            except ValueError:
                result = np.dot( b, a )

            # Finish dot product
            if key_a == 'atlas' or key_b == 'atlas':
                result = result.sum()
        elif backend == 'c/c++':

            # Publication-publication case
            if key_a != 'all' and key_b != 'all':

                # Get the sparse rows
                i_a = np.argmax( self.publications == key_a )
                i_b = np.argmax( self.publications == key_b )
                slice_a = slice(self.vectors.indptr[i_a], self.vectors.indptr[i_a+1])
                slice_b = slice(self.vectors.indptr[i_b], self.vectors.indptr[i_b+1])
                data_a = self.vectors.data[slice_a]
                data_b = self.vectors.data[slice_b]
                indices_a = self.vectors.indices[slice_a]
                indices_b = self.vectors.indices[slice_b]

                # Setup types
                self.c_cartography.inner_product_sparse.restype = ctypes.c_int
                self.c_cartography.inner_product_sparse.argtypes = [
                    np.ctypeslib.ndpointer( dtype=np.int64 ),
                    np.ctypeslib.ndpointer( dtype=np.int64 ),
                    ctypes.c_int,
                    np.ctypeslib.ndpointer( dtype=np.int64 ),
                    np.ctypeslib.ndpointer( dtype=np.int64 ),
                    ctypes.c_int,
                ]

                # Call
                result = self.c_cartography.inner_product_sparse(
                    data_a.astype( 'int64' ),
                    indices_a.astype( 'int64' ),
                    len( data_a ),
                    data_b.astype( 'int64' ),
                    indices_b.astype( 'int64' ),
                    len( data_b ),
                )
             
            elif key_a != 'all' and key_b == 'all':

                # Setup input
                # breakpoint()
                i_a = np.argmax( self.publications == key_a )

                # Setup types
                self.c_cartography.inner_product_row_all_sparse.restype = np.ctypeslib.ndpointer(
                    dtype = np.int64,
                    shape = ( self.publications.size, )
                )
                self.c_cartography.inner_product_row_all_sparse.argtypes = [
                    ctypes.c_int,
                    np.ctypeslib.ndpointer( dtype=np.int64 ),
                    np.ctypeslib.ndpointer( dtype=np.int64 ),
                    np.ctypeslib.ndpointer( dtype=np.int64 ),
                    ctypes.c_int,
                ]

                # Call
                result = self.c_cartography.inner_product_row_all_sparse(
                    i_a,
                    self.vectors.data.astype( 'int64' ),
                    self.vectors.indices.astype( 'int64' ),
                    self.vectors.indptr.astype( 'int64' ),
                    self.publications.size,
                )

        else:
            raise KeyError( 'Unrecognized backend, {}'.format( backend ) )

        return result

    ########################################################################

    @property
    def inner_product_matrix( self ):
        '''Matrix with element [i,j] describing the inner product between
        publications i and j.
        '''

        if not hasattr( self, '_inner_product_matrix' ):

            # Setup types
            n_pubs = self.publications.size
            self.c_cartography.inner_product_matrix.restype = ctypes.POINTER( ctypes.c_long )
            self.c_cartography.inner_product_matrix.argtypes = [
                np.ctypeslib.ndpointer( dtype=np.int64 ),
                np.ctypeslib.ndpointer( dtype=np.int64 ),
                np.ctypeslib.ndpointer( dtype=np.int64 ),
                ctypes.c_int,
            ]

            # Call
            result_pointer = self.c_cartography.inner_product_matrix(
                self.vectors.data.astype( 'int64' ),
                self.vectors.indices.astype( 'int64' ),
                self.vectors.indptr.astype( 'int64' ),
                n_pubs,
            )
            self._inner_product_matrix = np.ctypeslib.as_array( result_pointer, ( n_pubs, n_pubs ) )

        return self._inner_product_matrix

    ########################################################################

    def cospsi( self, key_a, key_b, **kwargs ):
        '''Cosine of the "angle" between two publications, defined as
        <A|B> / sqrt( <A|A><B|B> ).

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    'atlas':
                        Psi with the full atlas.
                    'all':
                        Array of inner product with each publication.
                    key from self.data:
                        Inner product for a particular publication.

            key_b (str):
                Reference to the second object, same options as key_a.

        Keyword Args:
            Passed to self.vectorize

        Returns:
            The cosine of a and b
        '''

        ip_ab = self.inner_product( key_a, key_b, **kwargs )
        ip_aa = self.inner_product( key_a, key_a, **kwargs )
        ip_bb = self.inner_product( key_b, key_b, **kwargs )
        return ip_ab / np.sqrt( ip_aa * ip_bb )

    ########################################################################

    @property
    def cospsi_matrix( self ):
        '''Matrix with element [i,j] describing the cosine between
        publications i and j.
        '''

        if not hasattr( self, '_cospsi_matrix' ):

            norms = np.diagonal( self.inner_product_matrix )
            self._cospsi_matrix = (
                self.inner_product_matrix /
                np.sqrt( norms ) /
                np.sqrt( norms[:,np.newaxis] )
            )

        return self._cospsi_matrix

    ########################################################################

    def psi( self, key_a, key_b, scaling=np.pi/180., **kwargs ):
        '''The "angle" between two publications, defined as
        arccos( <A|B> / sqrt( <A|A><B|B> ) ).

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    'atlas':
                        Psi with the full atlas.
                    'all':
                        Array of inner product with each publication.
                    key from self.data:
                        Inner product for a particular publication.

            key_b (str):
                Reference to the second object, same options as key_a.

            scaling (float):
                The number of radians to scale the output by. Defaults to
                pi/2, the usual maximum difference between two publications.

        Keyword Args:
            Passed to self.vectorize

        Returns:
            psi of a and b
        '''

        cospsi = self.cospsi( key_a, key_b, **kwargs )

        return np.arccos( cospsi ) / scaling

    ########################################################################

    def distance( self, key_a, key_b, normed=True, ):

        # Get the right components
        if normed:
            components = self.vectors_notsp_normed
        else:
            components = self.vectors_notsp

        # For all compared to all
        if key_a == 'all' and key_b == 'all':
            return cdist( components, components )

        # Swap a and b since it doesn't matter anyways
        if key_b != 'all' and key_a == 'all':
            key_a, key_b = key_b, key_a
        
        # Calc distance
        p_a = components[self.publications==key_a][0]
        d = np.linalg.norm( components - p_a, axis=1 )

        if key_b != 'all':
            return d[self.publications==key_b][0]
        else:
            return d

    ########################################################################

    def text_overlap( self, key_a, key_b, norm='a' ):
        '''Calculate the text overlap between a and b, using the
        pre-generated vector projection.

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    'all':
                        Array of text overlap with each publication.

                    key from self.data:
                        Text overlap for a particular publication.

            key_b (str):
                Reference to the second object, same options as key_a.

            norm (str):
                How to normalize the text overlap. Options are...
                    'a':
                        Normalize by the number of words in a.

                    'b':
                        Normalize by the number of words in b.

                    'geometric mean':
                        Normalize by the geometric mean of the
                        number of words in a and b.

        Keyword Args:
            Passed to self.vectorize

        Returns:
            The amount of text overlap between a and b.
        '''

        # Returning all, all is just the pairwise calculation
        if key_a == 'all' and key_b == 'all':
            return self.pairwise(
                'text_overlap',
                trim_and_reshape = False,
                norm = norm,
            )

        # Swap a and b since it doesn't matter anyways
        if key_b != 'all' and key_a == 'all':
            key_a, key_b = key_b, key_a
            if norm == 'a': norm = 'b'
            elif norm == 'b': norm = 'a'

        # Find the objects the keys refer to
        def interpret_key( key ):
            # A single publication
            if key in self.publications:
                is_p = self.publications == key
                return self.vectors_notsp[is_p][0]
            # The entire atlas
            elif key == 'atlas' or key == 'all':
                return self.vectors_notsp
        a = interpret_key( key_a )
        b = interpret_key( key_b )

        # Tile for compatibility
        if key_b != 'all':
            n_shared = np.min( np.array([ a, b ]), axis=0 ).sum()
        else:
            a_tiled = np.broadcast_to( a, b.shape )
            min_arr = np.min( np.array([ a_tiled, b ]), axis=0 )
            n_shared = min_arr.sum( axis=1 )

        # Normalization
        if norm == 'a':
            norm_value = a.sum()
        elif norm == 'b':
            if key_b == 'all':
                norm_value = b.sum( axis=1 )
            else:
                norm_value = b.sum()
        elif norm == 'geometric mean':
            if key_b == 'all':
                norm_value = np.sqrt( a.sum() * b.sum( axis=1 ) )
            else:
                norm_value = np.sqrt( a.sum() * b.sum() )

        return n_shared / norm_value

    def symmetric_text_overlap( self, key_a, key_b ):
        '''Calculate the text overlap between a and b, using the
        pre-generated vector projection. This option uses the
        geometric mean as normalization.

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    'all':
                        Array of text overlap with each publication.

                    key from self.data:
                        Text overlap for a particular publication.

            key_b (str):
                Reference to the second object, same options as key_a.

        Keyword Args:
            Passed to self.vectorize

        Returns:
            The amount of text overlap between a and b.
        '''

        return self.text_overlap( key_a, key_b, norm='geometric mean' )

    ########################################################################

    def pairwise( self, metric, trim_and_reshape=True, *args, **kwargs ):
        '''Calculate the pairwise metric between all publications in the vector projection.

        Args:
            metric (str):
                Name of the metric to calculate. Options include inner_product, psi

        Returns:
            pairwise_values (np.ndarray, ( n_pubs*(n_pubs-1)/2, ) ):
                Pairwise values between publications.
        '''

        # Calculate metric for all
        mat = []
        for key in tqdm( self.publications ):
            mat.append( getattr( self, metric )( key, 'all', *args, **kwargs ) )
        mat = np.array( mat )

        if not trim_and_reshape:
            return mat

        # Identify pairs, remove others, and reshape
        pairwise_values = mat[np.triu_indices( mat.shape[0], k=1 )]

        return pairwise_values

    ########################################################################
    # Automated exploration, expansion, or otherwise updating
    ########################################################################

    def expand( 
        self, 
        a, 
        api_name = api.DEFAULT_API, 
        center=None, 
        n_pubs_max=4000, 
        n_sources_max=None, 
        bibtex_fp = api.DEFAULT_BIB_NAME, 
    ):
        '''Expand an atlas by retrieving all publications cited by the
        the publications in the given atlas, or that reference a
        publication in the given atlas.

        Args:
            a (atlas.Atlas):
                Atlas to expand.

            api (str):
                The API to call to expand the publication region.

            center (str):
                If given, center the search on this publication, preferentially
                searching related publications.

            n_pubs_max (int):
                Maximum number of publications allowed in the expansion.

            n_sources_max (int):
                Maximum number of publications (already in the atlas) to draw references and citations from.

        Returns:
            a_exp (atlas.Atlas):
                Expanded atlas. Has the same save location as a.
        '''

        # Without a center
        if center is None:
            expand_keys = list( a.data.keys() )
        else:
            if center not in self.publications:
                raise Exception(f'Center {center} not in publications. Perhaps center is a different identifier from what is stored in publications.')

            cospsi = self.cospsi( center, 'all' )
            sort_inds = np.argsort( cospsi )[::-1]
            expand_keys = self.publications[sort_inds]

        if n_sources_max is not None:
            expand_keys = expand_keys[:n_sources_max]

        # Main expansion via collection of references and citations
        ids = get_ids_list(a, expand_keys, center, n_pubs_max, api_name)

        assert len( ids ) > 0, "Overly-restrictive search, no ids (bibcodes, etc) to retrieve."

        # Sample to account for max number of publications we want to retrieve at once
        if len( ids ) > n_pubs_max:
            ids = np.random.choice( ids, n_pubs_max, replace=False )

        print( 'Expansion will include {} new publications.'.format( len( ids ) ) )

        # New atlas
        a_exp = atlas.Atlas.to_and_from_ids(
            a.atlas_dir, 
            ids, 
            api_name = api_name, 
            bibtex_fp = bibtex_fp,
        )

        # Update the new atlas
        a_exp.data._storage.update( a.data )

        return a_exp

    ########################################################################

    def record_update_history( self, pubs_per_update ):
        '''Record when publications were added.

        Args:
            pubs_per_update (list of list of strs):
                A list of which publications existed at which iteration,
                with the index of the overall list corresponding to the
                iteration the publication was added.

        Returns:
            self.update_history (array of ints):
                When publications were added. A value of -2 indicates
                no record of being added.
        '''

        # Loop backwards
        i_max = len( pubs_per_update ) - 1
        self.update_history = np.full( self.publications.shape, -2 )
        for i, pubs_i in enumerate( pubs_per_update[::-1] ):

            is_in = np.in1d( self.publications, pubs_i )
            self.update_history[is_in] = i_max - i

        return self.update_history

    ########################################################################

    def converged_kernel_size( self, key, backend=None ):
        '''Calculate the largest size of the kernel that's converged (at differing levels of convergence).

        Args:
            key (str):
                Key to calculate convergence w.r.t.

        Returns:
            kernel_size (np.ndarray of ints):
                The kernel size for converged kernels. The first column (or value, for single publications)
                indicates the largest kernel size that hasn't changed since the beginning.
                The second column indicates the largest kernel size that hasn't changed since the first update.
                Etc.

            cospsi_kernel (np.ndarray of floats):
                Value of cospsi for the largest converged kernel.
        '''

        if -2 in self.update_history:
            raise ValueError( 'Incomplete update history as indicated by entries with values of -2.' )

        # Used later
        max_rank =  self.update_history.max() 

        if isinstance( key, int ):
            publications = np.random.choice( self.publications, key, replace=False )
        elif key != 'all':
            publications = [ key, ]
        else:
            publications = self.publications

        if backend is None:
            backend = self.backend

        # Pure python calculation
        if backend == 'python':
            # Loop over all publications
            full_result = []
            full_cospsi_result = []
            for pub in tqdm( publications ):

                cospsi = self.cospsi( pub, 'all' )
                sort_inds = np.argsort( cospsi )[::-1]
                sorted_cospsi = cospsi[sort_inds]
                sorted_history = self.update_history[sort_inds]

                result = []
                cospsi_result = []
                max_rank =  self.update_history.max() 
                for rank in range( max_rank ):

                    result_i = np.argmin( sorted_history <= rank ) - 1
                    result.append( result_i )
                    cospsi_result.append( sorted_cospsi[result_i] )

                full_result.append( result )
                full_cospsi_result.append( cospsi_result )

            if len( full_result ) == 1:
                return full_result[0], full_cospsi_result[0]

            return np.array( full_result ), np.array( full_cospsi_result )
        elif backend == 'c/c++':
            if key != 'all':

                # Sorting input
                cospsi = self.cospsi( key, 'all' )
                sort_inds = np.argsort( cospsi )[::-1]
                sorted_history = self.update_history[sort_inds]

                # Setup types
                self.c_cartography.converged_kernel_size_row.restype = ctypes.POINTER( ctypes.c_int )
                self.c_cartography.converged_kernel_size_row.argtypes = [
                    np.ctypeslib.ndpointer( dtype=np.int32 ),
                    ctypes.c_int,
                    ctypes.c_int,
                ]

                # Call
                result_pointer = self.c_cartography.converged_kernel_size_row(
                    sorted_history.astype( 'int32' ),
                    sorted_history.size,
                    max_rank,
                )
                result = np.ctypeslib.as_array( result_pointer, ( max_rank, ) )

                return result

            else:
                sort_inds = np.argsort( self.cospsi_matrix )[:,::-1]
                sorted_history_flat = self.update_history[sort_inds].flatten()

                # Setup types
                self.c_cartography.converged_kernel_size.restype = ctypes.POINTER( ctypes.c_int )
                self.c_cartography.converged_kernel_size.argtypes = [
                    np.ctypeslib.ndpointer( dtype=np.int32 ),
                    ctypes.c_int,
                    ctypes.c_int,
                ]

                # Call
                n_pubs = self.publications.size
                result_pointer = self.c_cartography.converged_kernel_size(
                    sorted_history_flat.astype( 'int32' ),
                    n_pubs,
                    max_rank,
                )
                result = np.ctypeslib.as_array( result_pointer, ( n_pubs, max_rank, ) )

                return result

        else:
            raise KeyError( 'Unrecognized backend, {}'.format( backend ) )


    ########################################################################
    # Estimators
    ########################################################################

    def topography_metric(
        self,
        publications = None,
        metric = 'constant_asymmetry',
        min_prior = 2,
        date_type = 'entry_dates',
        kernel_size = 16,
        **kwargs
    ):
        '''Estimate the asymmetry of all publications relative to prior
        publications.

        Args:
           metric (str):
                What metric to use. Options are...
                    constant_asymmetry:
                        e = mag( sum( |P>-|P_i> ) )
                    kernel_constant_asymmetry:
                        e = mag( sum( |P>-|P_i> ) ), w/in kernel only
                    density:
                        e = kernel_size/(4/3pi*h**3)

            min_prior (int):
                The minimum number of publications prior to the target
                publication to calculate the metric for it.

            date_type (str):
                What date to use for the publications when calculating prior
                publications

        Returns:
            es (np.ndarray of floats):
                Estimator vallue for all publications.
        '''

        # By default calculate for all publications
        if publications is None:
            publications = self.inds

        # Identify valid publications to use as input
        is_minnorm = self.norms > 1e-5

        es = []
        dates = getattr( self, date_type )
        for i in tqdm( publications ):

            # Don't try to calculate for publications we don't have a date for.
            date = dates[i]
            if str( date ) == 'NaT' or np.isclose( self.norms[i], 0. ):
                es.append( np.nan )
                continue

            # Identify prior publications
            is_prior = dates < date
            if is_prior.sum() < min_prior:
                es.append( np.nan )
                continue

            # Choose valid publications
            is_other = self.inds != i
            is_valid = is_prior & is_other & is_minnorm
            valid_is = self.inds[is_valid]

            # Get the metric
            fn = getattr( self, '{}_metric'.format( metric ) )

            # Identify arguments to pass
            fn_args = inspect.getfullargspec( fn )
            used_kwargs = {}
            for key, item in kwargs.items():
                if key in fn_args.args:
                    used_kwargs[key] = item
            if 'kernel_size' in fn_args.args:
                used_kwargs['kernel_size'] = kernel_size

            # Call
            e = fn( i, valid_is, **used_kwargs )

            es.append( e )
        es = np.array( es )

        return es

    ########################################################################

    def constant_asymmetry_metric(
        self,
        i,
        is_valid,
    ):
        '''Estimate the asymmetry of a publication by calculating the difference
        between that publication's projection and all other publications.

        Args:
            i (int):
                The index of the vector to calculate the asymmetry metric for.

            valid_is ((n_other) np.ndarray of ints):
                Indices of the other publication used when calculating the
                metric.

        Returns:
            mag (float):
                Magnitude of the asymmetry metric.
        '''

        p = self.vectors_notsp_normed[i]
        other_p = self.vectors_notsp_normed[is_valid]

        # Differences
        result = ( p - other_p ).sum( axis=0 )
        mag = np.linalg.norm( result )

        return mag

    ########################################################################

    def kernel_constant_asymmetry_metric(
        self,
        i,
        valid_is,
        kernel_size = 16,
    ):
        '''Estimate the asymmetry of a publication by calculating the difference
        between that publication's projection and the other publications within
        the kernel.

        Args:
            i (int):
                The index of the vector to calculate the asymmetry metric for.

            valid_is ((n_other) np.ndarray of ints):
                Indices of the other publication used when calculating the
                metric.

            kernel_size (int):
                Number of nearest neighbors to calculate the asymmetry on.

        Returns:
            mag (float):
                Magnitude of the asymmetry metric.
        '''

        # We can't have the kernel larger than the number of valid publications
        if kernel_size > len( valid_is ):
            return np.nan

        # Input
        cospsi = self.cospsi_matrix[i][valid_is]
        sorted_inds = np.argsort( cospsi )[::-1][:kernel_size]
        other_inds = self.inds[valid_is][sorted_inds]
        p = self.vectors_notsp_normed[i]
        used_p = self.vectors_notsp_normed[other_inds]

        # Differences
        diff = p - used_p
        diff_mag = np.linalg.norm( diff, axis=1 )
        result = ( diff / diff_mag[:,np.newaxis ]).sum( axis=0 )
        mag = np.linalg.norm( result )

        return mag

    ########################################################################

    def fringe_factor_metric(
        self,
        i,
        valid_is,
        kernel_size = 16,
    ):
        '''Estimate the asymmetry of a publication by calculating the difference
        between that publication's projection and the other publications within
        the kernel. Normalized to between 0 and 1.

        Args:
            i (int):
                The index of the vector to calculate the asymmetry metric for.

            valid_is ((n_other) np.ndarray of ints):
                Indices of the other publication used when calculating the
                metric.

            kernel_size (int):
                Number of nearest neighbors to calculate the asymmetry on.

        Returns:
            mag (float):
                Magnitude of the asymmetry metric.
        '''

        return self.kernel_constant_asymmetry_metric(
            i = i,
            valid_is = valid_is,
            kernel_size = kernel_size,
        ) / kernel_size

    ########################################################################

    def density_metric(
        self,
        i,
        valid_is,
        kernel_size = 16,
    ):
        '''Estimate the density of a publication by calculating the
        smoothing length that encloses kernel_size other publications.

        Args:
            i (int):
                The index of the vector to calculate the asymmetry metric for.

            valid_is ((n_other) np.ndarray of ints):
                Indices of the other publication used when calculating the
                metric.

            kernel_size (int):
                Number of nearest neighbors to calculate the asymmetry on.

        Returns:
            density (float):
                kernel_size divided by arc length containing kernel_size other publications.
        '''

        h = self.smoothing_length_metric( i, valid_is, kernel_size )
        density = kernel_size / h
                
        return density

    ########################################################################

    def smoothing_length_metric(
        self,
        i,
        valid_is,
        kernel_size = 16,
    ):
        '''Proxy for the density of a publication defined as the minimum
        arc length that encloses kernel_size other publications.

        Args:
            i (int):
                The index of the vector to calculate the asymmetry metric for.

            valid_is ((n_other) np.ndarray of ints):
                Indices of the other publication used when calculating the
                metric.

            kernel_size (int):
                Number of nearest neighbors to calculate the asymmetry on.

        Returns:
            h (float):
                Arc length containing kernel_size other publications.
                (Assumes normalized to a radius of 1.)
        '''

        # We can't have the kernel larger than the number of valid publications
        if kernel_size > len( valid_is ):
            return np.nan

        cospsi = self.cospsi_matrix[i][valid_is]
        cospsi_max = np.sort( cospsi )[::-1][kernel_size-1]
        return np.arccos( cospsi_max )

    ########################################################################
    # Mapping
    ########################################################################

    def map(
        self,
        center,
        distance_transformation = 'exponential',
        median_psi = None,
        std_psi = None,
        max_links = None,
        max_searched = None,
        save_filepath = None,
        overwrite = False,
        use_numba = True,
    ):
        '''Generate a map of the publications.
        When projecting from an N-dimensional space to a two dimensional space, we can only preserve
        two of the distances per publication. The default for the map preserves the distance between
        publication i and the central publication, and publication i and the most similar publication.

        Args:
            center (str):
                Citation key for the central publication.

            distance_transformation (str):
                What are the distances between particles? Options:
                'arc length':
                    arccos( cospsi ), since vectors are normalized to a sphere of radius 1.
                'exponential':
                    exp( ( psi_ij - [median psi] ) / [std psi] )
                    
            median_psi (float):
                Median psi to use for the exponential distance transform.
                If not provided will be calculated from self.cospsi_matrix.
                    
            std_psi (float):
                Standard deviation of psi to use for the exponential distance transform.
                If not provided will be calculated from self.cospsi_matrix.

            max_links (int):
                Maximum number of times an individual publication can be linked to, including
                the central publication. Changes the appearance of the map.

            max_searched (int):
                For most publications two distances can be preserved per publication.
                However, this is not always possible. max_searched is the number of times
                the algorithm will look for a position that preserves two distances
                before giving up and preserving one distance.

            save_filepath (str):
                Location to save the data at, if given.

            overwrite (bool):
                If True, overwrite any saved data, otherwise use existing data.

            use_numba (bool):
                If True, use numba for the calculation.
        '''
        
        # Retrieve saved data, if any
        if save_filepath is not None:
            save_data = verdict.Dict.from_hdf5( save_filepath, create_nonexistent=True )
            if not overwrite and ( center in save_data ):
                group = save_data[center]
                return group['coordinates'], group['ordered indices'], group['pairs']

        # Setup relation to central publication
        i_center = self.inds[center == self.publications][0] # NOTE: this seems like a typo?
        cospsi_0is = self.cospsi_matrix[i_center]
        sort_inds = np.argsort( cospsi_0is )[::-1]

        # Get distances
        d_matrix = np.arccos( self.cospsi_matrix )
        if distance_transformation == 'arc length':
            pass
        elif distance_transformation == 'exponential':
            if median_psi is None:
                median_psi = np.nanmedian( d_matrix )
            if std_psi is None:
                std_psi = np.nanstd( d_matrix )
            d_matrix = np.exp( ( d_matrix - median_psi ) / std_psi )
        else:
            raise KeyError( 'Unrecognized distance_transformation, {}'.format( distance_transformation ) )
        # Set diagonals to 0
        n_p = len( self.publications )
        d_matrix[np.arange(n_p),np.arange(n_p)] = 0.

        # Build sorted indices matrix
        sort_inds_matrix = np.full( d_matrix.shape, -9999 )
        print( 'Sorting....' )
        for m, i in enumerate( tqdm( sort_inds ) ):
            if m < 2:
                continue
            d_for_sorting = d_matrix[i][sort_inds[:m]]
            sort_inds_matrix[i][:m] = sort_inds[:m][np.argsort( d_for_sorting )]

        # Setup data structures
        coords = np.full( ( n_p, 2 ), fill_value=np.nan )
        # mapped_inds = np.full( len( sort_inds ), fill_value=-9999, dtype=int )
        pairs = np.full( ( n_p, 2 ), fill_value=-9999, dtype=int )
        n_linked = np.zeros( n_p, dtype=int )

        # Input central data
        coords[i_center,:] = np.array([
            [ 0., 0., ],
        ])
        coords[sort_inds[1],:] = np.array([
            [ d_matrix[i_center][sort_inds[1]], 0. ]
        ])
        pairs[sort_inds[1],0] = i_center
        n_linked[i_center] += 1
        n_linked[sort_inds[1]] += 1

        # Format for function
        if max_links is None:
            max_links = -1
        if max_searched is None:
            max_searched = -1

        if use_numba:
            map_fn = numba.njit( _generate_map )
        else:
            map_fn = _generate_map

        coords, mapped_inds, pairs = map_fn(
            sort_inds,
            coords,
            sort_inds_matrix,
            pairs,
            n_linked,
            d_matrix,
            max_links,
            max_searched,
        )

        if save_filepath is not None:
            save_data[center] = {
                'coordinates': coords,
                'ordered indices': mapped_inds,
                'pairs': pairs,
            }
            save_data.to_hdf5( save_filepath )

        return coords, mapped_inds, pairs

    ########################################################################

    def plot_map(
        self,
        center = None,
        coords = None,
        inds = None,
        pairs = None,
        ax = None,
        colors = None,
        edgecolors = None,
        cmap = 'cubehelix',
        norm = None,
        hatching = None,
        xlim = None,
        ylim = None,
        vlim = None,
        scatter = True,
        scatter_kwargs = {},
        links = False,
        links_kwargs = {},
        histogram = False,
        histogram_kwargs = {},
        histogram_plot_kwargs = {},
        voronoi = False,
        voronoi_kwargs = {},
        labels = False,
        labels_placer_voronoi = True,
        labels_formatter = None,
        labels_kwargs = {},
        **kwargs
    ):
        '''Plot a 2D projection of the N-dimensional feature space.
        The default method used is self.map, which preserves some pairwise distances.

        Args:
            center (str):
                The central publication for the map.

            coords ((N,2) dimensional array):
                2D location of each publication. If None, calculated via self.map.

            inds (N-dimensional array):
                Order in which the publications were mapped. Typically only used for labels.

            pairs ((N,2) dimensional array):
                For each publication what other publication it preserves a distance to.

            ax (matplotlib axis):
                Axis to plot on. Creates a new figure if this is not passed.

            colors (N-dimensional array):
                Values of the colors to use for the publications.

            edgecolors (N-dimensional array):
                Values of the edgecolors to use for the publications (voronoi cells only).

            cmap:
                Colormap to use.

            norm:
                Normalization to use.

            hatching (N-dimensional array):
                Hatching to use for the publications (voronoi cells only).

            xlim ((2,) list):
                x-limits to use. Important to set these through this function if using voronoi labels.

            ylim ((2,) list):
                y-limits to use. Important to set these through this function if using voronoi labels.

            clean_plot (bool):
                If True, remove axis ticks.

            scatter (bool):
                If True, add a scatter point for each publicaiton.

            scatter_kwargs (dict):
                Keyword arguments to pass to ax.scatter

            links (bool):
                If True, draw lines between pairs with preserved distances.

            links_kwargs (dict):
                Keyword arguments to pass to ax.plot

            histogram (bool):
                If True, plot the publication locations via a 2D histogram.

            histogram_kwargs (dict):
                Keyword arguments to pass to np.histogram2d

            histogram_plot_kwargs (dict):
                Keyword arguments to pass to ax.pcolormesh

            voronoi (bool):
                If True, represent each publication with a voronoi cell.

            voronoi_kwargs (dict):
                Keyword arguments to pass to cc.utils.plot_voronoi

            labels (bool):
                If True, label each point.

            labels_placer_voronoi (bool):
                If True, decide on the location of each label by placing it in a voronoi cell. Remove labels that don't fit.

            labels_formatter (function):
                If passed, a function that decides how each publication could be labeled.
                The function must take in arguments ( i, m_i, c ), where i is the index in c.inds of the publication,
                m_i is the index indicating order in which the publication was placed, and c is a Cartographer object.

            labels_kwargs (dict):
                Keyword arguments to pass to ax.annotate

            **kwargs:
                Keyword arguments to pass to self.map
                Only used if coords is None.
        '''

        # Get needed data
        if coords is None:
            coords, inds, pairs = self.map( center, **kwargs )
        if inds is None:
            inds = np.arange( coords.shape[0] )

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        # Setup limits
        if xlim is None:
            xmin = coords[:,0].min()
            xmax = coords[:,0].max()
            xwidth = xmax - xmin
            xlim = [ xmin - 0.1 * xwidth, xmax + 0.1 * xwidth ]
        if ylim is None:
            ymin = coords[:,1].min()
            ymax = coords[:,1].max()
            ywidth = ymax - ymin
            ylim = [ ymin - 0.1 * ywidth, ymax + 0.1 * ywidth ]
        def is_inside( x, y, f=0.5 ):
            '''Omit points that are a multiple f of the width outside the limits.'''
            xwidth = xlim[1] - xlim[0]
            ywidth = ylim[1] - ylim[0]
            inside_x = ( x > xlim[0] - f*xwidth ) & ( x < xlim[1] + f*xwidth )
            inside_y = ( y > ylim[0] - f*ywidth ) & ( y < ylim[1] + f*ywidth )
            inside = inside_x & inside_y
            return inside

        # Setup color limits
        if colors is not None:
            if vlim is None:
                vmin = np.nanmin( colors )
                vmax = np.nanmax( colors )
                vlim = [ vmin, vmax ]
            if norm is None:
                norm = matplotlib.colors.Normalize( vmin=vlim[0], vmax=vlim[1] )

        # Scatter plot
        if scatter:
            scatter_kwargs_used = {
                'color': 'k',
                'cmap': cmap,
                'norm': norm,
            }
            if colors is not None:
                scatter_kwargs_used['c'] = colors
                del scatter_kwargs_used['color']
            scatter_kwargs_used.update( scatter_kwargs )
            ax.scatter(
                coords[:,0],
                coords[:,1],
                **scatter_kwargs_used,
            )

        # Plot pairwise distances.
        if links:
            assert pairs is not None, 'pairs cannot be "None" when coords is not "None".'
            for i, pairs_i in enumerate( pairs ):
                for j in pairs_i:

                    if j < 0:
                        continue

                    # Only plot those in bounds...
                    if not is_inside( coords[i,0], coords[i,1], f=0. ):
                        continue
                    if not is_inside( coords[j,0], coords[j,1], f=0. ):
                        continue

                    links_kwargs_used = dict(
                        color = 'k',
                        zorder = -10,
                        alpha = 0.2,
                    )
                    links_kwargs_used.update( links_kwargs )

                    ax.plot(
                        [ coords[i,0], coords[j,0] ],
                        [ coords[i,1], coords[j,1] ],
                        **links_kwargs_used
                    )

        # Histogram plot
        if histogram:

            # Calculate
            hist_kwargs_used = {
                'bins': [
                    np.linspace( xlim[0], xlim[1], 64 ),
                    np.linspace( ylim[0], ylim[1], 64 ),
                ]
            }
            hist_kwargs_used.update( histogram_kwargs )
            hist2d, x_edges, y_edges = np.histogram2d(
                coords[:,0],
                coords[:,1],
                **hist_kwargs_used
            )
            if colors is not None:
                weighted_hist2d, x_edges, y_edges = np.histogram2d(
                    coords[:,0],
                    coords[:,1],
                    weights = colors,
                    **hist_kwargs_used
                )
                hist2d = weighted_hist2d / hist2d

            # Plot
            if norm is None:
                norm = matplotlib.colors.LogNorm()
            hist_plot_kwargs_used = {
                'cmap': cmap,
                'norm': norm,
            }
            hist_plot_kwargs_used.update( histogram_plot_kwargs )
            img = ax.pcolormesh(
                x_edges,
                y_edges,
                hist2d.transpose(),
                **hist_plot_kwargs_used
            )

            if colors is None:
                # Create divider for existing axes instance                                
                divider = make_axes_locatable( ax )                                   
                # Append axes to the right of ax, with 5% width of ax                      
                cax = divider.append_axes("right", pad=0.05, size='5%')                    
                # Create colorbar in the appended axes                                     
                cbar = plt.colorbar( img, cax=cax, )

        # Label formatting
        if labels:
            def default_labels_formatter( ii, m_ii, c ):
                return '{}: {}'.format( m_ii, c.publications[ii][:10] )
            if labels_formatter is None:
                labels_formatter = default_labels_formatter
            m_is = np.argsort( inds )
            labels_list = [
                labels_formatter( i, m_is[i], self ) for i in np.arange( inds.size )
            ]
        else:
            labels_list = None

        # Alternative to voronoi labels (not great)
        if labels and not labels_placer_voronoi:
            for m_i, i in enumerate( inds ):

                coord = coords[i]

                labels_kwargs_used = dict(
                    xycoords = 'data',
                    xytext = ( 5, 5 ),
                    textcoords = 'offset points',
                    va = 'bottom',
                    ha = 'left',
                )
                labels_kwargs_used.update( labels_kwargs )
                ax.annotate(
                    text = labels_list[i],
                    xy = coord,
                    **labels_kwargs_used
                )

        # Voronoi labels or cells
        voronoi_labels = ( labels and labels_placer_voronoi )
        if voronoi or voronoi_labels:

            # Only plot those that are visible (reduce expenses majorly)
            x, y = coords.transpose()
            inside = is_inside( x, y )
            
            # Set up keyword arguments
            voronoi_kwargs_used = {
                'cmap': cmap,
                'norm': norm,
            }
            voronoi_kwargs_used.update( voronoi_kwargs )

            if voronoi_labels:
                voronoi_labels_list = np.array( labels_list )[inside]
                voronoi_kwargs_used.update( labels_kwargs )
            else:
                voronoi_labels_list = None

            if colors is not None:
                colors = colors[inside]
            if edgecolors is not None:
                if edgecolors == 'colors':
                    edgecolors = colors
                else:
                    edgecolors = edgecolors[inside]
            if hatching is not None:
                hatching = hatching[inside]

            ax, vor = utils.plot_voronoi(
                coords[inside],
                labels = voronoi_labels_list,
                plot_cells = voronoi,
                xlim = xlim,
                ylim = ylim,
                ax = ax,
                colors = colors,
                edgecolors = edgecolors,
                hatching = hatching,
                **voronoi_kwargs_used
            )
        else:
            ax.set_xlim( xlim )
            ax.set_ylim( ylim )
            ax.set_aspect( 'equal' )

        # Colorbar
        if colors is not None:
            # Create divider for existing axes instance                                
            divider = make_axes_locatable( ax )                                   
            # Append axes to the right of ax, with 5% width of ax                      
            cax = divider.append_axes("right", pad=0.05, size='5%')                    
            # Create colorbar in the appended axes                                     
            matplotlib.colorbar.ColorbarBase(
                cax,
                cmap = cmap,
                norm = norm,
            )

        return ax, ( coords, inds, pairs )

    ########################################################################

    def concept_rank_map(
        self,
        n_features = 100,
        highlighted_publications = None,
        default_color = '#000000',
        cmap = palettable.cartocolors.qualitative.Safe_10.hex_colors,
        return_data = False,
    ):
        '''An exploratory visualization designed to identify which papers are
        associated with which concepts, focusing on the highest ranked concepts
        (default rank is a metric of the most used within the atlas).

        In detail, the "concept rank map" has one integer per concept on the
        y axis, with lowest having the largest contribution to the atlas as
        measured by $\sum \mid P_{i, {\rm normed}} \rangle$.
        The x-axis is the value of the vector in the direction of that concept.
        This plot is best explored by mousing over the points of interest.

        Args:
            n_features (int):
                Number of concepts to plot.

            highlighted_publications (list of strs):
                These publications will have larger markers that are also
                colored.

            default_color (str or other specifier of color):
                The color to use for the unhighlighted publications.

            cmap (discrete colormap):
                The colors to use for the highlighted publications.

        Returns:
            fig (plotly figure)
        '''

        # Parameters
        n_x = self.publications.size
        n_y = n_features
        publications = self.publications # Can maybe make this flexible later
        if highlighted_publications is None:
            highlighted_publications = []
        
        # Default rank is the concepts that contribute most to the projection
        rank = self.vectors_notsp_normed.sum( axis=0 )
        sort_inds = np.argsort( rank )[::-1]

        # Get the sorted components
        vec_norm_s_all = self.vectors_notsp_normed[:,sort_inds]
        vec_norm_s = vec_norm_s_all[:,:n_y]
        feat_s_all = self.feature_names[sort_inds]
        feat_s = feat_s_all[:n_y]

        # Reformat for the scatter plot
        xs = vec_norm_s.transpose().flatten()
        ys = np.tile(
            np.arange( 0., n_y, 1. ),
            ( n_x, 1 ),
        ).transpose().flatten()
        labels = np.tile( publications, n_y )
        for i, y in enumerate( ys ):
            labels[i] = labels[i] + ', {}'.format( feat_s[int(y)] )

        # Setup the colors for the highlighted publications
        # Values used
        colors = np.zeros( ( n_x, ) ).astype( int )
        for i, hp in enumerate( highlighted_publications ):

            is_hp = hp == publications
            colors[is_hp] = i + 1

            if is_hp.sum() == 0:
                raise KeyError( 'No publication {} found'.format( hp ) )
        colors = np.tile( np.array( colors ), n_y )

        # Colormap formatted for plotly
        cmap = [ default_color, ] + cmap
        colorscale = []
        for i, color in enumerate( cmap ):
            colorscale.append( [ i / len( cmap ), color ] )
            colorscale.append( [ ( i + 1 )  /len( cmap ), color ] )

        # Remove zeros
        is_nonzero = np.invert( np.isclose( xs, 0. ) )
        xs = xs[is_nonzero]
        ys = ys[is_nonzero]
        colors = colors[is_nonzero]
        labels = labels[is_nonzero]
            
        # So the highlighted publications are plotted over the others
        resort_inds = np.argsort( colors )
        xs = xs[resort_inds]
        ys = ys[resort_inds]
        colors = colors[resort_inds]
        labels = labels[resort_inds]

        # Size of the markers
        size = 5 + ( colors != 0 ).astype( int ) * 5

        # Plotly Plot
        fig = go.Figure(
            data=go.Scatter(
                x = xs,
                y = ys,
                mode = 'markers',
                text = labels,
                marker = dict(
                    color = colors,
                    colorscale = colorscale,
                    size = size,
                )
            ),
            layout = go.Layout( width=800, height=800),
        )
        fig.update_xaxes(
            title_text = r'$\langle c_y | P \rangle$',
            range = [ 0, 1 ],
        )
        fig.update_yaxes(
            title_text = r'Concept Rank',
            range = [ -0.5, n_y - 0.5 ],
        )

        if return_data:
            return fig, (vec_norm_s_all, feat_s_all)

        return fig

########################################################################

def get_ids_list(a: atlas.Atlas, expand_keys: list[str], center: str, n_pubs_max: int, api_name = api.DEFAULT_API):
    '''For each publication corresponding to an id in `expand_keys`, collect the ids corresponding to the publication's references and citations.
    '''
    # Make the ids list
    existing_keys = set( a.data.keys() )
    ids = []
    for key in expand_keys:

        ids_i = []

        # ADS
        if api_name == api.ADS_API_NAME:
            try:
                ids_i += list( a[key].references )
            except (AttributeError, TypeError) as e:
                pass
            try:
                ids_i += list( a[key].citations )
            except (AttributeError, TypeError) as e:
                pass
        
        # S2 
        if api_name == api.S2_API_NAME:
            # use the Paper associated with the Publication.

            if not a[key].has_s2_data:
                breakpoint()

            expand_paper = a[key].paper
            papers = expand_paper.references + expand_paper.citations

            for paper in papers:
                
                id_i = None
                if paper.paperId is not None:
                    id_i = paper.paperId # no id prefix
                # use alternative ids if necessary
                elif paper.url is not None:
                    id_i = f"{api.S2_EXTERNAL_ID_TO_API_QUERY['URL']}:{paper.url}"
                else:
                    externalIds = paper.externalIds
                    if externalIds is not None:
                        # check them
                        for xid in externalIds: # assumes all are worth using
                            id_i = f"{api.S2_EXTERNAL_ID_TO_API_QUERY['URL']}:{paper.externalIds[xid]}"
                            break
                if id_i is None:
                    # TODO: use s2 to query based on title, as well
                    warnings.warn(f'Could not find any identifier for paper {paper}; skipping.')
                    continue
                ids_i.append(id_i)

        # Prune ids_i for obvious overlap
        ids += list( set( ids_i ) - existing_keys )

        # Break when the search is centered and we're maxed out
        if len( ids ) > n_pubs_max and center is not None:
            break
    ids = list( set( ids ) )  
    return ids  

def _generate_map(
    sort_inds,
    coords,
    sort_inds_matrix,
    pairs,
    n_linked,
    d_matrix,
    max_links,
    max_searched,
):
    '''Internal function used for generating maps.'''

    # Progres
    percentile_mult = 10
    n_print = 100 // percentile_mult + 1
    n = len( sort_inds[2:] )
    print_progress = n > n_print
    if print_progress:
        percentile_int = n // percentile_mult

    for m, i in enumerate( sort_inds[2:] ):

        if print_progress:
            if m % percentile_int == 0:
                print( str( ( m * 100 ) // n + 1 ) + '%' )

        # Because we start at sort_inds[2]
        m_i = m + 2

        sort_inds_for_j = sort_inds_matrix[i,:m_i]
        sort_inds_for_k = sort_inds[:m_i]

        # Shorten, if necessary
        if max_searched != -1:
            if len( sort_inds_for_j ) > max_searched:
                sort_inds_for_j = sort_inds_for_j[:max_searched]
            if len( sort_inds_for_k ) > max_searched:
                sort_inds_for_k = sort_inds_for_k[:max_searched]

        # Omit publications linked too much
        if max_links != -1:

            n_linked_k = n_linked[sort_inds_for_k]
            if np.sum( n_linked_k ) == max_links * len( n_linked_k ):
                print(
                    'No available publications to link to.' + \
                    'Increasing number of max links allowed from ' + \
                    str( max_links ) + ' to ' + str( max_links + 1 )
                )
                max_links += 1

            sort_inds_for_j = sort_inds_for_j[n_linked[sort_inds_for_j] < max_links]
            sort_inds_for_k = sort_inds_for_k[n_linked_k < max_links]

        valid_pairs_may_exist = (
            ( len( sort_inds_for_j ) > 0 ) &
            ( len( sort_inds_for_k ) > 0 )
        )
        two_valid_pairs = False
        if valid_pairs_may_exist:
            for m_k, k in enumerate( sort_inds_for_k ):
                for m_j, j in enumerate( sort_inds_for_j ):

                    # Skip duplicates
                    if j == k:
                        continue

                    # Get distances
                    d_ij = d_matrix[i,j]
                    d_ik = d_matrix[i,k]
                    # Important to use actual distance here,
                    # because distance between j and k may not be preserved
                    r_kj = coords[j] - coords[k]
                    d_jk = ( r_kj[0]**2. + r_kj[1]**2. )**0.5

                    # Conditions needed for deprojection
                    # Requires points can be reached
                    valid_ij = d_ik + d_jk > d_ij
                    valid_ik = d_ij + d_jk > d_ik
                    valid_jk = d_ij + d_ik > d_jk
                    if valid_ij & valid_ik & valid_jk:
                        two_valid_pairs = True
                        break

                if two_valid_pairs:
                    break

        # Most common case, where we can find a way to preserve two distances
        if two_valid_pairs:

            # Calculate directions parallel and perpendicular to r_kj (vector between j and k)
            parallel_hat = r_kj / d_jk
            perpendicular_hat = np.array([ -parallel_hat[1], parallel_hat[0] ])

            # Calculate angle of publication i relative to r_kj
            costhetak = ( d_ik**2. + d_jk**2. - d_ij**2. ) / ( 2. * d_ik * d_jk )

            # coord i components
            r_ki_llel = d_ik * costhetak * parallel_hat 
            r_ki_perp = d_ik * ( 1. - costhetak**2. )**0.5 * perpendicular_hat
            coords_i_a = coords[k] + r_ki_llel + r_ki_perp
            coords_i_b = coords[k] + r_ki_llel - r_ki_perp

            # Identify which of the two intersections to use
            # We use the intersection farthest from the least similar publication
            if len( sort_inds_for_j ) < 3:
                coords[i] = coords_i_a
            else:
                for l in sort_inds_for_j[::-1]:
                    if ( l != j ) and ( l != k ):
                        break
                r_il_a = coords_i_a - coords[l]
                d_il_a = ( r_il_a[0]**2. + r_il_a[1]**2. )**0.5
                r_il_b = coords_i_b - coords[l]
                d_il_b = ( r_il_b[0]**2. + r_il_b[1]**2. )**0.5

                if d_il_a > d_il_b:
                    coords[i] = coords_i_a
                else:
                    coords[i] = coords_i_b

            # Append other info
            pairs[i,:] = np.array([ j, k ])
            n_linked[i] += 2
            n_linked[j] += 1
            n_linked[k] += 1

        # Backup case, where we can only preserve one distance
        else:
            k = sort_inds_for_k[0]
            d_ik = d_matrix[i,k]

            # Find the least similar publication
            # and place the publication opposite
            found_least_similar = False
            for l in sort_inds_for_j[::-1]:
                if  l != k:
                    found_least_similar = True
                    break
            # Place the publication opposite the least similar
            if found_least_similar:
                r_lk_hat = coords[l] - coords[k]
                r_lk_hat /= ( r_lk_hat[0]**2. + r_lk_hat[1]**2. )**0.5
                coords[i] = coords[k] - d_ik * r_lk_hat
            # Otherwise, choose a random location
            else:
                x = np.random.uniform( -1, 1 )
                y = np.random.uniform( -1, 1 )
                r_ik_hat = np.array([ x, y ]) / ( x**2. + y**2. )**0.5
                r_ik = d_ik * r_ik_hat
                coords[i] = coords[k] + r_ik

            # Append other info
            pairs[i,0] = k
            n_linked[i] += 1
            n_linked[k] += 1

    return coords, sort_inds, pairs
