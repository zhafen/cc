import copy
import ctypes
import glob
import inspect
import numpy as np
import os
import pandas as pd
import scipy.sparse as ss
import scipy.spatial
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings

import plotly
import plotly.graph_objects as go
import palettable

import augment
import verdict

from . import atlas

########################################################################

class Cartographer( object ):
    '''Class for analyzing and exploring projected data.

        Args:
            backend (str):
                What code to use for calculations?
                Options are 'python', 'c/c++'.
    '''

    def __init__( self, backend='c/c++', **kwargs ):

        self.backend = backend

        if self.backend == 'c/c++':
            # Sparse matrix
            ## Get the c executable
            cc_dir = os.path.dirname( os.path.dirname( __file__ ) )
            lib_glob = os.path.join( cc_dir, 'build', '*/cartography*.so' )
            lib_fp = glob.glob( lib_glob )[0]
            self.c_cartography = ctypes.CDLL( lib_fp )

        self.update_data( **kwargs )

    @augment.store_parameters
    def update_data(
        self,
        components,
        norms,
        component_concepts,
        publications,
        publication_dates,
        entry_dates,
        prune_zeros = True,
    ):
        '''Update the data used for calculations.
        '''

        # Convert date to a more useable array
        self.publication_dates = pd.to_datetime( publication_dates, errors='coerce' )
        self.entry_dates = pd.to_datetime( entry_dates, errors='coerce' )

        if prune_zeros:
            self.prune_zero_entries()

    ########################################################################

    @property
    def inds( self ):

        if not hasattr( self, '_inds' ):

            self._inds = np.arange( self.publications.size )

        return self._inds

    ########################################################################

    @property
    def components_sp( self ):

        if not hasattr( self, '_components_sp' ):

            self._components_sp = ss.csr_matrix( self.components )

        return self._components_sp

    ########################################################################

    def prune_zero_entries( self ):
        '''Toss out any entries which have no components,
        usually due to no abstract.
        '''

        is_nonzero = np.invert( np.isclose( self.components.sum( axis=1 ), 0. ) )
        for attr in [ 'components', 'norms', 'publications', 'publication_dates', 'entry_dates' ]:
            value = getattr( self, attr )[is_nonzero]
            setattr( self, attr, value )

    ########################################################################

    @classmethod
    def from_hdf5( self, fp, sparse=True, backend='c/c++' ):
        '''Load the cartographer from a saved file.

        Args:
            fp (str):
                Filepath to the projected data.

            sparse (int):
                Whether or not the components are saved as a sparse matrix.

            backend (str):
                What code to use for calculations?
                Options are 'python', 'c/c++'.

        Returns:
            Cartographer instance
        '''

        data = verdict.Dict.from_hdf5( fp, sparse=sparse )

        if sparse:
            components_sp = copy.copy( data['components'] )
            data['components'] = data['components'].toarray()

        c = Cartographer( backend=backend, **data )

        if sparse:
            c._components_sp = components_sp

        return c

    ########################################################################

    @property
    def components_normed( self ):
        '''Components normalized such that <P|P>=1 .
        '''

        if not hasattr( self, '_components_normed' ):

            # Divide by NaN is unimportant and handled
            with np.errstate(divide='ignore',invalid='ignore'):

                self._components_normed = self.components / self.norms[:,np.newaxis]

        return self._components_normed

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
    # Basic Analysis
    ########################################################################

    def inner_product( self, key_a, key_b, backend=None, **kwargs ):
        '''Calculate the inner product between a and b, using the
        pre-generated concept projection.

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
            Passed to self.concept_projection

        Returns:
            The inner product of a and b
        '''

        # Swap a and b since it doesn't matter anyways when one is all
        if key_b != 'all' and key_a == 'all':
            key_a, key_b = key_b, key_a

        # When a==b we can use the norms
        if key_a == key_b:
            if key_a == 'atlas':
                return ( self.components.sum( axis=0 )**2. ).sum()
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
                    return self.components[is_p][0]
                # The entire atlas
                elif key == 'atlas' or key == 'all':
                    return self.components
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
                slice_a = slice(self.components_sp.indptr[i_a], self.components_sp.indptr[i_a+1])
                slice_b = slice(self.components_sp.indptr[i_b], self.components_sp.indptr[i_b+1])
                data_a = self.components_sp.data[slice_a]
                data_b = self.components_sp.data[slice_b]
                indices_a = self.components_sp.indices[slice_a]
                indices_b = self.components_sp.indices[slice_b]

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
                    self.components_sp.data.astype( 'int64' ),
                    self.components_sp.indices.astype( 'int64' ),
                    self.components_sp.indptr.astype( 'int64' ),
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
                self.components_sp.data.astype( 'int64' ),
                self.components_sp.indices.astype( 'int64' ),
                self.components_sp.indptr.astype( 'int64' ),
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
            Passed to self.concept_projection

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
            Passed to self.concept_projection

        Returns:
            psi of a and b
        '''

        cospsi = self.cospsi( key_a, key_b, **kwargs )

        return np.arccos( cospsi ) / scaling

    ########################################################################

    def distance( self, key_a, key_b, normed=True, ):

        # Get the right components
        if normed:
            components = self.components_normed
        else:
            components = self.components

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
        pre-generated concept projection.

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
            Passed to self.concept_projection

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
                return self.components[is_p][0]
            # The entire atlas
            elif key == 'atlas' or key == 'all':
                return self.components
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
        pre-generated concept projection. This option uses the
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
            Passed to self.concept_projection

        Returns:
            The amount of text overlap between a and b.
        '''

        return self.text_overlap( key_a, key_b, norm='geometric mean' )

    ########################################################################

    def pairwise( self, metric, trim_and_reshape=True, *args, **kwargs ):
        '''Calculate the pairwise metric between all publications in the concept projection.

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

    def explore(
        self,
        cite_key,
        a,
        n = 5,
        max_searches = 5,
        bibtex_fp = None,
        identifier = 'key_as_bibcode'
    ):
        '''Ambitiously explore an atlas region by aggressively downloading
        all citations and references of papers that are most similar to
        a target publication.
        Warning: Currently over-aggresive and will chew through your ADS
        calls.

        Args:
            cite_key (str):
                Target publication to find similar publications to.

            a (atlas.Atlas):
                Atlas class containing relevant data.

            n (int):
                Number of similar publications.

            max_searches (int):
                Maximum number of iterations to perform before stopping.

            bibtex_fp (str):
                Location to save the bibtex file to.

            identifier (str):
                What identifer to use for retrieving new papers. Defaults to
                ADS bibcode.

        Returns:
            a (atlas.Atlas):
                Modified atlas to include new publications.
        '''

        # Identify target publication
        p = a[cite_key]

        # Loop until converged
        prev_target_keys = []
        for j in range( max_searches ):

            print( 'Exploration loop {}'.format( j ) )
            print('    Current number of publications = {}'.format(len(a.data)))

            if j != 0:
                # Make sure we have needed data
                print( '    Retrieving and processing new abstracts...' )
                a.process_abstracts( identifier=identifier )

                # Format existing projection
                existing = {}
                for key in self.stored_parameters:
                    existing[key] = getattr( self, key )
                    if 'date' in key:
                        existing[key] = [ str(_) for _ in existing[key] ]

                # Recalculate and update parameters
                print( '    Re-projecting publications...' )
                cp = a.concept_projection(
                    projection_fp = 'pass',
                    overwrite = True,
                    verbose = False,
                    existing = existing,
                )
                self.update_data( **cp )

            # Find publications with highest cospsi relative to target publication
            print( '    Identifying similar publications...' )
            cospsi = self.cospsi( cite_key, 'all' )
            is_not_nan = np.invert( np.isnan( cospsi ) )
            target_inds = np.argsort( cospsi[is_not_nan] )[::-1][:n]
            target_keys = list( self.publications[is_not_nan][target_inds] )

            if target_keys == prev_target_keys:
                print( 'No new similar publications found, exiting...' )
                break

            # Don't redownload data from ads we already downloaded
            keys_to_search = []
            for key in target_keys:
                if key not in prev_target_keys:
                    keys_to_search.append( key )

            prev_target_keys = copy.copy( target_keys )

            # Build bibcodes to import
            bibcodes = []
            for key in keys_to_search:
                bibcodes += list( a[key].citations )
                bibcodes += list( a[key].references )

            # Import the new data
            print(
                '    Importing {} new publications...'.format( len( bibcodes ) )
            )
            a.import_bibcodes( bibcodes, bibtex_fp )

        print( 'Finished. New atlas has {} publications.'.format(len(a.data)))

        return a

    ########################################################################

    def survey(
        self,
        cite_key,
        a,
        psi_max,
        bibtex_fp = None,
        max_per_pub = 100,
        identifier = 'key_as_bibcode',
        **kwargs
    ):
        '''Explore publication space by choosing a random publication and
        importing all references and citations for that publication.

        Args:
            cite_key (str):
                The publication to center the search on.

            a (atlas.Atlas):
                The atlas for searching.

            psi_max (float):
                Maximum angle difference between the random publication and the central publication.

            bibtex_fp (str):
                Where to save the bibtex.

            max_per_pub (int):
                Number of publications maximum to import per surveyed publication.

            identifier (str):
                Identifier used for processing the abstracts.

        Kwargs:
            Passed to Psi calculation.
        '''

        # Find publications in region
        psi = self.psi( cite_key, 'all', **kwargs )
        within_region = psi < psi_max

        # Choose a random publication
        survey_key = np.random.choice( self.publications[within_region] )

        # Import that publications references and citations
        if a[survey_key].citations is not None:
            citations = list( a[survey_key].citations )
        else:
            citations = []
        if a[survey_key].references is not None:
            references = list( a[survey_key].references )
        else:
            references = []
        bibcodes = citations + references
        if len( bibcodes ) > max_per_pub:
            bibcodes = np.random.choice( bibcodes, max_per_pub, replace=False )
        print(
            'Importing {} publications linked to random publication {}...'.format(
                len( bibcodes ),
                survey_key,
            )
        )
        a.import_bibcodes( bibcodes, bibtex_fp )

        # Make sure we have needed data
        print( 'Processing abstracts...' )
        a.process_abstracts( identifier=identifier )

        # Format existing projection
        existing = {}
        for key in self.stored_parameters:
            existing[key] = getattr( self, key )
            if 'date' in key:
                existing[key] = [ str(_) for _ in existing[key] ]

        # Recalculate and update parameters
        print( 'Re-projecting publications...' )
        cp = a.concept_projection(
            projection_fp = 'pass',
            overwrite = True,
            verbose = False,
            existing = existing,
        )
        self.update_data( **cp )

        return a

    ########################################################################

    def expand( self, a, center=None, n_pubs_max=4000 ):
        '''Expand an atlas by retrieving all publications cited by the
        the publications in the given atlas, or that reference a
        publication in the given atlas.

        Args:
            a (atlas.Atlas):
                Atlas to expand.

            center (str):
                If given, center the search on this publication, preferentially
                searching related publications.

            n_pubs_max (int):
                Maximum number of publications allowed in the expansion.

        Returns:
            a_exp (atlas.Atlas):
                Expanded atlas. Has the same save location as a.
        '''

        # Without a center
        if center is None:
            expand_keys = list( a.data.keys() )
        else:
            cospsi = self.cospsi( center, 'all' )
            sort_inds = np.argsort( cospsi )[::-1]
            expand_keys = self.publications[sort_inds]

        # Make the bibcode list
        existing_keys = set( a.data.keys() )
        bibcodes = []
        for key in expand_keys:

            bibcodes_i = []
            try:
                bibcodes_i += list( a[key].references )
            except (AttributeError, TypeError) as e:
                pass
            try:
                bibcodes_i += list( a[key].citations )
            except (AttributeError, TypeError) as e:
                pass

            # Prune bibcodes_i for obvious overlap
            bibcodes += list( set( bibcodes_i ) - existing_keys )

            # Break when the search is centered and we're maxed out
            if len( bibcodes ) > n_pubs_max and center is not None:
                break
        bibcodes = list( set( bibcodes ) )

        assert len( bibcodes ) > 0, "Overly-restrictive search, no bibcodes to retrieve."

        # Sample to account for max number of publications we want to retrieve at once
        if len( bibcodes ) > n_pubs_max:
            bibcodes = np.random.choice( bibcodes, n_pubs_max, replace=False )

        print( 'Expansion will include {} new publications.'.format( len( bibcodes ) ) )

        # New atlas
        a_exp = atlas.Atlas.from_bibcodes( a.atlas_dir, bibcodes )

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
                # sorted_cospsi = cospsi[sort_inds]
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
                # sorted_cospsi_matrix = np.array([ cospsi_row[sort_inds[i]] for i, cospsi_row in enumerate( self.cospsi_matrix ) ])
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

        p = self.components_normed[i]
        other_p = self.components_normed[is_valid]

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
        p = self.components_normed[i]
        used_p = self.components_normed[other_inds]


        # Differences
        diff = p - used_p
        diff_mag = np.linalg.norm( diff, axis=1 )
        result = ( diff / diff_mag[:,np.newaxis ]).sum( axis=0 )
        mag = np.linalg.norm( result )

        return mag

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
        radius that encloses kernel_size other publications.

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
        cospsi_max = np.sort( cospsi )[::-1][kernel_size]
        return np.arccos( cospsi_max )

    ########################################################################
    # Mapping
    ########################################################################

    def concept_rank_map(
        self,
        n_concepts = 100,
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
            n_concepts (int):
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
        n_y = n_concepts
        publications = self.publications # Can maybe make this flexible later
        if highlighted_publications is None:
            highlighted_publications = []
        
        # Default rank is the concepts that contribute most to the projection
        rank = self.components_normed.sum( axis=0 )
        sort_inds = np.argsort( rank )[::-1]

        # Get the sorted components
        comp_norm_s_all = self.components_normed[:,sort_inds]
        comp_norm_s = comp_norm_s_all[:,:n_y]
        conc_s_all = self.component_concepts[sort_inds]
        conc_s = conc_s_all[:n_y]

        # Reformat for the scatter plot
        xs = comp_norm_s.transpose().flatten()
        ys = np.tile(
            np.arange( 0., n_y, 1. ),
            ( n_x, 1 ),
        ).transpose().flatten()
        labels = np.tile( publications, n_y )
        for i, y in enumerate( ys ):
            labels[i] = labels[i] + ', {}'.format( conc_s[int(y)] )

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
            return fig, (comp_norm_s_all, conc_s_all)

        return fig
