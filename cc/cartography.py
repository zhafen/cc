import copy
import numpy as np
import pandas as pd
import scipy.spatial
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings

import augment
import verdict

from . import atlas

########################################################################

class Cartographer( object ):
    '''Class for analyzing and exploring projected data.
    '''

    def __init__( self, **kwargs ):
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
    ):
        '''Update the data used for calculations.
        '''

        # Convert date to a more useable array
        self.publication_dates = pd.to_datetime( publication_dates )
        self.entry_dates = pd.to_datetime( entry_dates )

    ########################################################################

    @classmethod
    def from_hdf5( self, fp ):
        '''Load the cartographer from a saved file.

        Args:
            fp (str):
                Filepath to the projected data.

        Returns:
            Cartographer instance
        '''

        data = verdict.Dict.from_hdf5( fp )
        return Cartographer( **data )

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
    # Basic Analysis
    ########################################################################

    def inner_product( self, key_a, key_b, **kwargs ):
        '''Calculate the inner product between a and b, using the
        pre-generated concept projection.

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    atlas:
                        Inner product with the full atlas.
                    all:
                        Array of inner product with each publication.
                    key from self.data:
                        Inner product for a particular publication.

            key_b (str):
                Reference to the second object, same options as key_a.

        Keyword Args:
            Passed to self.concept_projection

        Returns:
            The inner product of a and b
        '''

        # When a==b we can use the norms
        if key_a == key_b:
            if key_a == 'atlas':
                return ( self.components.sum( axis=0 )**2. ).sum()
            elif key_a == 'all':
                return self.norms**2.
            else:
                is_p = self.publications == key_a
                return self.norms[is_p]**2.

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

        # When we're doing the inner product with the atlas for all pubs
        if sorted([ key_a, key_b ]) == [ 'all', 'atlas' ]:
            result = np.array([ np.dot( a, row ).sum() for row in b ])
            return result

        # Dot product
        try:
            result = np.dot( a, b )
        except ValueError:
            result = np.dot( b, a )

        # Finish dot product
        if key_a == 'atlas' or key_b == 'atlas':
            result = result.sum()

        return result

    ########################################################################

    def cospsi( self, key_a, key_b, **kwargs ):
        '''Cosine of the "angle" between two publications, defined as
        <A|B> / sqrt( <A|A><B|B> ).

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    atlas:
                        Inner product with the full atlas.
                    all:
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

    def psi( self, key_a, key_b, scaling=np.pi/2.,**kwargs ):
        '''The "angle" between two publications, defined as
        arccos( <A|B> / sqrt( <A|A><B|B> ) ).

        Args:
            key_a (str):
                Reference to the first object. Options are...
                    atlas:
                        Inner product with the full atlas.
                    all:
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
    # Exploration
    ########################################################################

    def explore( self, cite_key, a, n=5, max_searches=5, bibtex_fp=None ):
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
                a.process_abstracts()

                # Recalculate and update parameters
                print( '    Re-projecting publications...' )
                cp = a.concept_projection(
                    projection_fp = 'pass',
                    overwrite = True,
                    verbose = False,
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

    def survey( self, cite_key, a, psi_max, bibtex_fp=None, max_per_pub=100, **kwargs ):

        # Find publications in region
        psi = self.psi( cite_key, 'all', **kwargs )
        within_region = psi < psi_max

        # Choose a random publication
        survey_key = np.random.choice( self.publications[within_region] )

        # Import that publications references and citations
        bibcodes = (
            list( a[survey_key].citations ) +
            list( a[survey_key].references )
        )
        if len( bibcodes ) > max_per_pub:
            bibcodes = np.random.choice( bibcodes, max_per_pub, replace=False )
        print(
            'Importing {} new publications linked to random publication {}...'.format(
                len( bibcodes ),
                survey_key,
            )
        )
        a.import_bibcodes( bibcodes, bibtex_fp )

        # Make sure we have needed data
        print( 'Processing abstracts...' )
        a.process_abstracts()

        # Recalculate and update parameters
        print( 'Re-projecting publications...' )
        cp = a.concept_projection(
            projection_fp = 'pass',
            overwrite = True,
            verbose = False,
        )
        self.update_data( **cp )

        return a

    ########################################################################
    # Estimators
    ########################################################################

    def topography_metric(
        self,
        publications = None,
        metric = 'constant_asymmetry',
        min_prior = 2,
        date_type = 'entry_dates',
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
            publications = np.arange( self.publications.size )

        # Identify valid publications to use as input
        is_minnorm = self.norms > 1e-5

        es = []
        for i in tqdm( publications ):

            # Don't try to calculate for publications we don't have a date for.
            dates = getattr( self, date_type )
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
            p = self.components_normed[i]
            is_other = np.arange( self.publications.size ) != i
            is_valid = is_prior & is_other & is_minnorm
            other_p = self.components_normed[is_valid]

            # Get the metric
            fn = getattr( self, '{}_metric'.format( metric ) )
            e = fn( p, other_p,  **kwargs )

            es.append( e )
        es = np.array( es )

        return es

    ########################################################################

    def constant_asymmetry_metric(
        self,
        p,
        other_p,
    ):
        '''Estimate the asymmetry of a publication by calculating the difference
        between that publication's projection and all other publications.

        Args:
            p ((n_concepts,) np.ndarray of floats):
                The vector of the publication to calculate the asymmetry
                metric for.

            other_p ((n_other,n_concepts) np.ndarray of floats):
                Vectors of the other publication used when calculating the
                metric.

        Returns:
            result (np.ndarray of floats):
                Full asymmetry metric in vector form.

            mag (float):
                Magnitude of the asymmetry metric.
        '''

        # Differences
        result = ( p - other_p ).sum( axis=0 )
        mag = np.linalg.norm( result )

        return mag

    ########################################################################

    def kernel_constant_asymmetry_metric(
        self,
        p,
        other_p,
        kernel_size = 16,
    ):
        '''Estimate the asymmetry of a publication by calculating the difference
        between that publication's projection and all other publications.

        Args:
            p ((n_concepts,) np.ndarray of floats):
                The vector of the publication to calculate the asymmetry
                metric for.

            other_p ((n_other,n_concepts) np.ndarray of floats):
                Vectors of the other publication used when calculating the
                metric.

            kernel_size (int):
                Number of nearest neighbors to calculate the asymmetry on.

        Returns:
            result (np.ndarray of floats):
                Full asymmetry metric in vector form.

            mag (float):
                Magnitude of the asymmetry metric.
        '''

        # We can't have the kernel larger than the number of valid publications
        if kernel_size > other_p.shape[0]:
            kernel_size = other_p.shape[0]

        # Identify the publications to use in the calculation
        kd_tree = scipy.spatial.cKDTree( other_p )
        dist, inds = kd_tree.query( p, k=kernel_size, )
        used_p = other_p[inds]

        # Result is the sum of the normalized vectors
        diff = p - used_p
        diff_normed = diff / dist[:,np.newaxis] 
        result = diff_normed.sum( axis=0 )

        mag = np.linalg.norm( result )

        return mag

    ########################################################################

    def density_metric(
        self,
        p,
        other_p,
        kernel_size = 16,
    ):
        '''Estimate the density of a publication by calculating the
        spherical volume that encloses kernel_size other publications.
        NOTE: This is typically calculated in a many-dimensional space,
        and naiive estimates of calculating the spherical volume numerically
        fail due to overflow errors.

        Args:
            p ((n_concepts,) np.ndarray of floats):
                The vector of the publication to calculate the asymmetry
                metric for.

            other_p ((n_other,n_concepts) np.ndarray of floats):
                Vectors of the other publication used when calculating the
                metric.

            kernel_size (int):
                Number of nearest neighbors to calculate the asymmetry on.

        Returns:
            result (np.ndarray of floats):
                Full asymmetry metric in vector form.

            mag (float):
                Magnitude of the asymmetry metric.
        '''

        warnings.warn( 'Density is not currently well-defined in our code for many dimensions, which is the default. Consider using smoothing length instead.' )

        # We can't have the kernel larger than the number of valid publications
        if kernel_size > other_p.shape[0]:
            kernel_size = other_p.shape[0]

        # Identify the publications to use in the calculation
        kd_tree = scipy.spatial.cKDTree( other_p )
        dist, inds = kd_tree.query( p, k=kernel_size, )

        # Calculate density from smoothing length
        h = np.nanmax( dist )
        n_dim = self.component_concepts.size
        # Volume of an n-ball in n-dimensional Euclidean space
        volume = (
            np.pi**(n_dim/2.)
            * h**n_dim
            / scipy.special.gamma( n_dim/2. + 1 )
        )
        density = h / volume
                
        return density

