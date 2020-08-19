import numpy as np
import pandas as pd

import augment
import verdict

from . import atlas

########################################################################

class Cartographer( object ):
    '''Class for analyzing and exploring projected data.
    '''

    @augment.store_parameters
    def __init__(
        self,
        components,
        norms,
        component_concepts,
        publications,
        publication_dates,
        entry_dates,
    ):

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
    # Exploration
    ########################################################################

    def explore( self, i, a ):

        pass

    ########################################################################
    # Estimators
    ########################################################################

    def asymmetry_estimator( self, estimator='constant', **kwargs ):
        '''Estimate the asymmetry of all publications relative to prior
        publications.

        Args:
           estimator (str):
                What estimator to use. Options are...
                    constant:
                        |A> = sum( |P>-|P_i> )

        Returns:
            all_mags (np.ndarray of floats):
                Asymmetry magnitude for all arrays.
        '''

        all_mags = []
        for i, cite_key in enumerate( self.publications ):

            # Get the estimator
            fn = getattr( self, '{}_asymmetry_estimator'.format( estimator ) )
            _, mag = fn( i, **kwargs )

            all_mags.append( mag )
        all_mags = np.array( all_mags )

        return all_mags

    ########################################################################

    def constant_asymmetry_estimator(
        self,
        i,
        min_prior = 2,
        date_type = 'entry_dates'
    ):
        '''Estimate the asymmetry of a publication by calculating the difference
        between that publication's projection and all other publications.

        Args:
            i (int):
                Index of the publication to calculate the estimator for.

            min_prior (int):
                The minimum number of publications prior to the target
                publication to calculate the estimator for it.

        Returns:
            result (np.ndarray of floats):
                Full asymmetry estimator in vector form.

            mag (float):
                Magnitude of the asymmetry estimator.
        '''

        # Don't try to calculate for publications we don't have a date for.
        dates = getattr( self, date_type )
        date = dates[i]
        if str( date ) == 'NaT' or np.isclose( self.norms[i], 0. ):
            return np.full( self.component_concepts.shape, np.nan ), np.nan

        # Identify prior publications
        is_prior = dates < date
        if is_prior.sum() < min_prior:
            return np.full( self.component_concepts.shape, np.nan ), np.nan

        # Identify valid publications to use as input
        is_other = np.arange( self.publications.size ) != i
        is_minnorm = self.norms > 1e-5
        is_valid = is_prior & is_other & is_minnorm

        # Differences
        p = self.components_normed[i]
        result = ( p - self.components_normed[is_valid] ).sum( axis=0 )
        mag = np.linalg.norm( result )

        return result, mag
