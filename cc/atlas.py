import ads
import bibtexparser
from collections import Counter
import copy
from tqdm import tqdm
import glob
import nltk
from nltk.metrics import edit_distance
import numpy as np
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt

import augment
import verdict

from . import publication
from . import utils

########################################################################

class Atlas( object ):
    '''Generate an Atlas from a bibliography.

    Args:
        atlas_dir (str):
            Primary location atlas data is stored in.

        bibtex_fp (str):
            Location to save the bibliography data at. Defaults to 
            $atlas_dir/cc_ads.bib

        data_fp (str):
            Location to save other atlas data at. Defaults to 
            $atlas_dir/atlas_data.h5

    Returns:
        Atlas:
            An atlas object, designed for exploring a collection of papers.
    '''

    @augment.store_parameters
    def __init__(
        self,
        atlas_dir,
        bibtex_fp = None,
        data_fp = None,
        load_bibtex = True,
    ):

        # Make sure the atlas directory exists
        os.makedirs( atlas_dir, exist_ok=True )
        
        self.data = verdict.Dict( {} )

        # Load bibtex data
        if load_bibtex:
            if bibtex_fp is None:
                bibtex_fp = os.path.join( atlas_dir, '*.bib' )
                bibtex_fps = glob.glob( bibtex_fp )
                if len( bibtex_fps ) > 1:
                    # Ignore the auxiliary downloaded biliography
                    cc_ads_fp = os.path.join( atlas_dir, 'cc_ads.bib' )
                    if cc_ads_fp in bibtex_fps:
                        bibtex_fps.remove( cc_ads_fp )
                    else:
                        raise IOError(
                            'Multiple possible BibTex files. Please specify.'
                        )
                if len( bibtex_fps ) == 0:
                    raise IOError( 'No *.bib file found in {}'.format( atlas_dir ) )
                bibtex_fp = bibtex_fps[0]
            self.import_bibtex( bibtex_fp )

        # Load general atlas data
        self.load_data( fp=data_fp )

    ########################################################################

    def __repr__( self ):
        return 'cc.atlas.Atlas:{}'.format( atlas_dir )

    def __repr__( self ):
        return 'Atlas'

    def __getitem__( self, key ):

        return self.data[key]

    ########################################################################

    @classmethod
    def from_bibcodes(
        cls,
        atlas_dir,
        bibcodes,
        bibtex_fp = None,
        data_fp = None,
        **kwargs
    ):
        '''Generate an Atlas from bibcodes by downloading and saving the
        citations from ADS as a new bibliography.

        Args:
            atlas_dir (str):
                Primary location atlas data is stored in.

            bibcodes (list of strs):
                Publications to retrieve.

            bibtex_fp (str):
                Location to save the bibliography data at. Defaults to 
                $atlas_dir/cc_ads.bib

            data_fp (str):
                Location to save other atlas data at. Defaults to 
                $atlas_dir/atlas_data.h5

        Returns:
            Atlas:
                An atlas object, designed for exploring a collection of papers.
        '''

        # Make sure the atlas directory exists
        os.makedirs( atlas_dir, exist_ok=True )

        # Save the bibcodes to a bibtex
        if bibtex_fp is None:
            bibtex_fp = os.path.join( atlas_dir, 'cc_ads.bib' )
        save_bibcodes_to_bibtex( bibcodes, bibtex_fp )

        result = Atlas(
            atlas_dir = atlas_dir,
            bibtex_fp = bibtex_fp,
            data_fp = data_fp,
            **kwargs
        )

        return result

    ########################################################################

    def import_bibcodes( self, bibcodes, bibtex_fp=None ):
        '''Import bibliography data using bibcodes.

        Args:
            bibcodes (list of strs):
                Publications to retrieve.

            bibtex_fp (str):
                Location to save the bibliography data at. Defaults to 
                $atlas_dir/cc_ads.bib

        Updates:
            self.data and the file at bibtex_fp:
                Saves the import bibliography data to the instance and disk.
        '''

        # Store original keys for later removing duplicates
        original_keys = copy.copy( list( self.data.keys() ) )

        # Import bibcodes
        if bibtex_fp is None:
            bibtex_fp = os.path.join( self.atlas_dir, 'cc_ads.bib' )
        save_bibcodes_to_bibtex( bibcodes, bibtex_fp, )
        self.import_bibtex( bibtex_fp, verbose=False )

        # Prune to remove duplicate references
        keys_to_remove = []
        for key in original_keys:
            item = self.data[key]
            for new_key, new_item in self.data.items():
                if key == new_key:
                    continue
                if 'arxivid' not in item.citation:
                    continue
                if 'arxivid' not in new_item.citation:
                    continue
                if item.citation['arxivid'] == new_item.citation['arxivid']:
                    keys_to_remove.append( new_key )
        for key in keys_to_remove:
            try:
                del self.data[key]
            except KeyError:
                # Already removed, it's okay
                continue

    ########################################################################

    def import_bibtex( self, bibtex_fp, verbose=True ):
        '''Import publications from a BibTex file.
        
        Args:
            bibtex_fp (str):
                Filepath to the BibTex file.
        '''

        if verbose:
            print( 'Loading bibliography entries.' )

        # Load the database
        with open( bibtex_fp ) as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        # Store into class
        if verbose:
            print( 'Storing bibliography entries.' )
        for citation in tqdm( bib_database.entries ):
            citation_key = citation['ID']

            # Avoid overwriting existing loaded data
            if citation_key in self.data:
                p = self.data[citation_key]
            else:
                p = publication.Publication( citation_key )

            p.citation = citation
            self.data[citation_key] = p

    ########################################################################

    def load_data( self, fp=None ):
        '''Load general data saved to atlas_data.h5
        
        Args:
            fp (str):
                Filepath to the atlas_data.h5 file.
                If None, looks in self.atlas_dir
        '''

        print( 'Loading saved atlas data.' )

        # Filepath
        if fp is None:
            fp = os.path.join( self.atlas_dir, 'atlas_data.h5' )

        # Exit if no data to load
        if not os.path.isfile( fp ):
            print( 'No saved data at {}'.format( fp ) )
            return

        # Load
        data_to_load = verdict.Dict.from_hdf5( fp )

        # Store data
        for key, item in tqdm( self.data.items() ):

            # When the paper doesn't have any data stored for it
            if key not in data_to_load:
                continue

            for ikey, iitem in data_to_load[key].items():
                setattr( item, ikey, iitem )

    ########################################################################

    def save_data(
        self,
        fp = None,
        attrs_to_save = [
            'abstract',
            'citations',
            'references',
            'bibcode',
            'entry_date'
        ],
        handle_jagged_arrs = 'row datasets',
        return_data = False,
    ):
        '''Save general data saved to atlas_data.h5
        
        Args:
            fp (str):
                Filepath to the atlas_data.h5 file.
                If None, looks in self.atlas_dir

            attrs_to_save (list of strs):
                List of attributes for each item of self.data to save.
        '''

        # Filepath
        if fp is None:
            fp = os.path.join( self.atlas_dir, 'atlas_data.h5' )

        # Retrieve data
        print( 'Preparing to save data.' )
        data_to_save = verdict.Dict( {} )
        for key, item in tqdm( self.data.items() ):
            data_to_save[key] = {}
            for attr in attrs_to_save:
                if hasattr( item, attr):
                    data_to_save[key][attr] = getattr( item, attr )
            # Don't try to save empty dictionaries
            if data_to_save[key] == {}:
                del data_to_save[key]

        # Save
        print( 'Saving to {}'.format( fp ) )
        data_to_save.to_hdf5( fp, handle_jagged_arrs=handle_jagged_arrs )

        if return_data:
            return data_to_save

    ########################################################################
    # Data Processing
    ########################################################################

    @property
    def key_concepts( self ):
        '''Easier access for key_concepts. Must be loaded for individual
        publications first.
        '''

        try:
            return self.data.notes.inner_item( 'key_concepts' )
        except KeyError:
            self.data.process_bibtex_annotations()
            return self.data.notes.inner_item( 'key_concepts' )

    ########################################################################

    @property
    def key_points( self ):
        '''Easier access for key_points. Must be loaded for individual
        publications first.
        '''

        try:
            return self.data.notes.inner_item( 'key_points' )
        except KeyError:
            self.data.process_bibtex_annotations()
            return self.data.notes.inner_item( 'key_points' )

    ########################################################################

    @property
    def all_key_concepts( self ):
        '''A set of all key concepts across publications.
        '''

        if not hasattr( self, '_all_key_concepts' ):

            # Flatten
            self._all_key_concepts = []
            for kcs in self.key_concepts.values():
                for kcs_point in kcs:
                    self._all_key_concepts += kcs_point

            self._all_key_concepts = set( self._all_key_concepts )

        return self._all_key_concepts

    ########################################################################

    def get_unique_key_concepts( self, **kwargs ):
        '''Unique key concepts, as simplified using nltk tools.
        Steps to retrieve unique key concepts:
        1. Union of the same stems.
        2. Concepts with a sufficiently low edit distance
           (accounts for mispellings)

        Optional Args:
            max_edit_distance (int):
                Maximum Levenshtein edit-distance between two concepts for them
                to count as the same concept.
        '''

        l = list( self.all_key_concepts )

        self.unique_key_concepts = utils.uniquify_words( l, **kwargs )

        return self.unique_key_concepts

    ########################################################################

    def get_ads_data(
        self,
        fl = [ 'abstract', 'citation', 'reference', 'entry_date', ],
        publications_per_request = 300,
        characters_per_request = 3000,
        identifier = 'key_as_bibcode',
    ):
        '''Get the ADS data for all publications.

        Args:
            fl (list of strs):
                Fields to retrieve from ADS.

            publications_per_request (int):
                Maximum number of publications to request per call to ADS.
                Not as limiting as characters_per_request in most cases.

            characters_per_request (int):
                Maximum number of characters per call to ADS. This is set a bit
                below the character limit ADS seems to have.

            identifier (str):
                What identifier to use to download papers. Options are...
                'key_as_bibcode':
                    This assumes self.data.keys() are ADS bibcodes
                    and we can just use them.
                'arxiv':
                    Use the arxiv ID contained in each publication's citation.
                    Requires some extra work to identify relevant papers.
        '''

        if identifier == 'key_as_bibcode' or identifier == 'bibcode':
            ids = list( self.data.keys() )
            identifier = 'bibcode'
            if identifier not in fl:
                fl.append( identifier )
        elif identifier == 'arxiv':
            # Create IDs
            ids = []
            for p in list( self.data.values() ):
                try:
                    ids.append( p.citation['arxivid'] )
                except KeyError:
                    ids.append( 'NULL' )
            # Make sure we can identify what's retrieved
            if 'identifier' not in fl:
                fl.append( 'identifier' ) 
        else:
            raise KeyError( 'Unrecognized identifier, {}'.format( identifier ))

        # Build query strings
        ids_str = ''
        n_pubs = 0
        ids_strs = []
        for i, id in enumerate( ids ):

            if id == 'NULL':
                continue

            ids_str += '{}:{}'.format( identifier, id )
            n_pubs += 1

            # Break conditions
            end = i + 1 >= len( ids )
            max_pubs = n_pubs >= publications_per_request
            max_chars = len( ids_str ) >= characters_per_request
            if end:
                ids_strs.append( ids_str )
                break
            if max_pubs or max_chars:
                ids_strs.append( ids_str )
                n_pubs = 0
                ids_str = ''
                continue

            ids_str += ' OR '

        # Query
        print( '    Making {} ADS calls...'.format( len( ids_strs ) ) )
        results = []
        for ids_str in tqdm( ids_strs ):
            q = ads.SearchQuery(
                query_dict={
                    'q': ids_str,
                    'fl': fl,
                    'rows': publications_per_request,
                },
            )
            results += list( q )

        # Collate results
        if identifier == 'arxiv':
            def key_fn( result ):
                '''Need a special key fn for arxiv ID because it's not
                easily accessible in the returned ADS results. Need to hunt
                it down in a list of possible identifiers.'''
                for _ in result.identifier:
                    if _[:6] == 'arXiv:':
                        return _[6:]
        else:
            key_fn = lambda x: getattr( x, identifier )
        result_dict = {}
        for result in results:
            result_dict[key_fn(result)] = result

        # Assign properties
        for i, item in enumerate( self.data.values() ):
            id = ids[i]

            # Handle missing publications
            if id == 'NULL':
                continue

            item.ads_data = {}
            for f in fl:
                try:
                    value = getattr( result_dict[id], f )
                # Some IDs may fail
                except KeyError:
                    ids[i] = 'NULL'
                    del item.ads_data
                    break
                item.ads_data[f] = value
                attr_f = copy.copy( f )
                if attr_f == 'citation' or attr_f == 'reference':
                    attr_f += 's'
                setattr( item, attr_f, value )

    ########################################################################

    def process_abstracts( self, *args, **kwargs ):
        '''Download and process the abstracts of all publications.
        Faster and with fewer API calls than for each paper individually.

        *Args, **Kwargs:
            Passed to self.get_ads_data
        '''

        self.get_ads_data( *args, **kwargs )

        print( '    Doing NLP...' )

        n_err = 0
        for key, item in tqdm( self.data.items() ):
            if hasattr( item, 'ads_data' ):
                abstract_str = item.ads_data['abstract']
            else:
                abstract_str = ''
                n_err += 1
            item.process_abstract( abstract_str=abstract_str, overwrite=True )
        self.n_err_abs = n_err

    ########################################################################

    def concept_search(
        self,
        concept,
        max_edit_distance = 2,
        return_paragraph = True,
    ):
        '''Search all publications for those that are noted as discussing
        a given concept.

        Args:
            concept (str):
                Concept to search for.

            max_edit_distance (int):
                Maximum Levenshtein edit-distance between two concepts for them
                to count as the same concept.

        Returns:
            tuple containing...
                dict:
                    Dictionary with list of points discussing the concept per
                    publication.

                string:
                    Paragraph with points for the concept from each publication.
        '''

        # Stem the searched concept
        s = nltk.stem.SnowballStemmer( 'english' )
        words = nltk.word_tokenize( concept )
        stemmed_words = [ s.stem( w ) for w in words ]
        concept = ' '.join( stemmed_words )

        # Retrieve data
        self.data.process_bibtex_annotations()

        # Search through all
        result = {}
        for cite_key, kcs_p in self.key_concepts.items():
            for i, kcs in enumerate( kcs_p ):
                n_matches = 0
                for kc in kcs:
                    # Make the key concept into a stemmed version
                    words = nltk.word_tokenize( kc )
                    stemmed_words = [ s.stem( w ) for w in words ]
                    kc_stemmed = ' '.join( stemmed_words )

                    # Check for edit distance
                    if edit_distance( concept, kc_stemmed, ) <= max_edit_distance:
                        n_matches += 1

                if n_matches > 0:
                    # Create a dictionary for storage
                    if cite_key not in result:
                        result[cite_key] = []
                    point = self.key_points[cite_key][i]
                    result[cite_key].append( point )

        if not return_paragraph:
            return result
        else:
            paragraph = ''
            for key, item in result.items():
                for p in item:
                    paragraph += '\\cite{' + key + '}' + ': {}\n'.format( p )

            return result, paragraph

    ########################################################################

    def concept_projection(
        self,
        component_concepts = None,
        projection_fp = None,
        overwrite = False,
        existing = None,
        verbose = True,
        return_data = True,
        retrieve_abstracts = True,
    ):
        '''Project the abstract of each publication into concept space.
        In simplest form this finds all shared, stemmed nouns, verbs, and
        adjectives between all publications and counts them.

        Args:
            component_concepts (array-like of strs):                                  
                Basis concepts to project onto. Defaults to all concepts across
                all publications.

            projection_fp (str):
                Location to save the concept projection at. Defaults to
                $atlas_dir/projection.h5
                If set to 'pass' then the projection is not saved.

            overwrite (bool):
                If False then check for a cached concept projection.

            existing (dict or None):
                Dictionary of existing result to build the projection upon.

            verbose (bool):
                If True print additional information while running.

            return_data (bool):
                If True return the resultant dictionary.

        Returns:
            Dictionary:
                Dictionary containing...
                components ((n_pub,n_concepts) np.ndarray of ints):
                    The value at [i,j] is the value of the projection for
                    publication for each i for each concept j.

                norms ((n_pub,) np.ndarray of floats):
                    Normalization for each publication.

                component_concepts ((n_concepts,) np.ndarray of strs):
                    The basis concepts used. By default the union of all
                    stemmed nouns, adjectives, and verbs across all abstracts.

                publications ((n_pubs,) np.ndarray of strs):
                    The publications that are projected.

                publication_dates ((n_pubs,) np.ndarray of strs):
                    Dates of publication.

                entry_dates ((n_pubs) np.ndarray of strs):
                    Dates the database became aware of the publication.
                    Typically pre-publication, due to preprints.
        '''

        if verbose:
            print( 'Generating concept projection...' )

        # File location
        if projection_fp is None:
            projection_fp = os.path.join(
                self.atlas_dir,
                'projection.h5'
            )

        # If cached or saved and not overwriting
        if os.path.isfile( projection_fp ) and not overwrite:
            if verbose:
                print( 'Using saved concept projection...' )
            if existing is not None:
                warnings.warn(
                    'Passing an existing concept projection and not ' \
                    + 'overwriting. The concept projection will fail if the ' \
                    + 'existing and new concept projection share a save ' \
                    + 'location.'
                )
            self.projection = verdict.Dict.from_hdf5( projection_fp )
            return self.projection
        if hasattr( self, 'projection' ) and not overwrite:
            if verbose:
                print( 'Using cached concept projection...' )
            return self.projection

        # Set up for component calculation
        if existing is not None:
            assert component_concepts is None, "Cannot pass component " \
                + "concepts in addition to an existing projection."
            component_concepts = list( existing['component_concepts'] )
            components_list = list( existing['components'] )
            projected_publications = list( existing['publications'] )
            pub_date = list( existing['publication_dates'] )
            entry_date = list( existing['entry_dates'] )
        else:
            components_list = []
            projected_publications = []
            pub_date = []
            entry_date = []

        # Retrieve abstracts efficiently beforehand
        if retrieve_abstracts:
            self.process_abstracts()

        # Loop through and calculate components
        for key, item in tqdm( self.data.items() ):

            # Don't reproject existing publications
            if key in projected_publications:
                continue

            comp_i, component_concepts = item.concept_projection(
                component_concepts,
            )
            components_list.append( comp_i )
            projected_publications.append( key )
            pub_date.append( item.publication_date )
            try:
                entry_date.append( str( item.entry_date ) )
            except AttributeError:
                entry_date.append( 'NaT' )

        # Format components
        shape = (
            len( projected_publications ),
            len( component_concepts )
        )
        components = np.zeros( shape )
        for i, component in enumerate( components_list ):
            components[i,:len(component)] = component

        # Normalized components
        norm = np.linalg.norm( components, axis=1 )

        # Store
        self.projection = verdict.Dict( {
            'components': components,
            'norms': norm,
            'component_concepts': np.array( component_concepts ).astype( str ),
            'publications': np.array( projected_publications ),
            'publication_dates': np.array( pub_date ),
            'entry_dates': np.array( entry_date ),
        } )
        if projection_fp != 'pass':
            self.projection.to_hdf5( projection_fp )

        if return_data:
            return self.projection

    ########################################################################
    # Publication-to-publication comparison
    ########################################################################

    def inner_product_custom( self, other, **kwargs ):
        '''Calculate the inner product with another object.
        This is much more customizable than inner_product, but much, much
        slower.
        '''

        inner_product = 0

        # When the other object is a publication
        # isinstance raises false exceptions
        is_pub = (
            isinstance( other, publication.Publication ) or
            str( type( other ) ) == "<class 'cc.publication.Publication'>"
        )
        if is_pub:
            for p in self.data.values():
                inner_product += other.inner_product_custom( p, **kwargs )

        # When the other object is an atlas
        elif str( type( other ) ) == "<class 'cc.atlas.Atlas'>":
            for p_self in self.data.values():
                for p_other in self.data.values():
                    inner_product += p_other.inner_product_custom( p_self, **kwargs )
        else:
            raise ValueError( "Unrecognized object for calculating the inner product, {}".format( other ) )

        if inner_product == 0:
            warnings.warn( "Inner product == 0. Did you forget to load the data?" )

        return inner_product

    ########################################################################

    def cospsi_data( self, other, **kwargs ):
        '''Calculate the cos(psi) between the atlas' data and another object.
        psi is the "angle" between two objects, defined as
        cos( psi ) = <self | other> / sqrt( <self | self> * <other | other> )

        Args:
            other:
                The other object.

            **kwargs:
                Keyword arguments passed to inner_product.

        Returns:
            cospsi (verdict.Dict of floats or ints):
                cos(psi) calculated for each item of self.data.
        '''

        ### Calculate cospsi
        # Inner products
        ip_self = {}
        ips = {}
        for key, p in self.data.items():
            ip_self[key] = p.inner_product_custom( p, **kwargs )
            ips[key] = p.inner_product_custom( other, **kwargs )
        ip_self = verdict.Dict( ip_self )
        ip_other = other.inner_product_custom( other, **kwargs )
        ips = verdict.Dict( ips )

        # Cospsi
        cospsi = ips / ( ip_self * ip_other ).apply( np.sqrt )

        return cospsi

    ########################################################################
    # Plots
    ########################################################################

    def plot_cospsi2d(
        self,
        x_key,
        y_key,
        ax = None,
        x_kwargs = {},
        y_kwargs = {},
        **kwargs
    ):
        '''Scatter plot cos(psi) of two objects calculated with all the
        publications in the library.

        Args:
            x_obj:
                The x object to calculate cos(psi) with.

            y_obj:
                The y object to calculate cos(psi) with.

            ax:
                The axis to place the plot on.

            **kwargs:
                Keyword arguments to pass to the inner products.

        Returns:
            cospsi_x:
                Dictionary of values for the x_obj.

            cospsi_y:
                Dictionary of values for the y_obj.
        '''

        ### Calculate cospsi
        used_x_kwargs = copy.deepcopy( kwargs )
        used_x_kwargs.update( x_kwargs )
        used_y_kwargs = copy.deepcopy( kwargs )
        used_y_kwargs.update( y_kwargs )
        ip_xall = self.inner_product( x_key, 'all', **used_x_kwargs )
        ip_yall = self.inner_product( y_key, 'all', **used_y_kwargs )
        ip_xs = self.inner_product( x_key, x_key, **used_x_kwargs )
        ip_ys = self.inner_product( y_key, y_key, **used_y_kwargs )
        ip_xallall = self.inner_product( 'all', 'all', **used_x_kwargs )
        ip_yallall = self.inner_product( 'all', 'all', **used_y_kwargs )
        cospsi_xs = ip_xall / np.sqrt( ip_xs * ip_xallall )
        cospsi_ys = ip_yall / np.sqrt( ip_ys * ip_yallall )

        # Setup figure
        if ax is None:
            fig = plt.figure( figsize=(8,8), facecolor='w' )
            ax = plt.gca()

        # Plot
        ax.scatter(
            cospsi_xs,
            cospsi_ys,
            color = 'k',
            s = 50,
        )

        # Labels
        ax.set_xlabel( r'$\cos \psi$(' + str( x_key ) + ')', fontsize=22 )
        ax.set_ylabel( r'$\cos \psi$(' + str( y_key ) + ')', fontsize=22 )

        # Axis tweaks
        ax.set_xlim( 0, 1 )
        ax.set_ylim( 0, 1 )
        ax.set_aspect( 'equal' )

        return cospsi_xs, cospsi_ys

    ########################################################################

    def plot_cospsi2d_custom(
        self,
        x_obj,
        y_obj,
        ax = None,
        x_kwargs = {},
        y_kwargs = {},
        **kwargs
    ):
        '''Scatter plot cos(psi) of two objects calculated with all the
        publications in the library.

        Args:
            x_obj:
                The x object to calculate cos(psi) with.

            y_obj:
                The y object to calculate cos(psi) with.

            ax:
                The axis to place the plot on.

            **kwargs:
                Keyword arguments to pass to the inner products.

        Returns:
            cospsi_x:
                Dictionary of values for the x_obj.

            cospsi_y:
                Dictionary of values for the y_obj.
        '''

        ### Calculate cospsi
        used_x_kwargs = copy.deepcopy( kwargs )
        used_x_kwargs.update( x_kwargs )
        used_y_kwargs = copy.deepcopy( kwargs )
        used_y_kwargs.update( y_kwargs )
        cospsi_xs = self.cospsi_data( x_obj, **used_x_kwargs )
        cospsi_ys = self.cospsi_data( y_obj, **used_y_kwargs )

        # Setup figure
        if ax is None:
            fig = plt.figure( figsize=(8,8), facecolor='w' )
            ax = plt.gca()

        # Plot
        xs = cospsi_xs.array()
        ys = cospsi_ys.array()
        ax.scatter(
            xs,
            ys,
            color = 'k',
            s = 50,
        )

        # Labels
        ax.set_xlabel( r'$\cos \psi$(' + str( x_obj ) + ')', fontsize=22 )
        ax.set_ylabel( r'$\cos \psi$(' + str( y_obj ) + ')', fontsize=22 )

        # Axis tweaks
        ax.set_xlim( 0, 1 )
        ax.set_ylim( 0, 1 )
        ax.set_aspect( 'equal' )

        return cospsi_xs, cospsi_ys

########################################################################

def save_bibcodes_to_bibtex( bibcodes, bibtex_fp ):

    # ADS doesn't like np arrays
    bibcodes = list( bibcodes )

    # Retrieve data from ADS
    q = ads.ExportQuery( bibcodes )
    bibtex_str = q.execute()

    # Reformat some lines to work with bibtexparser
    # This is not optimized.
    l = []
    for line in bibtex_str.split( '\n' ):
        # ADS puts quotes instead of double brackes around the title
        if 'title =' in line:
            bibtex_str = bibtex_str.replace( '"{', '{{' )
            bibtex_str = bibtex_str.replace( '}"', '}}' )
        # ADS' bib export for months doesn't have brackets around it...
        if 'month =' in line:
            line = line.replace( '= ', '= {' ).replace( ',', '},' )
        # The eprint is usually also the arxivid.
        if 'eprint =' in line:
            l.append( line.replace( 'eprint', 'arxivid' ) )
        l.append( line )
    bibtex_str = '\n'.join( l )

    # Save the bibtex
    with open( bibtex_fp, 'a' ) as f:
        f.write( bibtex_str )
