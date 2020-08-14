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
    def __init__( self, atlas_dir, bibtex_fp=None, data_fp=None ):
        
        self.data = verdict.Dict( {} )

        # Load bibtex data
        if bibtex_fp is None:
            bibtex_fp = os.path.join( atlas_dir, '*.bib' )
            bibtex_fps = glob.glob( bibtex_fp )
            if len( bibtex_fps ) > 1:
                # Ignore the auxiliary downloaded biliography
                cc_ads_fp = os.path.join( atlas_dir, 'cc_ads.bib' )
                if cc_ads_fp in bibtex_fps:
                    bibtex_fps.remove( cc_ads_fp )
                else:
                    raise IOError( 'Multiple possible BibTex files. Please specify.' )
            if len( bibtex_fps ) == 0:
                raise IOError( 'No *.bib file found in {}'.format( atlas_dir ) )
            bibtex_fp = bibtex_fps[0]
        self.import_bibtex( bibtex_fp )

        # Load general atlas data
        self.load_data( fp=data_fp )

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
        if bibtex_fp is None:
            bibtex_fp = os.path.join( atlas_dir, 'cc_ads.bib' )
        with open( bibtex_fp, 'a' ) as f:
            f.write( bibtex_str )

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

        # Import bibcodes
        new_a = Atlas.from_bibcodes( self.atlas_dir, bibcodes, bibtex_fp )

        # Prune to remove already existing references
        keys_to_remove = []
        for key, item in self.data.items():
            for new_key, new_item in new_a.data.items():
                if 'arxivid' not in item.citation:
                    continue
                if 'arxivid' not in new_item.citation:
                    continue
                if item.citation['arxivid'] == new_item.citation['arxivid']:
                    keys_to_remove.append( new_key )
        for key in keys_to_remove:
            del new_a.data[key]

        # Update data
        new_a.data._storage.update( self.data ) 
        self.data = new_a.data

    ########################################################################

    def __repr__( self ):
        return 'cc.atlas.Atlas:{}'.format( atlas_dir )

    def __repr__( self ):
        return 'Atlas'

    def __getitem__( self, key ):

        return self.data[key]

    ########################################################################

    def import_bibtex( self, bibtex_fp, ):
        '''Import publications from a BibTex file.
        
        Args:
            bibtex_fp (str):
                Filepath to the BibTex file.
        '''

        print( 'Loading bibliography entries.' )

        # Load the database
        with open( bibtex_fp ) as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        # Store into class
        print( 'Storing bibliography entries.' )
        for citation in tqdm( bib_database.entries ):
            citation_key = citation['ID']
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
        attrs_to_save = [ 'abstract', 'citations', 'references', 'bibcode' ],
        handle_jagged_arrs = 'row datasets',
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
                # Some attrs can be stored in ads_data
                else:
                    if hasattr( item, 'ads_data' ):
                        ads_key = attr[:-1]
                        if ads_key in item.ads_data:
                            data_to_save[key][attr] = item.ads_data[ads_key]
            # Don't try to save empty dictionaries
            if data_to_save[key] == {}:
                del data_to_save[key]

        # Save
        print( 'Saving to {}'.format( fp ) )
        data_to_save.to_hdf5( fp, handle_jagged_arrs=handle_jagged_arrs )

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
                    paragraph += '\cite{' + key + '}' + ': {}\n'.format( p )

            return result, paragraph

    ########################################################################

    def concept_projection(
        self,
        component_concepts = None,
        projection_fp = None,
        overwrite = False
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

            overwrite (bool):
                If False then check for a cached concept projection.

        Returns:
            Dictionary containing...
                components ((n_pub,n_concepts) np.ndarray of ints):
                    The value at [i,j] is the value of the projection for
                    publication for each i for each concept j.

                components_normed ((n_pub,n_concepts) np.ndarray of floats):
                    components normalized for each publication

                component_concepts ((n_concepts,) np.ndarray of strs):
                    The basis concepts used. By default the union of all
                    stemmed nouns, adjectives, and verbs across all abstracts.

                projected_publications ((n_pubs,) np.ndarray of strs):
                    The publications that are projected.
        '''

        print( 'Generating concept projection...' )

        # File location
        if projection_fp is None:
            projection_fp = os.path.join(
                self.atlas_dir,
                'projection.h5'
            )

        # If cached or saved and not overwriting
        if os.path.isfile( projection_fp ) and not overwrite:
            print( 'Using saved concept projection...' )
            self.projection = verdict.Dict.from_hdf5( projection_fp )
            return self.projection
        if hasattr( self, 'projection' ) and not overwrite:
            print( 'Using cached concept projection...' )
            return self.projection

        # Loop through and calculate components
        components_list = []
        projected_publications = []
        for key, item in tqdm( self.data.items() ):
            comp_i, component_concepts = item.concept_projection(
                component_concepts,
            )
            components_list.append( comp_i )
            projected_publications.append( key )

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
        components_normed = components / norm[:,np.newaxis]

        # Store
        self.projection = verdict.Dict( {
            'components': components,
            'components_normed': components_normed,
            'norms': norm,
            'component_concepts': component_concepts.astype( str ),
            'projected_publications': np.array( projected_publications ),
        } )
        self.projection.to_hdf5( projection_fp )

        return self.projection

    ########################################################################
    # Publication-to-publication comparison
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

        # Do projection or retrieve
        cp = self.concept_projection( **kwargs )

        # When a==b we can use the norms
        if key_a == key_b:
            if key_a == 'atlas':
                return ( cp['norms']**2. ).sum()
            elif key_a == 'all':
                return cp['norms']**2.
            else:
                is_p = cp['projected_publications'] == key_a
                return cp['norms'][is_p]**2.

        # Find the objects the keys refer to
        def interpret_key( key ):
            # A single publication
            if key in cp['projected_publications']:
                is_p = cp['projected_publications'] == key
                return cp['components'][is_p][0]
            # The entire atlas
            elif key == 'atlas' or key == 'all':
                return cp['components']
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
        ax.set_xlabel( r'$\cos \psi$(' + str( x_obj ) + ')', fontsize=22 )
        ax.set_ylabel( r'$\cos \psi$(' + str( y_obj ) + ')', fontsize=22 )

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
