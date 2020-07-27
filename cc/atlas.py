import bibtexparser
from collections import Counter
import copy
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

    @augment.store_parameters
    def __init__( self, atlas_dir, bibtex_fp=None, data_fp=None ):
        
        self.data = verdict.Dict( {} )

        # Load bibtex data
        if bibtex_fp is None:
            bibtex_fp = os.path.join( atlas_dir, '*.bib' )
            bibtex_fps = glob.glob( bibtex_fp )
            if len( bibtex_fps ) > 1:
                raise FileError( 'Multiple possible BibTex files. Please specify.' )
            if len( bibtex_fps ) == 0:
                raise FileError( 'No *.bib file found in {}'.format( atlas_dir ) )
            bibtex_fp = bibtex_fps[0]
        self.import_bibtex( bibtex_fp )

        # Load general atlas data
        self.load_data( fp=data_fp )

    def __repr__( self ):
        return 'cc.atlas.Atlas:{}'.format( atlas_dir )

    def __repr__( self ):
        return 'Atlas'

    ########################################################################

    def __getitem__( self, key ):

        return self.data[key]

    ########################################################################

    def import_bibtex( self, bibtex_fp, ):
        '''Import publications from a BibTex file.
        
        Args:
            bibtex_fp (str):
                Filepath to the BibTex file.
        '''

        # Load the database
        with open( bibtex_fp ) as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        # Store into class
        for citation in bib_database.entries:
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


        # Filepath
        if fp is None:
            fp = os.path.join( self.atlas_dir, 'atlas_data.h5' )

        # Exit if no data to load
        if not os.path.isfile( fp ):
            return

        # Load
        data_to_load = verdict.Dict.from_hdf5( fp )

        # Store data
        for key, item in self.data.items():
            for ikey, iitem in data_to_load[key].items():
                setattr( item, ikey, iitem )

    ########################################################################

    def save_data( self, fp=None, attrs_to_save=[ 'abstract', ] ):
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
        data_to_save = verdict.Dict( {} )
        for key, item in self.data.items():
            data_to_save[key] = {}
            for attr in attrs_to_save:
                data_to_save[key][attr] = getattr( item, attr )

        # Save
        data_to_save.to_hdf5( fp )

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
    # Publication-to-publication comparison
    ########################################################################

    def inner_product( self, other, **kwargs ):

        inner_product = 0

        # When the other object is a publication
        # isinstance raises false exceptions
        is_pub = (
            isinstance( other, publication.Publication ) or
            str( type( other ) ) == "<class 'cc.publication.Publication'>"
        )
        if is_pub:
            for p in self.data.values():
                inner_product += other.inner_product( p, **kwargs )

        # When the other object is an atlas
        elif str( type( other ) ) == "<class 'cc.atlas.Atlas'>":
            for p_self in self.data.values():
                for p_other in self.data.values():
                    inner_product += p_other.inner_product( p_self, **kwargs )
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
            ip_self[key] = p.inner_product( p, **kwargs )
            ips[key] = p.inner_product( other, **kwargs )
        ip_self = verdict.Dict( ip_self )
        ip_other = other.inner_product( other, **kwargs )
        ips = verdict.Dict( ips )

        # Cospsi
        cospsi = ips / ( ip_self * ip_other ).apply( np.sqrt )

        return cospsi

    ########################################################################
    # Plots
    ########################################################################

    def plot_cospsi2d(
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
