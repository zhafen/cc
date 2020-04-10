import bibtexparser
from collections import Counter
import glob
import nltk
from nltk.metrics import edit_distance
import numpy as np
import os

import augment
import verdict

from . import publication

########################################################################

class Atlas( object ):

    @augment.store_parameters
    def __init__( self, atlas_dir, bibtex_fp=None ):
        
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
        self.import_bibtex(  bibtex_fp )

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

    def get_unique_key_concepts( self, max_edit_distance=2, ):
        '''Unique key concepts, as simplified using nltk tools.
        Steps to retrieve unique key concepts:
        1. Union of the same stems.
        2. Concepts with a sufficiently low edit distance
           (accounts for mispellings)

        Args:
            max_edit_distance (int):
                Maximum Levenshtein edit-distance between two concepts for them
                to count as the same concept.
        '''

        l = list( self.all_key_concepts )

        # First pass through with a stemmer
        s = nltk.stem.SnowballStemmer( 'english' )
        ukcs = []
        for kc in l:
            words = nltk.word_tokenize( kc )
            stemmed_words = [ s.stem( w ) for w in words ]
            ukcs.append( ' '.join( stemmed_words ) )
        ukcs = set( ukcs )

        # Look for concepts with a sufficiently low Levenshtein edit-distance
        ukcs_arr = np.array( list( ukcs ) )
        ukcs_ed = []
        for kc in ukcs_arr:
            # Find matches
            edit_distances = np.array([
                edit_distance( kc, kc_check, )
                for kc_check
                in ukcs_arr
            ])
            matches = ukcs_arr[edit_distances<=max_edit_distance]
            # Count the matches and represent by maximum count
            count = Counter( matches )
            true_kc = verdict.Dict( count ).keymax()[0]
            
            # Store
            ukcs_ed.append( true_kc )
        ukcs = set( ukcs_ed )

        self.unique_key_concepts = ukcs

        return ukcs

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
