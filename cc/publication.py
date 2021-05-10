import ads
import bibtexparser
import copy
import nltk
import numba
from numba.typed import List
import numpy as np
import pandas as pd
import warnings

import augment

from . import config
from . import relation
from . import tex
from . import utils

########################################################################

class Publication( object ):

    @augment.store_parameters
    def __init__(
        self,
        citation_key,
        notes_categories = [ 'key_concepts', 'key_points', 'uncategorized' ],
        ignore_failed = True,
    ):

        # Setup notes dictionary
        self.notes = {}
        for cat in self.notes_categories:
            self.notes[cat] = []

        # For recording what data has been retrieved
        self.cached_bibtex_fp = 'not processed'

    def __repr__( self ):

        return 'cc.publication.Publication:{}'.format( self.citation_key )

    def __str__( self ):

        return self.citation_key

    ########################################################################

    @property
    def publication_date( self ):

        if not hasattr( self, '_publication_date' ):

            try:
                self._publication_date = '{} {}'.format(
                    self.citation['month'],
                    self.citation['year']
                )
            except KeyError:
                self._publication_date = 'nan'
        
        return self._publication_date

    ########################################################################
    # Data Retrieval
    ########################################################################

    def get_ads_data(
        self,
        fl = [ 'abstract', 'citation', 'reference', 'entry_date' ],
        keep_query_open = False,
        verbose = False,
        **kwargs
    ):
        '''Retrieve all data the NASA Astrophysical Data System has regarding
        a paper.

        NOTE: For this to work you MUST have your ADS API key
        saved to ~/.ads/dev_key

        Keyword Args:
            fl (list of strs):
                Fields to preload when the data is first retrieved.

            kwargs:
                Unique identifiers of the publication, e.g. the arXiv number
                with arxiv='1811.11753'.

        Returns:
            self.ads_data (dict):
                Dictionary containing ADS data.
        '''

        ads_query = ads.SearchQuery( fl=fl, **kwargs )
        query_list = list( ads_query )

        # Parse results of search
        if len( query_list ) < 1:
            raise ValueError( 'No matching papers found in ADS' )
        elif len( query_list ) > 1:
            if verbose:
                warnings.warn(
                    'Multiple papers found with identifying data {}'.format(
                        kwargs
                    )
                )

        ads_data = query_list[0]

        # Store
        # Duplication is okay
        self.ads_data = {}
        for key in fl:
            value = getattr( ads_data, key )
            self.ads_data[key] = value
            if key == 'citation' or key =='reference':
                key += 's'
            setattr( self, key, value )

        if keep_query_open:
            self.ads_query = ads_query

        return ads_data

    ########################################################################

    def citations_per_year( self ):
        '''Calculate the citations per year. This depends on the instance
        having access to the ADS entry date and citation list, self.entry_date
        and self.citations. Note that this is usually automatically done
        after processing and saving the abstracts.
        '''

        if self.ignore_failed:
            if not hasattr( self, 'citations' ):
                return np.nan

            if self.citations is None:
                return 0.

        time_elapsed = (
            pd.to_datetime( 'now', ) - 
            pd.to_datetime( self.entry_date, ).tz_localize(None)
        )
        time_elapsed_years = time_elapsed.total_seconds() / 3.154e7

        citations_per_year = (
            len( self.citations ) / time_elapsed_years
        )

        return citations_per_year

    ########################################################################

    def read_citation( self, bibtex_fp, ):
        '''Retrieve a citation from a BibTex file.

        Args:
            bibtex_fp (str):
                Filepath containing the BibTex file to read from.

        Returns:
            self.citation (dict):
                A dictionary containing entries for a BibTex-style citation.
        '''

        # Load the database
        with open( bibtex_fp ) as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        # Find the relevant citation
        matches = []
        for citation in bib_database.entries:
            if citation['ID'] == self.citation_key:
                matches.append( citation )

        # Parse results of search
        if len( matches ) < 1:
            raise Exception( 'Citation not found in file.' )
        elif len( matches ) > 1:
            raise Exception( 'Duplicate entries found in file.' )

        # Store result
        self.citation = matches[0]

        return self.citation

    ########################################################################

    def load_full_tex( self, tex_fp ):
        '''Loads a tex file for further manipulation.

        Args:
            tex_fp (str):
                Location of tex file to load.
        '''

        # Load the main file
        self.full_tex = tex.Tex( filepath=tex_fp )
        full_tex_str = self.full_tex.string

        # Process into sections
        self.tex = {}
        stack = []
        bracket_stack = []
        section_labels = []
        is_appendix = False
        for i, c in enumerate( full_tex_str ):
            if c == '\\':
                # Extract abstract
                if full_tex_str[i:i+16] == '\\begin{abstract}':
                    stack.append( i+16 )
                elif full_tex_str[i:i+14] == '\\end{abstract}':
                    start = stack.pop()
                    self.tex['Abstract'] = tex.Tex( full_tex_str[start:i] )

                # Extract sections
                start_section = full_tex_str[i:i+9] == '\\section{'
                end_document = full_tex_str[i:i+14] == '\\end{document}'
                start_appendix = full_tex_str[i:i+9] == '\\appendix'
                if start_section or end_document or start_appendix:

                    # Finish previous section
                    if len( stack ) > 0:
                        start = stack.pop()
                        label = section_labels.pop()
                        tex_instance = tex.Tex( full_tex_str[start:i] )
                        if not is_appendix:
                            self.tex[label] = tex_instance
                        if is_appendix:
                            if 'Appendix' not in self.tex:
                                self.tex['Appendix'] = {}
                            self.tex['Appendix'][label] = tex_instance

                    # Start looking for section name
                    if start_section:
                        bracket_stack.append( i+9 )

                if start_appendix:
                    is_appendix = True

            elif c == '}' and len( bracket_stack ) > 0:

                # Extract section name
                start = bracket_stack.pop()
                section_labels.append( full_tex_str[start:i] )

                # Start recording section text
                stack.append( i+1 )

    ########################################################################
    # Publication Analysis
    ########################################################################

    def process_abstract(
        self,
        abstract_str = None,
        return_empty_upon_failure = True,
        verbose = False,
        overwrite = False,
    ):
        '''Process the abstract with natural language processing.

        Args:
            abstract_str (str):
                Raw abstract. If none, download from ADS.

             return_empty_upon_failure (bool):
                If True, treat the abstract as an empty string when failing to
                download the abstract from ADS.

        Modifies:
            self.abstract (dict):
                Parsed abstract data.
        '''

        # Don't parse the abstract if already parsed
        if hasattr( self, 'abstract' ) and not overwrite:
            return

        # Load abstract if not given
        if abstract_str is None:
            abstract_str = self.abstract_str(
                return_empty_upon_failure,
                verbose,
            )

        self.abstract = {
            'str': abstract_str,
        }

        if abstract_str is None:
            return

        self.abstract['nltk'] = utils.tokenize_and_sort_text( abstract_str )

    ########################################################################

    def abstract_str( self, return_empty_upon_failure=True, verbose=False ):
        '''Retrieve the abstract text, either from the citation or from ADS

        Args:
             return_empty_upon_failure (bool):
                If True, treat the abstract as an empty string when failing to
                download the abstract from ADS.

            verbose (bool):
                If True, say more about what's going down.

        Returns:
            abstract_str (str):
                String containing the abstract.
        '''

        def upon_failure():
            failure_msg = (
                '''Unable to find arxiv ID or DOI for publication {}.\n
                Not processing abstract.'''.format( self.citation_key )
            )
            if return_empty_upon_failure:
                if verbose:
                    warnings.warn( failure_msg )
                return ''
            else:
                raise Exception( failure_msg )

        # Try to obtain from a processed abstract
        if hasattr( self, 'abstract' ):
            if self.abstract is not None:
                return self.abstract['str']

        # Or try using the abstract in the citation
        if hasattr( self, 'citation' ):
            if 'abstract' in self.citation:
                abstract_str = self.citation['abstract']

            # If neither of those work, auto-retrieve ADS data
            else:
                if not hasattr( self, 'ads_data' ):

                    # Search ADS using provided unique identifying keys
                    identifying_keys = [ 'arxivid', ]
                    for key in identifying_keys:

                        # Try to get the data
                        if key in self.citation:
                            try:
                                self.get_ads_data( arxiv=self.citation[key] )
                            except ValueError:
                                continue

                        # Exit upon success
                        if hasattr( self, 'ads_data' ):
                            break

                # Behavior upon failure
                if not hasattr( self, 'ads_data' ):
                    return upon_failure()
                else:
                    abstract_str = self.ads_data['abstract']
        else:
            return upon_failure()

        return abstract_str

    ########################################################################

    def process_bibtex_annotations(
        self,
        bibtex_fp = None,
        reload = False,
        word_per_concept = True,
    ):
        '''Process notes residing in a .bib file.

        Args:
            bibtex_fp (str):
                Filepath of the .bib file. Defaults to assuming one
                is already loaded.

            reload (bool):
                Forcibly reprocess annotations, even if already processed.

            word_per_concept (bool):
                If True, break each concept into its composite words.

        Modifies:
            self.notes (dict):
                Dictionary containing processed bibtex annotations.
        '''

        # If already processed
        if not reload and self.cached_bibtex_fp == bibtex_fp:
            return

        try:
            # Load the data
            if bibtex_fp is None:
                annotation = self.citation['annote']
            else:
                self.read_citation( bibtex_fp )
                annotation = self.citation['annote']

        # When no annotation is found
        except KeyError:
            self.cached_bibtex_fp = bibtex_fp
            return

        # Process the annotation
        annote_lines = annotation.split( '\n' )

        # Process the annotation
        for line in annote_lines:
            self.notes = self.process_annotation_line(
            line,
            self.notes,
            word_per_concept
        )

        self.cached_bibtex_fp = bibtex_fp

    ########################################################################

    def process_annotation_line(
        self,
        line,
        notes = None,
        word_per_concept = True,
        conditions = None,
    ):
        '''Process a line of annotation to extract more information.

        Args:
            line (str):
                The line of annotation to process.

            notes (dict):
                The dictionary storing notes on various lines.
                Defaults to using self.notes.

            word_per_concept (bool):
                If True, break each concept into its composite words.

        Returns:
            notes (dict):
                A dictionary containing updates notes on annotations.
        '''

        if notes is None:
            notes = self.notes

        # Empty lines
        if line == '':
            return notes
        # Key lines
        elif '[' in line and ']' in line:

            # assert line.count( '[' ) == line.count( ']' )

            # Parse and store key concepts
            key_concepts = relation.parse_relation_for_key_concepts(
                line,
                word_per_concept
            )
            notes['key_concepts'].append( key_concepts )
            notes['key_concepts'] = [
                list( set( key_concepts ) )
                for key_concepts in
                notes['key_concepts']
            ]
            notes['key_points'].append( line )
        # For flags
        elif line[0] == '!':
            variable, value = line[1:].split( '=' )
            notes[variable] = value
        # Otherwise
        else:
            notes['uncategorized'].append( line )

        # Include any conditions
        # Many conditions are on a per-point basis, so this is not ideal,
        # but it works until we actually plan on using them.
        if conditions is not None:
            if 'conditions' not in notes:
                notes['conditions'] = {}
            notes['conditions'].update( conditions )

        return notes

    ########################################################################

    def identify_unique_key_concepts( self, **kwargs ):
        '''Identify the unique key concepts derived from notes.

        Args:
            **kwargs:
                Passed to concept.uniquify_concepts.
        '''

        assert self.cached_bibtex_fp != 'not processed'

        self.notes['unique_key_concepts'] = [
            utils.uniquify_words( _, **kwargs )
            for _ in self.notes['key_concepts']
        ]

    ########################################################################

    def points( self, verbose=False ):
        '''Return all currently-processed points.

        Args:
            verbose (bool):
                Be talkative or not?
        '''

        points = []

        # Points in the notes
        if hasattr( self, 'notes' ):
            points += self.notes['key_points']
            points += self.notes['uncategorized']

        # Points in the abstract
        points += nltk.sent_tokenize(
            self.abstract_str(
                return_empty_upon_failure=True,
                verbose = verbose,
            )
        )

        return points

    ########################################################################

    def concept_projection( self, component_concepts=None, include_notes=True ):
        '''Project the abstract into concept space.
        In simplest form this can just be counting up the number of
        times each unique, stemmed noun, verb, or adjective shows up in the
        abstract.

        Args:
            component_concepts (array-like of strs):
                Basis concepts to project onto. Defaults to all concepts in
                the abstract.

            include_notes (bool):
                If True include key_points and uncategorized in the concept projection.

        Returns:
            components (np.ndarray of ints):
                The value of the projection for each concept (by default the
                number of times a word shows up in the abstract).

            component_concepts (np.ndarray of strs):
                The component concepts. If the component_concepts argument
                is None then all non-zero concepts in the abstract are used.
                If not None then the union of the non-zero concepts and
                the component_concepts arg.
        '''

        sents = []

        # Points in the notes
        if hasattr( self, 'notes' ) and include_notes:
            notes_str = ' '.join( self.notes['key_points'] )
            notes_str += ' '.join( self.notes['uncategorized'] )
            sents += utils.tokenize_and_sort_text( notes_str )['primary_stemmed']

        # Get the processed abstracts
        self.process_abstract()

        # When the abstract failed to retrieve
        def upon_failure():
            if component_concepts is None:
                return [], None
            else:
                values = np.zeros( len( component_concepts ) )
                return values, component_concepts

        if 'nltk' not in self.abstract: return upon_failure()

        # Project for non-zero concepts
        sents += self.abstract['nltk']['primary_stemmed']
        if len( sents ) == 0: return upon_failure()
        flattened = np.hstack( sents )
        nonzero_concepts, values = np.unique( flattened, return_counts=True )

        # Combine with existing component concepts
        if component_concepts is not None:
            @numba.njit
            def combine(
                original_concepts,
                added_components,
                added_concepts,
            ):
                # Store the concepts shared with other publications
                dup_inds = List()
                components = List()
                component_concepts = List( original_concepts )
                for i, ci in enumerate( original_concepts ):
                    no_match = True
                    for j, cj in enumerate( added_concepts ):
                        # If a match is found
                        if ci == cj:
                            components.append( added_components[j] )
                            dup_inds.append( j )
                            no_match = False
                            break
                    # If made to the end of the loop with no match
                    if no_match:
                        components.append( 0 )

                # Finish combining
                for i in range( len( added_concepts ) ):
                    if i in dup_inds:
                        continue
                    components.append( added_components[i] )
                    component_concepts.append( added_concepts[i] )

                return components, component_concepts

            components, component_concepts = combine(
                original_concepts = List( component_concepts ),
                added_components = List( values ),
                added_concepts = List( nonzero_concepts ),
            )
                
        else:
            components = values
            component_concepts = nonzero_concepts

        return components, component_concepts

    ########################################################################
    # Comparing to other publications
    ########################################################################
 
    def inner_product_custom(
        self,
        other,
        method = 'cached key-point concepts',
        max_edit_distance = None,
        **kwargs
    ):
        '''Calculate the inner product between the publication and another
        object.

        Args:
            other:
                The other object to calcualte the inner product with.

            method (str):
                How to calculate the inner product. Options are...
                'key-point concepts':
                    The inner product is the the sum of the inner products
                    between all relations in the two publications, and the
                    inner product between two relations is the number of
                    key concepts they share.

                'cached key-point concepts':
                    Faster but less-accurate version of 'key-point concepts'.

                'abstract similarity':
                    The inner product is the sum of the inner product between
                    all sentences in the abstract of each publications. The
                    inner product between two abstract sentences is the number
                    of nouns, verbs, and adjectives ("important" words)
                    shared by the two sentences.
                    ("Important" is up to user choice. I've defined it in the
                    config using nltk['tag_tier'].)

                max_edit_distance (int):
                    If not None this is the Maximum Levenshtein edit-distance
                    between two words for them to count as the same word.
                    Computationally expensive currently.

            **kwargs:
                Passed to the inner product between relations.
        '''

        # Check that we can calculate the inner product
        # isinstance raises false exceptions
        is_pub = (
            isinstance( other, Publication ) or
            str( type( other ) ) == "<class 'cc.publication.Publication'>"
        )
        if is_pub:
            pass
        elif str( type( other ) ) == "<class 'cc.atlas.Atlas'>":
            return other.inner_product_custom(
                self,
                method = method,
                max_edit_distance = max_edit_distance,
                **kwargs
            )
        else:
            raise Exception( "Incompatible object for calculating the inner product with, {}.".format( other ) )

        inner_product = 0

        if method == 'key-point concepts':
            for point in self.notes['key_points']:
                for other_point in other.notes['key_points']:
                    inner_product += relation.inner_product(
                        point,
                        other_point,
                        max_edit_distance = max_edit_distance,
                        **kwargs
                    )

        elif method == 'cached key-point concepts':

            if 'unique_key_concepts' not in self.notes:
                raise KeyError( 'Unique key concepts are not calculated for publication {}'.format( self.citation_key ) )
            if 'unique_key_concepts' not in other.notes:
                raise KeyError( 'Unique key concepts are not calculated for publication {}'.format( other.citation_key ) )

            for point_i_kcs in self.notes['unique_key_concepts']:
                for point_j_kcs in other.notes['unique_key_concepts']:

                        matching_concepts = utils.match_words(
                            point_i_kcs,
                            point_j_kcs,
                            max_edit_distance = max_edit_distance,
                            **kwargs
                        )
                        inner_product += len( matching_concepts )

        elif method == 'abstract similarity':

            # Get the processed abstracts
            self.process_abstract()
            other.process_abstract()

            # When the abstract failed to retrieve
            no_nltk = 'nltk' not in self.abstract
            other_no_nltk = 'nltk' not in other.abstract
            if no_nltk or other_no_nltk:
                return 0.

            sents = self.abstract['nltk']['primary_stemmed']
            sents_other = other.abstract['nltk']['primary_stemmed']

            # Calculate the inner product
            for sent in sents:
                for sent_other in sents_other:

                    matching_words = utils.match_words(
                        sent,
                        sent_other,
                        stemmed = True,
                        max_edit_distance = max_edit_distance,
                        **kwargs
                    )
                    inner_product += len( matching_words )

        else:
            raise Exception( 'Unrecognized inner_product method, {}'.format( method ) )

        return inner_product

########################################################################

class UnofficialPublication( Publication ):

    @augment.store_parameters
    def __init__(self, citation_key, strictness='warn', **kwargs ):
        '''
        Args:
            strictness (str):
                How strict to be with operations that don't make sense for
                an unofficial publication. Options are permit, warn, or raise.

        *args, **kwargs:
            Passed to parent class.
        '''
        super().__init__( citation_key=citation_key, **kwargs )

        assert strictness in [ 'permit', 'warn', 'raise' ]

        self.unofficial_flag = True

    def __repr__( self ):

        return 'cc.publication.UnofficialPublication:{}'.format(
            self.citation_key
        )

    ########################################################################

    @property
    def publication_date( self ):

        warning_msg = '{} is unofficial and has no publication date.'.format(
            self.citation_key,
        )
        if self.strictness == 'warn':
            warnings.warn( warning_msg )
        elif self.strictness == 'raise':
            raise UnofficialPublicationException( warning_msg )
        else:
            pass

        return 'NaT'


class UnofficialPublicationException( Exception ):

    def __init__( self, message="Unofficial Publication is not compatible with this action." ):

        self.message = message
        super().__init__( self.message )