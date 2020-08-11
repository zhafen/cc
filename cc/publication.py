import ads
import bibtexparser
import nltk
import numpy as np
import warnings

import augment

from . import config
from . import relation
from . import utils

########################################################################

class Publication( object ):

    @augment.store_parameters
    def __init__(
        self,
        citation_key,
        notes_categories = [ 'key_concepts', 'key_points', 'uncategorized' ],
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
    # Data Retrieval
    ########################################################################

    def get_ads_data(
        self,
        fl = [ 'abstract', 'citation', 'reference' ],
        keep_query_open = False,
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
            warnings.warn(
                'Multiple papers found with identifying data {}'.format(
                    kwargs
                )
            )

        ads_data = query_list[0]

        self.ads_data = {}
        for key in fl:
            self.ads_data[key] = getattr( ads_data, key )

        if keep_query_open:
            self.ads_query = ads_query

        return self.ads_data

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

        # Retrieve full text
        self.full_text = []
        with open( tex_fp ) as f:
            for line in f:
                self.full_text.append( line )  

    ########################################################################
    # Publication Analysis
    ########################################################################

    def process_abstract(
        self,
        abstract_str = None,
        return_empty_upon_failure = True
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
        if hasattr( self, 'abstract' ):
            return

        # Load abstract if not given
        if abstract_str is None:

            if not hasattr( self, 'ads_data' ):

                # Search ADS using provided unique identifying keys
                identifying_keys = [ 'arxivid', 'doi' ]
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
                ads_not_loaded = not hasattr( self, 'ads_data' )
                if ads_not_loaded:
                    failure_msg = (
                        '''Unable to find arxiv ID or DOI for publication {}.\n
                        Not processing abstract.'''.format( self.citation_key )
                    )
                    if return_empty_upon_failure:
                        warnings.warn( failure_msg )
                        abstract_str = ''
                    else:
                        raise Exception( failure_msg )
                else:
                    abstract_str = self.ads_data['abstract']

        self.abstract = {
            'str': abstract_str,
        }

        # Parse using NLTK
        sentences = nltk.sent_tokenize( abstract_str )
        sentences = [nltk.word_tokenize(sent) for sent in sentences] 
        self.abstract['nltk'] = {}
        self.abstract['nltk']['all'] = [
            nltk.pos_tag(sent) for sent in sentences
        ] 

        # Classify into primary and secondary tiers, i.e. effectively
        # nouns, verbs, and adjectives vs everything else.
        self.abstract['nltk']['primary'] = []
        self.abstract['nltk']['secondary'] = []
        self.abstract['nltk']['primary_stemmed'] = []
        uncategorized = []
        tag_tier = config.nltk['tag_tier']
        for sent in self.abstract['nltk']['all']:
            nltk1 = []
            nltk2 = []
            for word, tag in sent:
                if tag in tag_tier[1]:
                    nltk1.append( word )
                elif tag in tag_tier[2]:
                    nltk2.append( word )
                else:
                    uncategorized.append( tag )
            self.abstract['nltk']['primary'].append( nltk1 )
            self.abstract['nltk']['secondary'].append( nltk2 )
            self.abstract['nltk']['primary_stemmed'].append(
                utils.stem( nltk1 )
            )
        self.abstract['nltk']['uncategorized'] = set( uncategorized )

    ########################################################################

    def process_bibtex_annotations(
        self,
        bibtex_fp = None,
        reload = False,
        word_per_concept = False,
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
        word_per_concept = False
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

            assert (
                ( line.count( '[' ) == line.count( ']' ) ),
                'Mismatch in number of brackets ([) for line {}'.format( line )
            )

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
    # Comparing to other publications
    ########################################################################
 
    def inner_product(
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
            return other.inner_product(
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

    def asymmetry_estimator( self, a, ):

        # Get the processed abstracts
        a.data.process_abstract()

        # Identify the unique concepts
        all_concepts = []
        for key, p in a.data.items():
            all_concepts += p.abstract['nltk']['primary_stemmed']
        all_concepts = set( all_concepts )

        # Calculate match
        a.data.inner_product(
            all_concepts,
            method = 'abstract similarity',
            other_type = 'basis vector',
        )
