import ads
import bibtexparser

import augment

from . import relation

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

    ########################################################################
    # Data Retrieval
    ########################################################################

    def get_ads_data( self, **kwargs ):
        '''Retrieve all data the NASA Astrophysical Data System has regarding
        a paper.

        NOTE: For this to work you MUST have your ADS API key
        saved to ~/.ads/dev_key

        Keyword Args:
            kwargs (str):
                Unique identifiers of the publication, e.g. the arXiv number
                with arxiv='1811.11753'.

        Returns:
            self.ads_data (ads.search.Article):
                Class containing ADS data.
        '''

        self.ads_query = ads.SearchQuery( **kwargs )
        query_list = list( self.ads_query )

        # Parse results of search
        if len( query_list ) < 1:
            raise Exception( 'No matching papers found in ADS' )
        elif len( query_list ) > 1:
            raise Exception( 'Multiple papers found with identifying data.' )

        self.ads_data = query_list[0]

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

    def process_bibtex_annotations( self, bibtex_fp=None, reload=False ):
        '''Process notes residing in a .bib file.

        Args:
            bibtex_fp (str):
                Filepath of the .bib file. Defaults to assuming one
                is already loaded.

            reload (bool):
                Forcibly reprocess annotations, even if already processed.

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
            return

        # Process the annotation
        annote_lines = annotation.split( '\n' )

        # Process the annotation
        for line in annote_lines:
            self.notes = self.process_annotation_line( line, self.notes )

        self.cached_bibtex_fp = bibtex_fp

    ########################################################################

    def process_annotation_line( self, line, notes=None ):
        '''Process a line of annotation to extract more information.

        Args:
            line (str):
                The line of annotation to process.

            notes (dict):
                The dictionary storing notes on various lines.
                Defaults to using self.notes.

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
            key_concepts = relation.parse_relation_for_key_concepts( line )
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
    # Comparing to other publications
    ########################################################################
 
    def inner_product(
        self,
        other
    ):

        pass
