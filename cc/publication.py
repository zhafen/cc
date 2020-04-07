import ads
import bibtexparser

import augment

########################################################################

class Publication( object ):

    @augment.store_parameters
    def __init__( self, citation_key ):

        pass

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

    def process_bibtex_annotations( self, bibtex_fp=None ):
        '''Process notes residing in a .bib file.

        Args:
            bibtex_fp (str):
                Filepath of the .bib file. Defaults to assuming one
                is already loaded.

        Modifies:
            self.notes (dict):
                Dictionary containing processed bibtex annotations.
        '''

        # Load the data
        if bibtex_fp is None:
            annotation = self.citation['annote']
        else:
            self.read_citation( bibtex_fp )
            annotation = self.citation['annote']

        # Process the annotation
        annote_lines = annotation.split( '\n' )

        # Process the annotation
        self.notes = {}
        for line in annote_lines:
            self.notes = self.process_annotation_line( line, self.notes )

    ########################################################################

    def process_annotation_line( self, line, notes={} ):
        '''Process a line of annotation to extract more information.

        Args:
            line (str):
                The line of annotation to process.

            notes (dict):
                The dictionary storing notes on various lines.

        Returns:
            notes (dict):
                A dictionary containing updates notes on annotations.
        '''

        # Empty lines
        if line == '':
            return notes
        # Key lines
        elif '[' in line and ']' in line:

            assert (
                line.count( '[' ) == line.count( ']' ),
                'Mismatch in number of brackets ([) for line {}'.format( line )
            )

            # Parse key concepts, including nested brackets
            key_concepts = []
            stack = []
            for i, char in enumerate( line ):
                if char == '[':
                    stack.append( i )
                elif char == ']' and stack:
                    start = stack.pop()
                    key_concept = line[start+1:i]
                    key_concept = key_concept.replace( '[', '' )
                    key_concept = key_concept.replace( ']', '' )
                    key_concepts.append( key_concept )
            # Store
            if 'key_concepts' not in notes:
                notes['key_concepts'] = key_concepts
            else:
                notes['key_concepts'] += key_concepts
            notes['key_concepts'] = list( set( notes['key_concepts'] ) )
            if 'key_points' not in notes:
                notes['key_points'] = [ line, ]
            else:
                notes['key_points'].append( line )
        # For flags
        elif line[0] == '!':
            variable, value = line[1:].split( '=' )
            notes[variable] = value
        # Otherwise
        else:
            if 'uncategorized' not in notes:
                notes['uncategorized'] = [ line, ]
            else:
                notes['uncategorized'].append( line )

        return notes
