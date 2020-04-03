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

        # Load the data
        if bibtex_fp is None:
            annotation = self.citation['annote']
        else:
            self.read_citation( bibtex_fp )
            annotation = self.citation['annote']

        # Process the annotation
        self.key_points = annotation.split( '\n' )

    ########################################################################

    def process_annotation( self, point, notes={} ):

        ### Parse key concepts
        # Find boundaries
        def find_char_inds( char ):
            inds = []
            i = 0
            while True:
                ind = point.find( char, i )
                if ind == -1:
                    break
                inds.append( ind )
                i = ind + 1
            return inds
        starts = find_char_inds( '[' )
        ends = find_char_inds( ']' )
        # Find nested ends
        pair_inds = []
        for i in range( len( starts ) - 1 ):
            j = i
            while ends[j] < starts[i+1]:
                j += 1
            pair_inds.append( ( starts[i], ends[j] ) )
        # Store
        key_concepts = []
        for start, end in zip( *[ starts, ends] ):
            key_concepts.append( point[start+1:end] )
        if 'key_concepts' not in notes:
            notes['key_concepts'] = key_concepts
        else:
            notes['key_concepts'] += key_concepts

        # Store as a key point if recognized as one
        if len( starts ) > 0 and len( ends ) > 0:
            if 'key_points' not in notes:
                notes['key_points'] = [ point, ]
            else:
                notes['key_points'].append( point )
            
        return notes
