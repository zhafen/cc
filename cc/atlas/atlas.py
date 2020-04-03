import bibtexparser

import augment
import verdict

from .. import publication

########################################################################

class Atlas( object ):

    @augment.store_parameters
    def __init__( self, atlas_dir ):
        
        self.data = verdict.Dict( {} )

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


