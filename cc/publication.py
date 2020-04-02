import augment

########################################################################

class Publication( object ):

    @augment.store_parameters
    def __init__( self, citation_key ):

        pass

    ########################################################################

    def load_full_tex( self, filepath ):
        '''Loads a tex file for further manipulation.

        Args:
            filepath (str):
                Location of tex file to load.
        '''

        # Retrieve full text
        self.full_text = []
        with open( filepath ) as f:
            for line in f:
                self.full_text.append( line )  
