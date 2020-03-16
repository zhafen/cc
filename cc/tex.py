import augment

########################################################################

class Paper( object ):

    @augment.store_parameters
    def __init__( self, filepath ):

        # Retrieve full text
        self.full_text = []
        with open( filepath ) as f:
            for line in f:
                self.full_text.append( line )  
