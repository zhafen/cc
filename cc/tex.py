import nltk

import augment

########################################################################

class Tex( object ):
    '''Class for parsing chunks of tex, without regard to context.
    '''

    @augment.store_parameters
    def __init__(
        self,
        string,
    ):

        pass

    def __repr__( self ):

        return 'cc.tex.Tex'

    def __str__( self ):

        return self.string

    ########################################################################

    def clean( self, string ):
        '''Clean up the text, i.e. remove comments.

        Args:
            string (str):
                The string to clean.

        Returns:
            cleaned (str):
                The string once cleaned.

            comments (list of strs):
                A list of any removed comments.
        '''

        cleaned = ''
        comments = []
        stack = []
        for i, c in enumerate( string ):
            # Find comment starts
            if c == '%':
                stack.append( i )
            if len( stack ) > 0:
                # Find comment ends and append
                if c == '\n':
                    start = stack.pop()
                    comments.append( string[start+1:i] )
            else:
                cleaned += c

        return cleaned, comments
    
    @property
    def cleaned( self ):
        '''String with comments removed.
        '''

        if not hasattr( self, '_cleaned' ):

            self._cleaned, self._comments = self.clean( self.string )

        return self._cleaned

    @property
    def comments( self ):
        '''Comments found in the tex
        '''

        if not hasattr( self, '_comments' ):

            self._cleaned, self._comments = self.clean( self.string )

        return self._comments

    ########################################################################

    @property
    def sentences( self ):

        if not hasattr( self, '_sentences' ):

            self._sentences = nltk.tokenize.sent_tokenize( self.cleaned )
                    
        return self._sentences

    ########################################################################
