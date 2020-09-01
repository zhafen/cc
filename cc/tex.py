import nltk
import os

import augment

########################################################################

class Tex( object ):
    '''Class for parsing tex, without regard to context.
    Can load a main tex file and all included files.
    '''

    @augment.store_parameters
    def __init__(
        self,
        string = None,
        filepath = None,
    ):

        # Accept either a string or a filepath
        if string is not None:
            assert filepath is None
            return
        if filepath is not None:
            assert string is None

        # Retrieve text
        with open( filepath ) as f:
            string = f.read()

        # Handle include statements
        new_string = ''
        stack = []
        fd = os.path.dirname( filepath )
        for i, c in enumerate( string ):

            # Search for include statements
            if c == '\\':
                if string[i:i+9] == '\\include{':
                    stack.append( i+9 )

            # Load include files
            if len( stack ) > 0:
                if c == '}':
                    start = stack.pop()
                    fn = string[start:i] + '.tex'
                    fp = os.path.join( fd, fn )
                    with open( fp ) as f:
                        new_string += f.read()

            # Otherwise
            else:
                new_string += c
        self.string = new_string

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

    @property
    def words( self ):

        if not hasattr( self, '_words' ):

            self._words = nltk.tokenize.word_tokenize( self.cleaned )

        return self._words

    ########################################################################

    @property
    def tokens( self ):

        if not hasattr( self, '_tokens' ):

            self._tokens = nltk.pos_tag( self.words )

        return self._tokens


