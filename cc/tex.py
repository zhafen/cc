import nltk
import os
import palettable

import matplotlib.pyplot as plt

import augment

from . import config
from . import utils

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
        commands = {},
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
        statement_str = ''
        stack = []
        fd = os.path.dirname( filepath )
        for i, c in enumerate( string ):

            # Search for include statements
            if c == '\\':
                if string[i:i+9] == '\\include{':
                    stack.append( i+9 )
                if string[i:i+7] == '\\input{':
                    stack.append( i+7 )

            # Load include files
            if len( stack ) > 0:

                statement_str += c

                if c == '}':
                    start = stack.pop()
                    fn = string[start:i] + '.tex'
                    fp = os.path.join( fd, fn )
                    try:
                        with open( fp ) as f:
                            new_string += f.read()

                    # For missing files just keep the command
                    except FileNotFoundError:
                        new_string += statement_str
                        statement_str = ''

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
        '''Clean up the text, i.e. remove comments, handle "~" characters.

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
            if c == '%' and string[i-1] != '\\':
                stack.append( i+1 )

            if len( stack ) > 0:
                # Find comment ends and append
                if c == '\n':
                    start = stack.pop()
                    comments.append( string[start:i] )
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

    def interpret( self, string ):
        '''Interpret the LaTeX, i.e. process macros, and some commands and
        characters.
        '''

        # We will use this to skip parts of the string we're modifying.
        # Initialize to a value that won't cause accidental skipping
        j = -1

        interpreted = ''
        stack = []
        for i, c in enumerate( string ):

            # Skip characters that have been removed and replaced
            if i < j:
                continue

            # Locate possible commands
            if c == '\\':
                stack.append( i+1 )

            # Handle the ~ character
            if c == '~' and string[i-1] != '\\':
                c = ' '

            if len( stack ) > 0:

                # Don't quit on the stack we just started
                if i+1 == stack[-1]:
                    continue

                # Start the command interpretation
                if not c.isalpha():

                    # Split command from input tex
                    # Input tex is the entire remaining document. The commands
                    # will parse when the input ends
                    start = stack.pop()
                    command = string[start:i]
                    subsequent_tex = string[i:]

                    command_tex, dj, self.commands = interpret(
                        command,
                        subsequent_tex,
                        commands = self.commands,
                    )

                    # Store results
                    interpreted += command_tex
                    j = i + dj

                    # Edge case: the command has no arguments,
                    # so c should be stored
                    if i >= j:
                        interpreted += c

            else:
                interpreted += c

        return interpreted

    @property
    def interpreted( self ):
        '''String with comments removed.
        '''

        if not hasattr( self, '_interpreted' ):

            self._interpreted = self.interpret( self.cleaned )

        return self._interpreted

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

            self._words = word_tokenize( self.cleaned )

        return self._words

    ########################################################################

    @property
    def tokens( self ):

        if not hasattr( self, '_tokens' ):

            self._tokens = nltk.pos_tag( self.words )

        return self._tokens

    ########################################################################

    @property
    def sentence_words( self ):

        if not hasattr( self, '_sentence_words' ):

            result = []
            for sent in self.sentences:
                words = word_tokenize( sent )
                tokens = nltk.pos_tag( words )
                result.append( tokens )
            self._sentence_words = result
                    
        return self._sentence_words

    ########################################################################

    def tier_chunk( self, tagged_words ):
        '''Chunk the sentence according to defined "tiers" of relevance.
        '''

        tier_chunks = []
        tiers = []
        for sent in tagged_words:

            # Sort and tag according to sentences
            sent_chunks = []
            sent_tiers = []
            current = []
            for i, (w, tag) in enumerate( sent ):

                # Find the word tier
                for tier, tags in config.nltk['tag_tier'].items():
                    if tag in tags:
                        current_tier = tier
                # Edge case
                if w == 'is':
                    current_tier = 2

                # Special case
                if i == 0:
                    prev_tier = current_tier


                # Store chunk
                if current_tier != prev_tier:
                    sent_chunks.append( current )
                    current = []
                    sent_tiers.append( prev_tier )

                # Setup next loop
                current.append( w )
                prev_tier = current_tier

                # Last loop edge case
                if i == len( sent ) - 1:
                    sent_chunks.append( current )
                    sent_tiers.append( prev_tier )

            tier_chunks.append( sent_chunks )
            tiers.append( sent_tiers )

        return tier_chunks, tiers

    @property
    def tier_chunks( self ):

        if not hasattr( self, '_tier_chunks' ):
            self._tier_chunks, self._tiers = self.tier_chunk(
                self.sentence_words,
            )
                    
        return self._tier_chunks

    @property
    def tiers( self ):

        if not hasattr( self, '_tiers' ):
            self._tier_chunks, self._tiers = self.tier_chunk(
                self.sentence_words,
            )
                    
        return self._tiers
    
    ########################################################################

    @property
    def ne_chunks( self ):

        if not hasattr( self, '_ne_chunks' ):

            result = []
            for sent in self.sentences:
                words = word_tokenize( sent )
                tokens = nltk.pos_tag( words )
                chunks = nltk.ne_chunk( tokens )
                result.append( chunks )
            self._ne_chunks = result
                    
        return self._ne_chunks

    ########################################################################

    def display_chunked_sentence(
        self,
        i,
        x = 0, y = 1,
        tier1_colors = palettable.cartocolors.qualitative.Antique_10.mpl_colors,
        default_color = '0.7',
        fontsize = 24,
        tier_fontweights = { 1: 'bold', 2: None },
        max_word_per_line = 10,
        **kwargs
    ):

        sentence = self.tier_chunks[i]
        tiers = self.tiers[i]

        # Loop through and assign colors
        strings = []
        word_colors = []
        fontweights = []
        j = 0
        for i, chunk in enumerate( sentence ):

            # Tier 1 cycles through the colors
            if tiers[i] == 1:
                color = tier1_colors[j]
                j += 1
                if j > len( tier1_colors ) - 1:
                    j = 0

            # Other tiers are black
            else:
                color = default_color
            strings += chunk
            word_colors += [ color, ] * len( chunk )

            # Fontweights
            fontweights += [ tier_fontweights[tiers[i]], ] * len( chunk )

        utils.multicolor_text(
            x,
            y,
            strings,
            word_colors,
            fontsize = fontsize,
            fontweights = fontweights,
            annotated = True,
            **kwargs
        )

        plt.axis( 'off' )

########################################################################

def word_tokenize( sent, **kwargs ):
    '''Word tokenize that accounts for math-mode in LaTex, producing more
    comprehendible results.
    * All math mode inside one set of "$"s is grouped into one word.
    * All escaped "\\$"s outside mathmode are replaced with "$"s.

    Args:
        sent (str): Sentence to Tokenize

        **kwargs: Passed to nltk.tokenize.word_tokenize

    Returns:
        words (list of strs): Word tokenized sentence.
    '''

    nltk_words = nltk.tokenize.word_tokenize( sent, **kwargs )

    # Account for Math Mode
    words = []
    stack = ''
    for i, w in enumerate( nltk_words ):

        escaped = (
            i > 0 and
            ( nltk_words[i-1] == '\\' or nltk_words[i-1][-1] == '\\' )
        )

        # Search for LaTex
        if w == '$' and not escaped:

            # Start the stack
            stack += w

            # Finish the stack off
            if len( stack ) > 1:
                words.append( stack )
                stack = ''

        else:
            # Build the stack if started
            if len( stack ) != 0:
                if w == '{' or escaped:
                    seperator = ''
                else:
                    seperator = ' '
                stack += seperator + w
            # If not in the middle of latex, just append
            else:
                if not escaped:
                    words.append( w )
                else:
                    # Remove slash and place $
                    words[-1] = words[-1][:-1] + w

    return words

########################################################################

def interpret( command, subsequent_tex, commands={}, **kwargs ):

    # Check if the command is one we can interpret
    if command in globals():
        command_tex, i, new_commands = globals()[command]( subsequent_tex )
    elif command in commands:
        command_tex, i, new_commands = commands[command]( subsequent_tex )
    # If the command cannot be interpreted stick it back into the string
    else:
        command_tex = '\\' + command
        i = 0
        new_commands = {}

    # Store any new commands created
    commands.update( new_commands )

    return command_tex, i, new_commands

########################################################################

def newcommand( subsequent_tex ):

    command_name = ''
    output = ''
    bracket_count = 0
    for i, c in enumerate( subsequent_tex ):
        
        # Process command
        if c == '{' or c == '}' and subsequent_tex[i-1] != '\\':
            bracket_count += 1
            continue

        # Command name
        if bracket_count == 1:
            
            # Don't include the \
            if command_name == '' and c == '\\':
                continue

            command_name += c

        elif bracket_count == 2:
            pass
        elif bracket_count == 3:
            output += c
        elif bracket_count == 4:
            # The command should be ended
            assert not subsequent_tex[i+1].isalpha()
            break

    # The command text and the arguments are removed
    command_tex = ''

    # Create the new command
    def command_fn( tex_in ):
        '''Macros always return this output, regardless of what else follows.'''
        return output, 0, {}

    new_commands = { command_name: command_fn }

    return command_tex, i, new_commands

        

        

















































































    
