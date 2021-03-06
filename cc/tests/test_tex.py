from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pytest
import unittest

import cc.tex
import cc.tex_commands as tex_commands

########################################################################

class TestParse( unittest.TestCase ):

    def setUp( self ):

        s = '''% Broad intro
The circumgalactic medium (CGM) of galaxies is inferred to contain a baryonic mass comparable to or in excess of the galaxy mass~\\citep[e.g.][]{Werk2014,Tumlinson2017}.        This large reservoir of gas, loosely defined as the gas immediately outside the galaxy but inside the dark matter halo, interfaces strongly with the galaxy:
accretion onto the galaxy is necessary to sustain galaxy growth over the age of the Universe \\citep[e.g.][]{Prochaska2009, Bauermeister2010},
and in turn material from the galaxy returns to the CGM in the form of galactic winds driven by stellar and AGN feedback \\citep[e.g.,][]{Heckman2000, Steidel2010, Jones2012, Rubin2014, Cicone2014}.
For recent reviews of the CGM see \\cite{Putman2012}, \\cite{Tumlinson2017}, and~\\cite{Fox2017}.

% Simulations
Building a theoretical framework for the CGM requires accurately modeling both galaxies and their larger environment.
One way to approach this problem is through cosmological hydrodynamic simulations of galaxy formation~\\citep[e.g.][]{Somerville2015}.
These simulations calculate the evolution of dark matter, gas, and stars according to the relevant physics (e.g. gravity, hydrodynamics, star formation, feedback, etc.). 
Cosmological galaxy formation simulations have been used to understand the CGM in a variety of ways, from analyses of the dynamics of the CGM in simulations~\\citep[e.g.][]{Keres2005, Keres2009, Faucher-Giguere2011a, 2011MNRAS.414.2458V, Nelson2013, Oppenheimer2010, Wetzel2015, Oppenheimer2018} to those that use simulations to provide context to observations~\\citep[e.g.][]{Faucher-Giguere2010, Faucher-Giguere2011,Hummels2013,Liang2015,Corlies2016, 2016MNRAS.462.2440T, Gutcke2017, Nelson2017, Roca-Fabrega2018}.
'''

        self.tex = cc.tex.Tex( s )

    ########################################################################

    def test_sentences( self ):

        # Sentences
        actual = self.tex.sentences[0]
        expected = 'The circumgalactic medium (CGM) of galaxies is inferred to contain a baryonic mass comparable to or in excess of the galaxy mass \\citep[e.g.][]{Werk2014,Tumlinson2017}.'
        assert actual == expected

        # Comments
        actual = self.tex.comments
        expected = [ ' Broad intro', ' Simulations' ]
        assert actual == expected

    ########################################################################

    @pytest.mark.onhold
    def test_tier_chunks( self ):

        # Chunked
        actual = self.tex.tier_chunks[0]
        expected = [
            [ 'The', ],
            [ 'circumgalactic', 'medium', ],
            [ '(', ],
            [ 'CGM', ],
            [ ')', 'of', ],
            [ 'galaxies', ],
            [ 'is', ],
            [ 'inferred', ],
            [ 'to', ],
            [ 'contain', ],
            [ 'a', ],
            [ 'baryonic', 'mass', 'comparable', ],
            [ 'to', 'or', 'in', ],
            [ 'excess', ],
            [ 'of', 'the', ],
            [ 'galaxy', 'mass', ],
            [ '\\citep[e.g.][]{Werk2014,Tumlinson2017}', '.', ],
        ]
        assert actual == expected

        # Tiers
        actual = self.tex.tiers[0]
        expected = [ 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 ]
        assert actual == expected

########################################################################

class TestHandleCommands( unittest.TestCase ):

    def test_simple( self ):

        # Create object
        string = '''
        \\newcommand{\DM}{D_M}

        A value of $\\DM = 5$.
        '''
        tex = cc.tex.Tex( string )

        # Test cleaned
        actual = tex.interpreted
        expected = '''
        

        A value of $D_M = 5$.
        '''
        assert actual == expected

    ########################################################################

    def test_arg( self ):

        # Create object
        string = '''
        \\newcommand{\\rr}[1]{\\textcolor{red}{(\\bf #1)}}

        This is \\rr{wrong}!
        '''
        tex = cc.tex.Tex( string )

        # Test cleaned
        actual = tex.interpreted
        expected = '''
        

        This is \\textcolor{red}{(\\bf wrong)}!
        '''
        assert actual == expected

    ########################################################################

    def test_arg2( self ):

        # Create object
        string = '''
        \\newcommand{\\rr}[1]{\\textcolor{red}{(\\bf #1)}}

        This is \\rr1!
        '''
        tex = cc.tex.Tex( string )

        # Test cleaned
        actual = tex.interpreted
        expected = '''
        

        This is \\textcolor{red}{(\\bf 1)}!
        '''
        assert actual == expected

    ########################################################################

    @pytest.mark.onhold
    def test_def( self ):

        # Create object
        string = '''
        \\def\\simlt{\\stackrel{<}{{}_\\sim}}

        $1 \\simlt 2$
        '''
        tex = cc.tex.Tex( string )

        # Test cleaned
        actual = tex.interpreted
        expected = '''
        

        $1 \\stackrel{<}{{}_\\sim} 2$
        '''
        assert actual == expected

########################################################################

@pytest.mark.onhold
class ParseArgs( unittest.TestCase ):
    '''One of the main issues in the previous tests is how LaTeX is parsing
    args. These tests make sure we can parse args successfully for LaTeX
    commands. As of writing all fail because parse_args does not exist...
    '''

    def test_simple( self ):

        string = '{abc} '
        args, subsequent_tex = tex_commands.parse_args( string )
        assert args == [ 'abc', ]
        assert subsequent_tex == ' '

    ########################################################################

    def test_multiple_args( self ):

        string = '{abc}{def} '
        args, subsequent_tex = tex_commands.parse_args( string )
        assert args == [ 'abc', 'def' ]
        assert subsequent_tex == ' '

        string = '{abc}{def}{ghi} '
        args, subsequent_tex = tex_commands.parse_args( string )
        assert args == [ 'abc', 'def', 'ghi' ]
        assert subsequent_tex == ' '

    ########################################################################

    def test_no_brackets( self ):

        string = '\\urgh{whynobrackets}?'
        args, subsequent_tex = tex_commands.parse_args( string )
        assert args == [ 'urgh', 'whynobrackets', ]
        assert subsequent_tex == '?'

    ########################################################################

    def test_complicated( self ):

        string = '\\are{U}serious{mate}?'
        args, subsequent_tex = tex_commands.parse_args( string )
        assert args == [ 'are', 'U', 'serious', 'mate' ]
        assert subsequent_tex == '?'

########################################################################

class TestWordTokenize( unittest.TestCase ):

    def test_mathmode( self ):

        sent = 'A mass of $M_\\rm{ h } \\sim 10^7M_\\odot$ or so'
        actual = cc.tex.word_tokenize( sent )
        expected = [
            'A',
            'mass',
            'of',
            '$ M_\\rm{ h } \\sim 10^7M_\\odot$',
            'or',
            'so',
        ]
        assert actual == expected

    ########################################################################

    def test_mathmode_escaped_characters( self ):

        sent = '$\\$ $ = 1\\$ = $1^2\\$ = 1^3 \\$ $'
        actual = cc.tex.word_tokenize( sent )
        expected = [
            '$ \\$$',
            '=',
            '1$',
            '=',
            '$ 1^2\\$ = 1^3 \\$$',
        ]
        assert actual == expected

    ########################################################################

    @pytest.mark.onhold
    def test_mathmode_no_space( self ):
        '''Word tokenize may fail with cases like that below.
        This is on hold because it requires locating the spaces that were
        stripped off during the NLTK step.
        '''

        sent = 'deg$^2$ deg $^2$'
        actual = cc.tex.word_tokenize( sent )
        expected = [
            'deg$^2$',
            'deg',
            '$^2$',
        ]
        assert actual == expected
