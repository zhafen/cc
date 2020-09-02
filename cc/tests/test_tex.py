from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pytest
import unittest

import cc.tex

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
