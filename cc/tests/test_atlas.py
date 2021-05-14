import copy
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pandas as pd
import pytest
import shutil
import unittest

import cc.atlas as atlas
import cc.publication as publication

import verdict

########################################################################

filepath = './tests/data/arxiv_source/Hafen2019/CGM_origin.tex'

class TestBibTexData( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_bibtex( self ):

        self.a.import_bibtex( './tests/data/example_atlas/example.bib' )

        assert self.a.data['Hafen2019'].citation['eprint'] == '1811.11753'

    ########################################################################

    def test_bibtex_zotero( self ):

        self.a.import_bibtex( './tests/data/example_atlas_zotero/example.bib' )

        assert self.a.data['Hafen2019'].citation['eprint'] == '1811.11753'

    ########################################################################

    def test_process_bibtex_anotations( self ):

        self.a.data.process_bibtex_annotations()

        before = copy.deepcopy( self.a.data['Hafen2019'].notes )

        # Make sure that it caches
        self.a.data.process_bibtex_annotations()

        after = self.a.data['Hafen2019'].notes

        assert before == after

########################################################################

class TestAPIUsage( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

########################################################################
    
    def test_get_ads_data( self ):

        self.a.get_ads_data( identifier='from_citation', skip_unofficial=False )

########################################################################
    
    def test_get_ads_data_unofficial_publication( self ):

        self.a.add_unpub(
            citation_key = r'Craaaazy citation key~: asf^*&',
            point = 'Some sort of point here.',
            conditions = { 'tcool/tff': np.array([ -np.inf, 10. ]) }
        )

        self.a.get_ads_data( identifier='from_citation' )

########################################################################

    @patch( 'ads.SearchQuery' )
    def test_get_ads_data_skip_done( self, mock_search ):

        for key, item in self.a.data.items():
            item.ads_data = { 'fake_dict': True }

        self.a.get_ads_data( identifier='arxiv', )

        mock_search.assert_not_called()

########################################################################

class TestFromBibcodes( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

        self.bibtex_fp = './tests/data/example_atlas/cc_ads.bib' 

    def tearDown( self ):
        if os.path.isfile( self.bibtex_fp ):
            os.remove( self.bibtex_fp )

    ########################################################################

    def test_from_bibcodes( self ):

        bibcodes = [
            '2019MNRAS.488.1248H',
            '2020MNRAS.494.3581H',
        ]

        a = atlas.Atlas.from_bibcodes(
            self.a.atlas_dir,
            bibcodes,
        )

        # Saved in the right spot
        assert a.bibtex_fp == self.bibtex_fp

        # Expected values for entries
        for key in [ 'title', 'year', 'arxivid' ]:
            assert a.data['2019MNRAS.488.1248H'].citation[key] == self.a.data['Hafen2019'].citation[key]

        # Test we can get the abstracts
        a.process_abstracts()
        assert a.n_err_abs == 0

    ########################################################################

    def test_from_bibcodes_existing_bib( self ):

        # Copy the duplicate file so there's already a bib there
        shutil.copyfile(
            './tests/data/duplicate.bib',
            self.bibtex_fp,
        )

        bibcodes = [
            '2020MNRAS.494.3581H',
        ]

        a = atlas.Atlas.from_bibcodes(
            self.a.atlas_dir,
            bibcodes,
        )

        # Saved in the right spot
        assert a.bibtex_fp == self.bibtex_fp

        # Expected values for entries
        for key in [ 'title', 'year', 'arxivid' ]:
            assert a.data['2019MNRAS.488.1248H'].citation[key] == self.a.data['Hafen2019'].citation[key]

    ########################################################################

    def test_import_bibcodes( self ):

        bibcodes = [
            '2019MNRAS.488.1248H',
            '2020MNRAS.491.6102B',
        ]

        self.a.import_bibcodes( bibcodes )

        # Check that the values exist
        for key in [ 'title', 'year', 'arxivid' ]:
            self.a.data['2020MNRAS.491.6102B'].citation[key]

        # This should already exist, so yeah, make sure it doesn't
        assert '2019MNRAS.488.1248H' not in self.a.data.keys()

########################################################################

class TestRealisticAtlas( unittest.TestCase ):
    '''Test functionality for a realistic, messy atlas.'''

    def setUp( self ):

        self.atlas_dir = './tests/data/realistic_atlas'

    def tearDown( self ):
        for f in [ 'atlas_data.h5', 'projection.h5', ]:
            fp = os.path.join( self.atlas_dir, f )
            if os.path.exists( fp ):
                os.remove( fp )

    ########################################################################

    @pytest.mark.slow
    def test_basic_pipeline( self ):

        # Load
        bibtex_fp = os.path.join( self.atlas_dir, 'zotero.bib' )
        a = atlas.Atlas( self.atlas_dir, bibtex_fp=bibtex_fp)

        # Open the bibliography to ensure nothing was skipped
        with open( bibtex_fp, 'r' ) as bibfile:
            bibfile_str = bibfile.read()
        # Do our own basic parsing...
        not_loaded = []
        skipped_types = [ 'online', 'thesis' ]
        for entry in bibfile_str.split( '\n@' )[1:]:
            head = entry.split( ',' )[0]
            entry_type, key = head.split( '{' )
            if entry_type in skipped_types:
                continue
            if key not in a.data.keys():
                not_loaded.append( key )
        assert len( not_loaded ) == 0

        # Add in an unofficial publication or two
        a.add_unpub(
            'Unofficial2021',
            'Cats are truly amazing, and better than galaxies often.',
            [ 'Hafen', 'Saavedra', 'maybe-Alex' ],
        )
        a.add_unpub(
            'Unofficial2021',
            'Dogs are also not bad.',
        )
        a.add_unpub(
            'Unofficial+in prep',
            'Lizards, sure?',
            [ 'But what about Lizards+in prep', ]
        )

        # Process abstracts
        a.process_abstracts( identifier='from_citation' )

        # Check completeness
        successes = []
        failures = []
        for key, item in a.data.items():
            if hasattr( item, 'abstract' ):
                abs_str = item.abstract['str']
                if abs_str is not None and abs_str != '':
                    successes.append( key )
                    continue
            failures.append( key )

        # There are some failures that are expected
        not_in_ads = [
            'Riedl2006',
            'Hartigan1982',
            'Chan2017',
            'Scheufele1999',
            'Anderson2016',
            'Turnbull1976',
            'Whittaker2000',
            'Runeson2006',
            'Coelho2017',
            'Varotsis2018',
            'Kaplan1958',
            'Pirker2015',
            'Deterding2011',
            'Reiners2015',
            'Yarkoni2019',
            'Hanes1940',
            'Schilling2005',
            'Price1976',
            'Merton1968',
            'Kluyver2016',
            'Perez2007',
            'DeSollaPrice1989',
            'Vinkers2015',
            'Fortunato2018',
            'Sinatra2016',
            'Strevens2003',
            'Kitcher1990',
            'steegenIncreasingTransparencyMultiverse2016',
            'Shockley1957',
            'Azoulay2011',
            'Small1973',
            'Stringer2010',
            'Liu2013',
            'Uzzi2005',
            'Stringer2008',
            'Falk-Krzesinski2011',
            'Stokols2008',
            'Fiore2008',
            'CommitteeonFacilitatingInterdisciplinaryResearch2004',
            'Fleming2001',
            'Schilling2011',
            'Cluley2012',
            'JONES2009',
            'Weitzman1998',
            'westMisinformationScience2021',
        ]
        no_abstract_exists = [
            'Fox2017',
            'Smagorinsky1963',
            'Whiteside1970',
        ]
        unofficial = [
            'Hafen:Unofficial2021',
            'Saavedra:Unofficial2021',
            'maybe-Alex:Unofficial2021',
            'Unofficial2021',
            'But what about Lizardsin prep:Unofficial+in prep',
        ]
        expected_failures = not_in_ads + no_abstract_exists + unofficial
        unhandled = []
        for i, key in enumerate( failures ):
            if key not in expected_failures:
                unhandled.append( key )
        assert len( unhandled ) == 0

        # Save
        # a.save_data()

        # Calculate concept projection
        cp_dict = a.concept_projection()

        unhandled = []
        expected_failures = not_in_ads + no_abstract_exists
        for key in a.data.keys():
            if key not in cp_dict['publications']:
                if key not in expected_failures:
                    unhandled.append( key )
        assert len( unhandled ) == 0

    ########################################################################

    def test_sanchez2018( self ):
        '''Individual case prone to breaking.
        In this case, the "\}" in the abstract of the paper is what's causing
        the crash for some reason...
        '''

        cite_key = 'Sanchez2018'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'hummels2016.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_vandevoort2012a( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'VandeVoort2012a'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'vandevoort2012.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_chen2005( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'Chen2005'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'chen2005.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_prochaska2005( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'Prochaska2005'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'chen2005.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_petitjean1993( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'Petitjean1993'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'chen2005.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_noterdaeme2012( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'Noterdaeme2012'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'noterdaeme2012.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_fumagalli2010( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'Fumagalli2010'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'noterdaeme2012.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_fox2017( self ):
        '''Individual case prone to breaking.
        This actually has no abstract.
        '''

        cite_key = 'Fox2017'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'fox2017.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() == ''

    ########################################################################

    def test_ellison2018( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'Ellison2018'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'fox2017.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''

    ########################################################################

    def test_danovich2012( self ):
        '''Individual case prone to breaking.'''

        cite_key = 'danovich2012'

        # Load and make into a mini atlas
        bibtex_fp = os.path.join( self.atlas_dir, 'danovich2012.bib' )
        a = atlas.Atlas(
            self.atlas_dir,
            bibtex_fp = bibtex_fp,
            bibtex_entries_to_load = [cite_key, ]
        )
        assert list( a.data.keys() ) == [ cite_key, ]

        # Add in an unofficial publication or two
        a.add_unpub(
            'Unofficial2021',
            'Cats are truly amazing, and better than galaxies often.',
            [ 'Hafen', 'Saavedra', 'maybe-Alex' ],
        )
        a.add_unpub(
            'Unofficial2021',
            'Dogs are also not bad.',
        )
        a.add_unpub(
            'Unofficial+in prep',
            'Lizards, sure?',
            [ 'But what about Lizards+in prep', ]
        )

        a.process_abstracts( identifier='from_citation' )

        assert a[cite_key].abstract_str() != ''
        
########################################################################

class TestRandomAtlas( unittest.TestCase ):

    def setUp( self ):

        self.atlas_dir = './tests/data/random_atlas'

    def tearDown( self ):
        if os.path.exists( self.atlas_dir ):
            shutil.rmtree( self.atlas_dir )

    ########################################################################

    def test_standard( self ):

        a = atlas.Atlas.random_atlas( self.atlas_dir, 5, seed=123 )
        a.save_data()

        assert len( a.data ) == 5
        assert os.path.exists( self.atlas_dir )

        for key, item in a.data.items():
            for f in [ 'references', 'citations', 'entry_date' ]:
                assert hasattr( item, f )

    ########################################################################

    def test_astro_ga_only( self ):

        a = atlas.Atlas.random_atlas( self.atlas_dir, 3, seed=123, arxiv_class='astro-ph.GA' )
        a.save_data()

        assert len( a.data ) == 3
        assert os.path.exists( self.atlas_dir )

        for key, item in a.data.items():
            assert 'Astrophysics - Astrophysics of Galaxies' in item.citation['keywords']

    ########################################################################

    def test_astro_only( self ):

        a = atlas.Atlas.random_atlas( self.atlas_dir, 3, seed=123, arxiv_class='astro-ph' )
        a.save_data()

        assert len( a.data ) == 3
        assert os.path.exists( self.atlas_dir )

        for key, item in a.data.items():
            assert item.citation['primaryclass'].split( '.' )[0] == 'astro-ph'

    ########################################################################
    
    def test_date_range( self ):

        a = atlas.Atlas.random_atlas(
            self.atlas_dir,
            5,
            '2014',
            '2015',
            seed=123,
        )

        for key, item in a.data.items():
            assert pd.to_datetime( item.entry_date ).year == 2014

    ########################################################################
    
    def test_date_range_astro( self ):

        a = atlas.Atlas.random_atlas(
            self.atlas_dir,
            5,
            start_time='2014',
            end_time = '2015',
            seed=1234,
            arxiv_class='astro-ph'
        )

        for key, item in a.data.items():
            assert pd.to_datetime( item.entry_date ).year == 2014

########################################################################

class TestUnofficialPublication( unittest.TestCase ):

    def setUp( self ):

        # Delete and recreate
        self.empty_dir = './tests/data/empty_atlas'
        try:
            shutil.rmtree( self.empty_dir )
        except FileNotFoundError:
            pass
        os.makedirs( self.empty_dir )
        shutil.copyfile(
            './tests/data/example_atlas/example.bib',
            os.path.join( self.empty_dir, 'example.bib' ),
        )

        self.a = atlas.Atlas( self.empty_dir )

    def tearDown( self ):

        try:
            shutil.rmtree( self.empty_dir )
        except FileNotFoundError:
            pass

    ########################################################################

    def test_add_unpub( self ):

        point_a = (
            'A robust outcome of thermal instability/precipitation ' \
            'models is that the gaseous halos (and coronae) in general ' \
            'cannot spend a lot of time with min(tcool/tff)<10.'
        )
        self.a.add_unpub(
            citation_key = 'Prateek Sharma',
            point = point_a,
            conditions = { 'tcool/tff': np.array([ -np.inf, 10. ]) }
        )
        assert self.a.data['Prateek Sharma'].points() == [ point_a, ]

        def tcool_tff_constraint( tcool, tff ):
            return tcool/tff < 10.

        point_b = (
            'Hot halos are prone to condensation of cold gas which' \
            'decouples from the hot phase and precipitates' \
            'and circularizes at some small radius.'
        )
        self.a.add_unpub(
            citation_key = 'Prateek Sharma',
            point = point_b,
            conditions = { ('tcool', 'tff'): tcool_tff_constraint }
        )
        assert self.a.data['Prateek Sharma'].points() == [ point_a, point_b ]

        # Should fail if a publication date is attempted to retrieve
        try:
            self.a.data['Prateek Sharma'].publication_date
            assert False
        except Exception:
            pass

    ########################################################################

    def test_add_unpub_references( self ):

        point = 'Galaxy major axis sightlines often have absorption with ' \
        'the Doppler shift sharing the same sign as the galactic disk.'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point,
            references = [ 'Steidel2002', 'Kacprzak2010', 'Kacprzak2011' ],
        )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2011:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_parse( self ):

        point = 'Galaxy major axis sightlines often have absorption with ' \
        'the Doppler shift sharing the same sign as the galactic disk.'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point,
            references = 'Steidel+2002; Kacprzak et al. 2010, 2011; ' \
                'Bouche+2013, 2016; Diamond-Stanic et al. 2016; Ho+2017; ' \
                'Martin+2019, Ho+2019',
        )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouche2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]
        assert self.a.data['Martin2019:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_fullparse( self ):

        point = 'Galaxy major axis sightlines often have absorption with ' \
        'the Doppler shift sharing the same sign as the galactic disk.'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point + '(Steidel+2002; Kacprzak et al. 2010, 2011; ' \
                'Bouche+2013, 2016; Diamond-Stanic et al. 2016; Ho+2017; ' \
                'Martin+2019)',
        )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouche2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_roughfullparse( self ):

        point = 'Sightlines along galaxy major axes often detect ' \
            'cir- cumgalactic absorption with the Doppler shift sharing the ' \
            'same sign as the galactic disk. This implies the CGM corotates ' \
            'with the galaxy disks out to large radii '
        point_second_half = '(Stei- del et al. 2002; Kacprzak et al. 2010, ' \
            '2011; Bouch´e et al. 2013, 2016; Diamond-Stanic et al. 2016; ' \
            'Ho et al. 2017; Martin et al. 2019).'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point + point_second_half,
            clean_text = False,
        )
        point += '.' # Remainder after the citation is cut out...
        assert self.a.data['Stei- del2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouch´e2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_cleanfullparse( self ):

        point = 'Sightlines along galaxy major axes often detect ' \
            'cir- cumgalactic absorption with the Doppler shift sharing the ' \
            'same sign as the galactic disk. This implies the CGM corotates ' \
            'with the galaxy disks out to large radii '
        point_second_half = '(Stei- del et al. 2002; Kacprzak et al. 2010, ' \
            '2011; Bouch´e et al. 2013, 2016; Diamond-Stanic et al. 2016; ' \
            'Ho et al. 2017; Martin et al. 2019).'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point + point_second_half,
        )
        point += '.' # Remainder after the citation is cut out...
        point = point.replace( '- ', '' )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kacprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouch´e2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_save_and_load( self ):

        point = 'Galaxy major axis sightlines often have absorption with ' \
        'the Doppler shift sharing the same sign as the galactic disk.'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point,
            references = [ 'Steidel2002', 'Kacprzak2010', 'Kacprzak2011' ],
        )

        # Save and load
        self.a.save_data()
        new_a = atlas.Atlas( self.empty_dir )

        assert new_a.data['Steidel2002:Ho2019'].points() == [ point, ]

        assert isinstance(
            new_a.data['Steidel2002:Ho2019'],
            publication.UnofficialPublication
        )

########################################################################

class TestAtlasData( unittest.TestCase ):

    def setUp( self ):

        # Delete and recreate
        self.empty_dir = './tests/data/empty_atlas'
        try:
            shutil.rmtree( self.empty_dir )
        except FileNotFoundError:
            pass
        os.makedirs( self.empty_dir )
        shutil.copyfile(
            './tests/data/example_atlas/example.bib',
            os.path.join( self.empty_dir, 'example.bib' ),
        )

        self.a = atlas.Atlas( self.empty_dir )

    def tearDown( self ):

        try:
            shutil.rmtree( self.empty_dir )
        except FileNotFoundError:
            pass

    ########################################################################

    def test_load_data_no_data( self ):
        '''We just don't want it to fail.'''

        self.a.load_data( fp=self.empty_dir )

    ########################################################################

    def test_load_data( self ):

        # Create test data
        d = verdict.Dict( {} )
        for key, item in self.a.data.items():
            d[key] = {}
            d[key]['abstract'] = 'Fake abstract for {}'.format( key )
        d.to_hdf5( os.path.join( self.empty_dir, 'atlas_data.h5' ) )

        self.a.load_data()

        # Test
        for key, item in self.a.data.items():
            assert d[key]['abstract'] == self.a[key].abstract

    ########################################################################

    def test_save_data( self ):

        # Create some fake attributes
        for key, item in self.a.data.items():
            item.test_attr = key

        # Function itself
        self.a.save_data(
            attrs_to_save = [ 'test_attr', ],
        )

        # Load saved data
        d = verdict.Dict.from_hdf5( 
            './tests/data/empty_atlas/atlas_data.h5',
        )
        
        for key, item in self.a.data.items():
            assert d[key]['test_attr'] == self.a[key].test_attr

    ########################################################################

    def test_save_data_ads_abstract( self ):

        a_copy = copy.deepcopy( self.a )

        # Get the data
        self.a.process_abstracts( identifier='from_citation' )

        # Compare to the inefficient way
        # We don't want to use the abstracts contained in the citation
        for key, item in a_copy.data.items():
            # Exception for publication missing an arxiv ID
            if key == 'VandeVoort2018a': continue
            if 'abstract' in item.citation:
                del a_copy[key].citation['abstract']
        a_copy.data.process_abstract( return_empty_upon_failure=False )

        # Compare abstracts
        for key, item in a_copy.data.items():

            # Exception for publication missing an arxiv ID
            if key == 'VandeVoort2018a': continue

            abstract = item.abstract['nltk']
            for ikey, iitem in abstract.items():
                for i, v_i in enumerate( iitem ):
                    for j, v_j in enumerate( v_i ):
                        for k, v_k in enumerate( v_j ):
                            assert (
                                v_k ==
                                self.a[key].abstract['nltk'][ikey][i][j][k]
                            )

        # Save function
        self.a.save_data()

        # Load saved data
        d = verdict.Dict.from_hdf5( 
            './tests/data/empty_atlas/atlas_data.h5',
        )
                        
        for key, item in self.a.data.items():
            abstract = item.abstract['nltk']
            for ikey, iitem in abstract.items():
                for i, v_i in enumerate( iitem ):
                    for j, v_j in enumerate( v_i ):
                        for k, v_k in enumerate( v_j ):
                            assert (
                                v_k ==
                                d[key]['abstract']['nltk'][ikey][i][j][k]
                            )

########################################################################

class TestKeyConcepts( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_unique_key_concepts( self ):

        self.a._all_key_concepts = [
            'dog',
            'dogs',
            'kitty cat',
        ]

        actual = self.a.get_unique_key_concepts()
        try:
            expected = set( [
                'dog',
                'kitti cat',
            ] )
            assert actual == expected
        except AssertionError:
            expected = set( [
                'kittycat',
                'dog',
            ] )
            assert actual == expected

    ########################################################################

    def test_unique_key_concepts_forgotten_space( self ):

        np.random.seed( 1234 )

        self.a._all_key_concepts = [
            'dog',
            'dogs',
            'kittycat',
            'kitty cat',
        ]

        actual = self.a.get_unique_key_concepts()
        try:
            expected = set( [
                'dog',
                'kitti cat',
            ] )
            assert actual == expected
        except AssertionError:
            expected = set( [
                'kittycat',
                'dog',
            ] )
            assert actual == expected

########################################################################

class TestSearchPublicationsKeyConcepts( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

    ########################################################################

    def test_default( self ):

        actual = self.a.concept_search(
            'origins of the CGM',
            return_paragraph = False,
            word_per_concept = False,
        )

        h19_kps = self.a['Hafen2019'].notes['key_points']
        h19a_kps = self.a['Hafen2019a'].notes['key_points']
        expected = {
            'Hafen2019': [ h19_kps[0], ],
            'Hafen2019a': [ h19a_kps[-1], ],
        }

        for key, item in expected.items():
            assert item == actual[key]

########################################################################

class TestConceptProjection( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )

        # Ensure we have necessary data available
        self.a['Hafen2019'].abstract['nltk']['primary_stemmed']

        self.alt_fp = './tests/data/example_atlas/projection_alt.h5'

    def tearDown( self ):

        # Make sure we remove extra files
        if os.path.isfile( self.alt_fp ):
            os.remove( self.alt_fp )

    ########################################################################

    def test_concept_projection( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.process_abstracts( identifier='from_citation' )
        cp = self.a.concept_projection()

        # The dimensions of the concept projection
        expected_dim = (
            len( self.a.data ),
            len( cp['component_concepts'] )
        )
        assert cp['components'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == cp['publications'][i]

        assert cp['publication_dates'][0] == self.a[cp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = cp['components'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

    ########################################################################

    def test_cached_concept_projection( self ):

        # Full calculation
        cp = self.a.concept_projection( projection_fp=self.alt_fp )

        with patch( 'numpy.zeros' ) as mock_zeros:
            # This will cause the function to break if it tries to do the
            # actual calculation

            # Loaded fiducial full calculation
            cp_cache =  self.a.concept_projection( projection_fp=self.alt_fp )

        # Cached should equal full
        npt.assert_allclose( cp['components'], cp_cache['components'] )

    ########################################################################

    def test_concept_projection_extend_existing( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        a_partial = copy.deepcopy( self.a )
        for key in [ 'Hafen2019', 'Howk2017', 'Stern2018' ]:
            del a_partial.data[key]

        # Test
        cp_partial = a_partial.concept_projection()
        cp = self.a.concept_projection( existing=cp_partial, overwrite=True )

        # The dimensions of the concept projection
        expected_dim = (
            len( self.a.data ),
            len( cp['component_concepts'] )
        )
        assert cp['components'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v in cp['publications']

        assert cp['publication_dates'][0] == self.a[cp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = cp['components'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

    ########################################################################

    def test_concept_projection_notes_too( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.process_abstracts( identifier='from_citation' )
        self.a.data.process_bibtex_annotations()
        cp = self.a.concept_projection()

        # The dimensions of the concept projection
        expected_dim = (
            len( self.a.data ),
            len( cp['component_concepts'] )
        )
        assert cp['components'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == cp['publications'][i]

        assert cp['publication_dates'][0] == self.a[cp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = cp['components'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

        # The concept projection should also be using notes (properly)
        assert 'garglflinx' in cp['component_concepts']
        assert 'author' not in cp['component_concepts']
        assert 'read' not in cp['component_concepts']

    ########################################################################

    def test_concept_projection_unofficial( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection_unofficial.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.process_abstracts( identifier='from_citation' )
        point_a = (
            'A robust outcome of thermal instability/precipitation ' \
            'models is that the gaseous halos (and coronae) in general ' \
            'cannot spend a lot of time with min(tcool/tff)<10.'
        )
        self.a.add_unpub(
            citation_key = 'Prateek Sharma',
            point = point_a,
            conditions = { 'tcool/tff': np.array([ -np.inf, 10. ]) }
        )
        assert self.a.data['Prateek Sharma'].points() == [ point_a, ]
        cp = self.a.concept_projection( projection_fp=fp )

        # The dimensions of the concept projection
        expected_dim = (
            len( self.a.data ),
            len( cp['component_concepts'] )
        )
        assert cp['components'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == cp['publications'][i]

        assert cp['publication_dates'][0] == self.a[cp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = cp['components'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

        # Make sure we clean up
        fp = './tests/data/example_atlas/projection_unofficial.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

########################################################################

class TestComparison( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas' )
        self.a.import_bibtex( './tests/data/example_atlas/example.bib' )
        self.a.data.process_bibtex_annotations( word_per_concept=True )
        self.a.data.identify_unique_key_concepts()

    ########################################################################

    def test_inner_product_custom_self( self ):

        np.random.seed( 1234 )

        w_aa = self.a.inner_product_custom( self.a, )

        npt.assert_allclose( w_aa, 4266, rtol=0.05 )

    ########################################################################

    def test_inner_product_custom_publication( self ):

        np.random.seed( 1234 )

        w_pa = self.a.inner_product_custom(
            self.a.data['Hafen2019'],
        )

        npt.assert_allclose( w_pa, 529, rtol=0.05 )

