import copy
from mock import patch, call
import numpy as np
import numpy.testing as npt
import os
import pandas as pd
import pytest
import scipy.sparse as ss
import shutil
import string
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

        self.a.get_ads_data( identifier='from_citation', )

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

    @patch( 'ads.ExportQuery' )
    def test_import_bibcodes_chunk( self, mock_export ):

        bibcodes = np.random.randint( 0, high=1000, size=3000 ).astype( 'str' )

        self.a.import_bibcodes( bibcodes )

        expected_first = list( bibcodes[:2000] )
        actual_first = mock_export.call_args_list[0][0][0]
        assert expected_first == actual_first
        expected_second = list( bibcodes[2000:] )
        actual_second = mock_export.call_args_list[1][0][0]
        assert expected_second == actual_second

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

        # Calculate vector projection
        vp_dict = a.vectorize()

        unhandled = []
        expected_failures = not_in_ads + no_abstract_exists
        for key in a.data.keys():
            if key not in vp_dict['publications']:
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
            check_a = 'Astrophysics - Astrophysics of Galaxies' in item.citation['keywords']
            check_b = 'Astrophysics - Galaxy Astrophysics' in item.citation['keywords']

    ########################################################################

    def test_astro_ga_only_min_loops( self ):

        a = atlas.Atlas.random_atlas( self.atlas_dir, 3, seed=1234, start_time='2015', end_time='2016', arxiv_class='astro-ph.GA', max_loops=20 )
        a.save_data()

        assert len( a.data ) == 3
        assert os.path.exists( self.atlas_dir )

        for key, item in a.data.items():
            check_a = 'Astrophysics - Astrophysics of Galaxies' in item.citation['keywords']
            check_b = 'Astrophysics - Galaxy Astrophysics' in item.citation['keywords']

    ########################################################################

    def test_astro_only( self ):

        a = atlas.Atlas.random_atlas( self.atlas_dir, 3, seed=12345, arxiv_class='astro-ph' )
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
            references = [ 'Steidel2002', 'Kavprzak2010', 'Kavprzak2011' ],
        )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2011:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_parse( self ):

        point = 'Galaxy major axis sightlines often have absorption with ' \
        'the Doppler shift sharing the same sign as the galactic disk.'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point,
            references = 'Steidel+2002; Kavprzak et al. 2010, 2011; ' \
                'Bouche+2013, 2016; Diamond-Stanic et al. 2016; Ho+2017; ' \
                'Martin+2019, Ho+2019',
        )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouche2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]
        assert self.a.data['Martin2019:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_fullparse( self ):

        point = 'Galaxy major axis sightlines often have absorption with ' \
        'the Doppler shift sharing the same sign as the galactic disk.'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point + '(Steidel+2002; Kavprzak et al. 2010, 2011; ' \
                'Bouche+2013, 2016; Diamond-Stanic et al. 2016; Ho+2017; ' \
                'Martin+2019)',
        )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouche2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_roughfullparse( self ):

        point = 'Sightlines along galaxy major axes often detect ' \
            'cir- cumgalactic absorption with the Doppler shift sharing the ' \
            'same sign as the galactic disk. This implies the CGM corotates ' \
            'with the galaxy disks out to large radii '
        point_second_half = '(Stei- del et al. 2002; Kavprzak et al. 2010, ' \
            '2011; Bouch´e et al. 2013, 2016; Diamond-Stanic et al. 2016; ' \
            'Ho et al. 2017; Martin et al. 2019).'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point + point_second_half,
            clean_text = False,
        )
        point += '.' # Remainder after the citation is cut out...
        assert self.a.data['Stei- del2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouch´e2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_add_unpub_references_cleanfullparse( self ):

        point = 'Sightlines along galaxy major axes often detect ' \
            'cir- cumgalactic absorption with the Doppler shift sharing the ' \
            'same sign as the galactic disk. This implies the CGM corotates ' \
            'with the galaxy disks out to large radii '
        point_second_half = '(Stei- del et al. 2002; Kavprzak et al. 2010, ' \
            '2011; Bouch´e et al. 2013, 2016; Diamond-Stanic et al. 2016; ' \
            'Ho et al. 2017; Martin et al. 2019).'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point + point_second_half,
        )
        point += '.' # Remainder after the citation is cut out...
        point = point.replace( '- ', '' )
        assert self.a.data['Steidel2002:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2010:Ho2019'].points() == [ point, ]
        assert self.a.data['Kavprzak2011:Ho2019'].points() == [ point, ]
        assert self.a.data['Bouch´e2013:Ho2019'].points() == [ point, ]
        assert self.a.data['Diamond-Stanic2016:Ho2019'].points() == [ point, ]

    ########################################################################

    def test_save_and_load( self ):

        point = 'Galaxy major axis sightlines often have absorption with ' \
        'the Doppler shift sharing the same sign as the galactic disk.'

        self.a.add_unpub(
            citation_key = 'Ho2019',
            point = point,
            references = [ 'Steidel2002', 'Kavprzak2010', 'Kavprzak2011' ],
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
        # Special case.
        d.to_json( os.path.join( self.empty_dir, 'atlas_data.json' ) )

        self.a.load_data()

        # Test
        for key, item in self.a.data.items():
            assert d[key]['abstract'] == self.a[key].abstract

    ########################################################################

    def test_load_data_vectorize( self ):

        # First load to alter data. Is assumed to work.
        self.a.load_data()
        for key, item in self.a.data.items():
            self.a[key].process_abstract( 'Fake abstract for {}'.format( key ) )
        self.a.save_data()

        # Reload
        self.a.load_data()

        # Test that a loaded atlas can be used for a vector projection
        self.a[key].vectorize()

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
        d = verdict.Dict.from_json( 
            './tests/data/empty_atlas/atlas_data.json',
        )
        
        for key, item in self.a.data.items():
            assert d[key]['test_attr'] == self.a[key].test_attr

    ########################################################################

    def test_get_process_and_save_abstracts( self ):

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
        d = verdict.Dict.from_json( 
            './tests/data/empty_atlas/atlas_data.json',
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

        # Check for the stemmed content words, a common vectorization input
        for key, item in self.a.data.items():
            assert item.primary_stemmed_points_str() == d[key]['stemmed_content_words']

    ########################################################################

    def test_load_data_hdf5( self ):

        # Create test data
        d = verdict.Dict( {} )
        for key, item in self.a.data.items():
            d[key] = {}
            d[key]['abstract'] = 'Fake abstract for {}'.format( key )
        # Special case.
        d.to_hdf5( os.path.join( self.empty_dir, 'atlas_data.h5' ) )

        self.a.load_data( format='hdf5' )

        # Test
        for key, item in self.a.data.items():
            assert d[key]['abstract'] == self.a[key].abstract

    ########################################################################

    def test_load_data_vectorize_hdf5( self ):

        # First load to alter data. Is assumed to work.
        self.a.load_data()
        for key, item in self.a.data.items():
            self.a[key].process_abstract( 'Fake abstract for {}'.format( key ) )
        self.a.save_data()

        # Reload
        self.a.load_data( format='hdf5' )

        # Test that a loaded atlas can be used for a vector projection
        self.a[key].vectorize()

    ########################################################################

    def test_save_data_hdf5( self ):

        # Create some fake attributes
        for key, item in self.a.data.items():
            item.test_attr = key

        # Function itself
        self.a.save_data(
            attrs_to_save = [ 'test_attr', ],
            format = 'hdf5',
        )

        # Load saved data
        d = verdict.Dict.from_hdf5( 
            './tests/data/empty_atlas/atlas_data.hdf5',
        )
        
        for key, item in self.a.data.items():
            assert d[key]['test_attr'] == self.a[key].test_attr

    ########################################################################

    def test_get_process_and_save_abstracts_hdf5( self ):

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
        self.a.save_data( format='hdf5' )

        # Load saved data
        d = verdict.Dict.from_hdf5( 
            './tests/data/empty_atlas/atlas_data.hdf5',
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

class TestVectorize( unittest.TestCase ):

    def setUp( self ):

        self.a = atlas.Atlas( './tests/data/example_atlas', atlas_data_format='hdf5' )

        # Ensure we have necessary data available
        self.a['Hafen2019'].abstract['nltk']['primary_stemmed']

        self.alt_fp = './tests/data/example_atlas/projection_alt.h5'

    def tearDown( self ):

        # Make sure we remove extra files
        if os.path.isfile( self.alt_fp ):
            os.remove( self.alt_fp )

    ########################################################################

    def test_vectorize( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.process_abstracts( identifier='from_citation' )
        vp = self.a.vectorize()

        # The dimensions of the vector projection
        expected_dim = (
            len( self.a.data ),
            len( vp['feature_names'] )
        )
        assert vp['vectors'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == vp['publications'][i]

        assert vp['publication_dates'][0] == self.a[vp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = vp['vectors'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

        # The vector projection should be able to go back and forth from compressed to not
        vectors_recomp = ss.csr_matrix( vp['vectors'].toarray() )
        npt.assert_allclose( vp['vectors'].indptr, vectors_recomp.indptr )
        npt.assert_allclose( vp['vectors'].data, vectors_recomp.data )
        npt.assert_allclose( vp['vectors'].indices, vectors_recomp.indices )

    ########################################################################

    def test_vectorize_nonumbers_or_punctuation( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.process_abstracts( identifier='from_citation' )
        vp = self.a.vectorize()

        numpun_chars = string.punctuation.replace( '-', '' ) + '0123456789'

        for word in vp['feature_names']:
            for char in numpun_chars:
                assert not char in word

    ########################################################################

    def test_vectorize_empty_abstract( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.get_ads_data( identifier='from_citation' )
        self.a.data['Hafen2019'].abstract = ''
        self.a.vectorize()

    ########################################################################

    def test_vectorize_sparse_save( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Not sparse
        self.a.process_abstracts( identifier='from_citation' )
        vp = self.a.vectorize( sparse=False )
        size = os.path.getsize( fp )

        # Sparse
        self.a.process_abstracts( identifier='from_citation' )
        vp = self.a.vectorize( overwrite=True, sparse=True )
        sparse_size = os.path.getsize( fp )

        assert sparse_size < size

    ########################################################################

    def test_cached_vectorize( self ):

        # Full calculation
        vp = self.a.vectorize( projection_fp=self.alt_fp )

        with patch( 'verdict.Dict.from_hdf5', wraps=verdict.Dict.from_hdf5 ) as mock_from_hdf5:
            # This will cause the function to break if it tries to do the
            # actual calculation

            # Loaded fiducial full calculation
            vp_cache = self.a.vectorize( projection_fp=self.alt_fp )

            mock_from_hdf5.assert_called_once()

        # Cached should equal full
        if ss.issparse( vp['vectors'] ):
            vp['vectors'] = vp['vectors'].toarray()
        if ss.issparse( vp_cache['vectors'] ):
            vp_cache['vectors'] = vp_cache['vectors'].toarray()
        npt.assert_allclose( vp['vectors'], vp_cache['vectors'] )

    ########################################################################

    def test_vectorize_extend_existing( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        a_partial = copy.deepcopy( self.a )
        for key in [ 'Hafen2019', 'Howk2017', 'Stern2018' ]:
            del a_partial.data[key]

        # Test
        vp_partial = a_partial.vectorize( method='homebuilt', sparse=False )
        vp = self.a.vectorize( existing=vp_partial, overwrite=True, method='homebuilt', sparse=False )

        # The dimensions of the vector projection
        expected_dim = (
            len( self.a.data ),
            len( vp['feature_names'] )
        )
        assert vp['vectors'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v in vp['publications']

        assert vp['publication_dates'][0] == self.a[vp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = vp['vectors'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

    ########################################################################

    def test_vectorize_notes_too( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.process_abstracts( identifier='from_citation' )
        self.a.data.process_bibtex_annotations()
        vp = self.a.vectorize()

        # The dimensions of the vector projection
        expected_dim = (
            len( self.a.data ),
            len( vp['feature_names'] )
        )
        assert vp['vectors'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == vp['publications'][i]

        assert vp['publication_dates'][0] == self.a[vp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = vp['vectors'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

        # The vector projection should also be using notes (properly)
        assert 'garglflinx' in vp['feature_names']
        assert 'author' not in vp['feature_names']
        assert 'read' not in vp['feature_names']

    ########################################################################

    def test_vectorize_unofficial( self ):

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
        vp = self.a.vectorize( projection_fp=fp )

        # The dimensions of the vector projection
        expected_dim = (
            len( self.a.data ),
            len( vp['feature_names'] )
        )
        assert vp['vectors'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == vp['publications'][i]

        assert vp['publication_dates'][0] == self.a[vp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = vp['vectors'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

        # Make sure we clean up
        fp = './tests/data/example_atlas/projection_unofficial.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

    ########################################################################

    def test_vectorize_consistent_methods( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.get_ads_data( identifier='from_citation' )
        vp = self.a.vectorize( method='stemmed content words' )
        vp_homebuilt = self.a.vectorize( method='homebuilt' )

        npt.assert_allclose( vp['vectors'].toarray(), vp_homebuilt['vectors'].toarray() )

    ########################################################################

    def test_vectorize_homebuilt( self ):

        # Make sure we don't count cached files
        fp = './tests/data/example_atlas/projection.h5' 
        if os.path.isfile( fp ):
            os.remove( fp )

        # Test
        self.a.process_abstracts( identifier='from_citation' )
        vp = self.a.vectorize( method='homebuilt' )

        # The dimensions of the vector projection
        expected_dim = (
            len( self.a.data ),
            len( vp['feature_names'] )
        )
        assert vp['vectors'].shape == expected_dim

        # Projected publications check
        for i, v in enumerate( list( self.a.data.keys() ) ):
            assert v == vp['publications'][i]

        assert vp['publication_dates'][0] == self.a[vp['publications'][0]].publication_date

        # There should be no component with entirely zeros
        unnormed_a = vp['vectors'].sum( axis=0 )
        assert np.nanmin( unnormed_a  ) > 0.

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

