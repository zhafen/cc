import ads
import bibtexparser
import copy
import glob
import io
import nltk
from nltk.metrics import edit_distance
import numpy as np
import os
import pandas as pd
import re
import scipy.sparse as ss
import sklearn.feature_extraction.text as skl_text_features
import warnings
## API_extension::maybe_unnecessary
## The imports can probably be pared down, post-extension.

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper

from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

# Import tqdm but change default options
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, ncols=79)

import matplotlib.pyplot as plt

import augment
import verdict

from . import publication
from . import utils

from . import api
from . import nlp

########################################################################

class Atlas( object ):
    '''Generate an Atlas from a bibliography.

    Args:
        atlas_dir (str):
            Primary location atlas data is stored in.

        load_bibtex (bool):
            If True, load data from a .bib.

        bibtex_fp (str):
            Location to load the bibliography data from. Defaults to 
            $atlas_dir/api.DEFAULT_BIB_NAME
            ## API_extension::default_name_change

        bibtex_entries_to_load (str or list-like):
            If not 'all', load specific bibtex entried.

        load_atlas_data (bool):
            If True load saved data that has been retrieved/processed.

        data_fp (str):
            Location to load atlas data from. Defaults to 
            $atlas_dir/atlas_data.h5

        atlas_data_format (str):
            How the atlas data should be saved or loaded.
            Options are json, hdf5. Given the large amount of nesting and text,
            json is recommended, and can actually save more space than json.

    Returns:
        Atlas:
            An atlas object, designed for exploring a collection of papers.
    '''

    @augment.store_parameters
    def __init__(
        self,
        atlas_dir,
        load_bibtex = True,
        bibtex_fp = None,
        bibtex_entries_to_load = 'all',
        load_atlas_data = True,
        data_fp = None,
        atlas_data_format = 'json',
        api_name = api.DEFAULT_API,
    ):
        # if bibtex_fp == '/Users/nathanielimel/data/literature_topography/regions/region_0/cc_s2.bib':
            # breakpoint()

        # Make sure the atlas directory exists
        os.makedirs( atlas_dir, exist_ok=True )
        
        self.data = verdict.Dict( {} )

        # Load bibtex data
        if load_bibtex:
            if bibtex_fp is None:
                bibtex_fp = os.path.join( atlas_dir, '*.bib' )
                bibtex_fps = glob.glob( bibtex_fp )
                if len( bibtex_fps ) > 1:

                    ## API_extension::default_name_change
                    # Ignore the auxiliary downloaded biliography
                    cc_default_fp = os.path.join( atlas_dir, api.DEFAULT_BIB_NAME )
                    if cc_default_fp in bibtex_fps:
                        bibtex_fps.remove( cc_default_fp )
                    else:
                        raise IOError(
                            'Multiple possible BibTex files. Please specify.'
                        )

                if len( bibtex_fps ) == 0:
                    raise IOError( 'No *.bib file found in {}'.format( atlas_dir ) )
                bibtex_fp = bibtex_fps[0]

            self.import_bibtex( bibtex_fp, entries=bibtex_entries_to_load, api_name = api_name )

        # Load general atlas data
        if load_atlas_data:
            self.load_data( fp=data_fp, format=atlas_data_format )

    ########################################################################

    def __repr__( self ):
        return 'cc.atlas.Atlas:{}'.format( self.atlas_dir )

    def __repr__( self ):
        return 'Atlas'

    def __getitem__( self, key ):

        return self.data[key]

    ########################################################################

    @classmethod
    def random_atlas(
        cls,
        atlas_dir,
        n_sample,
        start_time = '1990',
        end_time = '2015',
        api_name = api.DEFAULT_API,
        fl = [ 'arxivid', 'doi', 'date', 'citation', 'reference', 'abstract', 'bibcode', 'entry_date', 'arxiv_class' ],
        arxiv_class = None,
        seed = None,
        max_loops = None,
        load_atlas_data = True,
        load_bibtex = False,
        **kwargs
    ):
        '''Make an atlas of random publications by choosing random dates and then choosing
        a random publication announced on that date.
        Note that while this means that publications announced on the same date
        as many other publications are less likely to be selected, this is not
        expected to typically be an important effect.

        Args:
            atlas_dir (str):
                Directory to store the atlas data in.

            n_sample (int):
                Number of publications to make the atlas of.

            start_time (pd.Timestamp or pd-compatible string):
                Beginning time to use for the range of selectable publications.

            end_time (pd.Timestamp or pd-compatible string):
                End time to use for the range of selectable publications.

            api_name (str): 
                The API to call. See validate_api for more details.

            arxiv_class (str):
                What arxiv class the publications should belong to, if any.
                If set to 'astro-ph' it will include all subcategories of 'astro-ph'.

            seed (int):
                Integer to use for setting the random number selection.
                Defaults to not being used.

            max_loops (int):
                Number of iterations before breaking. Defaults to 20 * n_sample.

        Returns:
            atlas (Atlas):
                Atlas of publications selected.
        '''

        # Start by checking if there is an existing atlas that has data
        if load_atlas_data:
            result = Atlas( atlas_dir=atlas_dir, load_atlas_data=True, load_bibtex=load_bibtex, **kwargs )
            if len( result.data ) > 0:
                return result

        ## API_extension:random_publications
        ## This call may need to be changed
        pubs = api.random_publications(
            n_sample = n_sample,
            start_time = start_time,
            end_time = end_time,
            api_name = api_name,
            fl = fl,
            arxiv_class = arxiv_class,
            seed = seed,
            max_loops = max_loops,
        )

        if len( pubs ) != n_sample:
            raise Exception( 'Retrieved only {} publications out of {}'.format(
                    len( pubs ),
                    n_sample
                )
            )

        return create_random_atlas_from_pubs( atlas_dir, pubs, fl, api_name, )

    ########################################################################

    @classmethod
    def to_and_from_ids( 
        cls,
        atlas_dir,
        ids,
        api_name = api.DEFAULT_API,
        bibtex_fp = None,
        overwrite_bibtex = False,
        data_fp = None,
        load_atlas_data = False,
        **kwargs
     ):
        '''Generate an Atlas from string ids by downloading and saving the
        citations from an API as a new bibliography.

        ## API_extension::to_and_from_bibcodes

        Args:
            atlas_dir (str):
                Primary location atlas data is stored in.

            ids (list of strs):
                Publications to retrieve; must be ADS bibcodes or S2 paperIDs.

            bibtex_fp (str):
                Location to save the bibliography data at and load from. Defaults to 
                $atlas_dir/api.DEFAULT_BIB_NAME

            overwrite_bibtex (bool):
                If False (default), only append to bibtex_fp.

            data_fp (str):
                Location to save other atlas data at. Defaults to 
                $atlas_dir/atlas_data.h5

            load_atlas_data (bool):
                If False don't load the atlas data from data_fp.

        Returns:
            Atlas:
                An atlas object, designed for exploring a collection of papers.
        '''

        api.validate_api(api_name)

        # Make sure the atlas directory exists
        os.makedirs( atlas_dir, exist_ok=True )

        ## API_extension::default_name_change
        # Save the ids to a bibtex
        if bibtex_fp is None:
            bibtex_fp = os.path.join( atlas_dir, api.DEFAULT_BIB_NAME )

        save_ids_to_bibtex( 
            ids, 
            bibtex_fp = bibtex_fp, 
            api_name = api_name,
            overwrite_bibtex = overwrite_bibtex,
        )

        result = Atlas(
            atlas_dir = atlas_dir,
            bibtex_fp = bibtex_fp,
            data_fp = data_fp,
            load_atlas_data = load_atlas_data,
            **kwargs
        )

        return result

    ########################################################################

    @classmethod
    def export_ads_atlas_to_s2_via_bibtex( 
        cls,
        atlas_dir,
        read_bibtex_fp,
        write_bibtex_fp,
        data_fp = None,
        # load_atlas_data = False,
        **kwargs,
     ):
        '''Generate an Atlas from an ADS bibliography by downloading and saving the citations from S2 as a new bibliography.

        # NOTE: this should probably go in a more general `import_bibtex`, eventually
        # TODO: make this function more general if we want to go from s2 to ads

        Args:
            atlas_dir (str):
                Primary location atlas data is stored in.

            read_bibtex_fp (str):
                Location to load the bibliography data from. Defaults to 
                $atlas_dir/api.DEFAULT_BIB_NAME                

            write_bibtex_fp (str):
                Location to save the bibliography data at. Defaults to 
                $atlas_dir/api.DEFAULT_BIB_NAME

            data_fp (str):
                Location to save other atlas data at. Defaults to 
                $atlas_dir/atlas_data.h5

            load_atlas_data (bool):
                If False don't load the atlas data from data_fp.

        Returns:
            Atlas:
                An atlas object, designed for exploring a collection of papers.
        '''
        api_name = api.S2_API_NAME

        # Make sure the atlas directory exists
        os.makedirs( atlas_dir, exist_ok=True )

        # load an atlas from an ads bibtex file
        atl = Atlas(
            atlas_dir = atlas_dir,
            load_bibtex = True,
            bibtex_fp = read_bibtex_fp,
            load_atlas_data = False, # we want from bib, not saved json or h5.
        )

        ids = [
            # N.B.: item is a Publication, not a Paper
            api.citation_to_api_call( item.citation, api_name = api_name )
            for item in atl.data.values()
        ]
        papers = api.call_s2_api( ids )

        # convert each entry to s2 key and paper values
        _data = {}
        for i, (_, item) in enumerate(atl.data.items()):
            # N.B.: item is a Publication
            # By adding the api result, we add the citations and refs to a Publication's Paper.
            paper_id = papers[i].paperId
            pub = publication.Publication( paper_id ) 
            pub.paper = papers[i]
            pub.citation = s2_paper_to_bib_entry(pub.paper)
            _data[paper_id] = pub
        # replace data
        atl.data = _data

        # save to a different bibtex file
        save_s2_papers_to_bibtex( papers, bibtex_fp = write_bibtex_fp )

        # save atlas for reloading
        atl.save_data(
            fp = data_fp,
        )

        return atl


    ########################################################################    

    @classmethod
    def to_and_from_bibcodes(
        cls,
        atlas_dir,
        bibcodes,
        bibtex_fp = None,
        data_fp = None,
        load_atlas_data = False,
        **kwargs
    ):
        '''Generate an Atlas from bibcodes by downloading and saving the
        citations from ADS as a new bibliography.

        ## API_extension::to_and_from_bibcodes

        Args:
            atlas_dir (str):
                Primary location atlas data is stored in.

            bibcodes (list of strs):
                Publications to retrieve.

            bibtex_fp (str):
                Location to save the bibliography data at. Defaults to 
                $atlas_dir/api.DEFAULT_BIB_NAME
            ## API_extension::default_name_change

            data_fp (str):
                Location to save other atlas data at. Defaults to 
                $atlas_dir/atlas_data.h5

            load_atlas_data (bool):
                If False don't load the atlas data from data_fp.

        Returns:
            Atlas:
                An atlas object, designed for exploring a collection of papers.
        '''

        # Make sure the atlas directory exists
        os.makedirs( atlas_dir, exist_ok=True )

        ## API_extension::default_name_change
        # Save the bibcodes to a bibtex
        if bibtex_fp is None:
            bibtex_fp = os.path.join( atlas_dir, api.DEFAULT_BIB_NAME )
        save_ids_to_bibtex( bibcodes, bibtex_fp, api_name = api.ADS_API_NAME )

        result = Atlas(
            atlas_dir = atlas_dir,
            bibtex_fp = bibtex_fp,
            data_fp = data_fp,
            load_atlas_data = load_atlas_data,
            **kwargs
        )

        return result

    ########################################################################

    def import_ids( self, ids, bibtex_fp=None , api_name = api.DEFAULT_API):
        '''Import bibliography data using the publication ids, e.g. bibcodes or paperIds.

        Args:
            bibcodes (list of strs):
                Publications to retrieve.

            bibtex_fp (str):
                Location to save the bibliography data at. Defaults to 
                $atlas_dir/cc_ads.bib
            ## API_extension::default_name_change

        Updates:
            self.data and the file at bibtex_fp:
                Saves the import bibliography data to the instance and disk.
        '''

        # Store original keys for later removing duplicates
        original_keys = copy.copy( list( self.data.keys() ) )        
        if bibtex_fp is None:
            bibtex_fp = os.path.join( self.atlas_dir, api.DEFAULT_BIB_NAME )
        save_ids_to_bibtex( ids, bibtex_fp, api_name = api_name )
        self.import_bibtex( bibtex_fp, verbose=False )

        self.prune_duplicates( original_keys )        

    ########################################################################

    # def import_bibcodes( self, bibcodes, bibtex_fp=None ):
    #     '''Import bibliography data using bibcodes.

    #     ## API_extension::to_and_from_bibcodes

    #     Args:
    #         bibcodes (list of strs):
    #             Publications to retrieve.

    #         bibtex_fp (str):
    #             Location to save the bibliography data at. Defaults to 
    #             $atlas_dir/cc_ads.bib
    #         ## API_extension::default_name_change

    #     Updates:
    #         self.data and the file at bibtex_fp:
    #             Saves the import bibliography data to the instance and disk.
    #     '''

    #     # Store original keys for later removing duplicates
    #     original_keys = copy.copy( list( self.data.keys() ) )

    #     # API_extension::default_name_change
    #     # Import bibcodes
    #     if bibtex_fp is None:
    #         bibtex_fp = os.path.join( self.atlas_dir, api.DEFAULT_BIB_NAME )
    #     save_bibcodes_to_bibtex( bibcodes, bibtex_fp, )
    #     self.import_bibtex( bibtex_fp, verbose=False )

    #     self.prune_duplicates( original_keys )

    ########################################################################

    def import_bibtex( self, bibtex_fp, entries='all', process_annotations=True, verbose=True, api_name = api.ADS_API_NAME, ):
        '''Import publications from a BibTex file.
        
        Args:
            bibtex_fp (str):
                Filepath to the BibTex file.

            entries (list-like):
                Which entries from the bibtex to load. If None, load all.

            verbose (bool):
                Verbosity.
        '''

        if verbose:
            print( 'Loading bibliography entries.' )

        # Load the database
        with open( bibtex_fp, 'r' ) as bibtex_file:
            # Preprocess to get rid of characters that give the parser trouble
            bibtex_file_str = bibtex_file.read()
            bibtex_file_str = bibtex_file_str.replace( '\\}', '' ).replace( '\\{', '' )
            bibtex_file = io.StringIO( bibtex_file_str )

            bib_database = bibtexparser.load( bibtex_file )

        # Store into class
        if verbose:
            print( 'Storing bibliography entries.' )

        for citation in tqdm( bib_database.entries ):
            citation_key = citation['ID']

            if entries != 'all' and citation_key not in entries:
                continue

            # Avoid overwriting existing loaded data
            if citation_key in self.data:
                p = self.data[citation_key]
            else:
                p = publication.Publication( citation_key )

                if api_name == api.S2_API_NAME:
                    # construct a Paper from an S2 bib, 
                    # then associate it with a Publicatione

                    # N.B.: this is basically useless; the important thing is to get a publication associated with a paper that has citations and references. Which requires an api call.

                    # NOTE: more precisely, in our export_... constructor, we call the api. So what this means is that when loading from already saved data (currently bibtex, but obviously that's not enough, we need atlas_data for refs and cites), the problem is that we're not correctly saving the full paper refs and cites, or loading it.

                    # paper = api.citation_to_s2_paper( citation )
                    # p.paper = paper # so comment this out for explicitness
                    pass
                
                p.citation = citation # fine to add this to Paper

            if process_annotations:
                for annote_key in [ 'annote', 'annotation' ]:
                    if annote_key in p.citation:
                        p.process_bibtex_annotations( annotation_str=p.citation[annote_key] )

            self.data[citation_key] = p

    ########################################################################

    def load_data( self, fp=None, format='json' ):
        '''Load general data saved to atlas_data.h5
        
        Args:
            fp (str):
                Filepath to the atlas_data.h5 file.
                If None, looks in self.atlas_dir

            format (str):
                Format the data is saved in. Options are json, hdf5.
        '''

        print( 'Loading saved atlas data.' )

        # Filepath
        if fp is None:
            fp = os.path.join( self.atlas_dir, 'atlas_data.{}'.format( format ) )
            if not os.path.isfile( fp ) and format == 'hdf5':
                fp = os.path.join( self.atlas_dir, 'atlas_data.h5'.format( format ) )

        # Exit if no data to load
        if not os.path.isfile( fp ):
            print( 'No saved data at {}'.format( fp ) )
            return

        # Load
        if format == 'json':
            data_to_load = verdict.Dict.from_json( fp )
        elif format == 'hdf5':
            data_to_load = verdict.Dict.from_hdf5( fp )
        else:
            raise KeyError( 'Unrecognized file format, {}'.format( format ) )

        # Update data with loaded data
        for key, item in tqdm( self.data.items() ):

            # When the paper doesn't have any data stored for it
            if key not in data_to_load:
                continue
            
            breakpoint()
            for ikey, iitem in data_to_load[key].items():
                try:
                    setattr( item, ikey, iitem )
                except AttributeError:
                    breakpoint()

        # Store new data
        for key, item in tqdm( data_to_load.items() ):

            # NOTE: I bet something S2 specific needs to happen here

            # When the atlas contains the entry skip
            if key in self.data:
                continue

            # Choose unofficial or standard publication
            unofficial_flag = False
            if 'unofficial_flag' in item:
                unofficial_flag = item['unofficial_flag']
            if unofficial_flag:
                pub = publication.UnofficialPublication( key )
            else:
                pub = publication.Publication( key )

            # Store
            for ikey, iitem in data_to_load[key].items():
                
                # For the notes that are loaded we want lists, not arrays
                if ikey == 'notes':
                    for iikey, iiitem in iitem.items():
                        iitem[iikey] = list( iiitem )
    
                setattr( pub, ikey, iitem )
            self.data[key] = pub

    ########################################################################

    def save_data(
        self,
        fp = None,
        format = 'json',
        attrs_to_save = [
            'abstract',
            'citations',
            'references',
            'bibcode',
            'entry_date',
            'notes',
            'unofficial_flag',
            'citation',
            'stemmed_content_words',
        ],
        handle_jagged_arrs = 'row datasets',
        return_data = False,
    ):
        '''Save general data saved to atlas_data.h5
        
        Args:
            fp (str):
                Filepath to the atlas_data.h5 file.
                If None, looks in self.atlas_dir

            format (str):
                Data format. Options are json, hdf5.

            attrs_to_save (list of strs):
                List of attributes for each item of self.data to save.

            handle_jagged_arrs (str):
                How, when saving using hdf5, to handle jagged arrays.
        '''

        # Filepath
        if fp is None:
            fp = os.path.join( self.atlas_dir, 'atlas_data.{}'.format( format ) )

        # Retrieve data
        print( 'Preparing to save data.' )
        data_to_save = verdict.Dict( {} )
        for key, item in tqdm( self.data.items() ):
            # breakpoint()
            data_to_save[key] = {}
            for attr in attrs_to_save:
                if hasattr( item, attr):
                    data_to_save[key][attr] = getattr( item, attr )
            # Don't try to save empty dictionaries
            if data_to_save[key] == {}:
                del data_to_save[key]

        # Save
        print( 'Saving to {}'.format( fp ) )
        if format == 'json':
            data_to_save.to_json( fp, )
        elif format == 'hdf5':
            data_to_save.to_hdf5( fp, handle_jagged_arrs=handle_jagged_arrs )
        else:
            raise KeyError( 'Unrecognized file format, {}'.format( format ) )

        if return_data:
            return data_to_save

    ########################################################################

    def add_unpub(
        self,
        citation_key,
        point,
        references = None,
        parse_references = True,
        clean_text = True,
        conditions = None,
    ):
        '''Add an unofficial publication to the Atlas. If the publication
        already exists then add to it.

        Args:
            citation_key (str):
                Key to use for the unofficial publication.

            point (str):
                The information to store. Usually some fact.

            references (str or list of strs):
                In some cases the unofficial publication isn't making the
                statement itself, but is referring to another publication.
                In such cases we will want to create different publications
                with the point. These publications will be referred to by the
                citation key "reference:citation_key".

            parse_references (bool):
                If True, parse both the point and references to identify
                references that are in a less-clear format. Can identify
                "Author+XXXX" and "Author et al. XXXX" formats and store
                appropriately.

            clean_text (bool):
                If True, do some easy and common cleaning.
                TODO: integrate this with a better text cleaning tool.

            conditions (dict):
                Conditions that must be valid for the point to be true.
                Currently not used for anything and conditions aren't
                reliably paired with points.

        Modifies:
            self.data (verdict.Dict):
                Adds new publications.
        '''

        # Parse if point or points
        if not pd.api.types.is_list_like( point ):
            points = [ point, ]
        else:
            points = point

        # Parse points for references
        if references is None:
            references = []
        elif isinstance( references, str ):
            references = [ references, ]

        if clean_text:
            points = [ _.replace( '- ', '' ) for _ in points ]
            references = [ _.replace( '- ', '' ) for _ in references ]

        if parse_references:
            new_points = []
            for i, point in enumerate( points ):

                # We'll be cleaning up some points, so let's make a copy
                new_point = point

                # Find references in parenthesis
                stack = []
                for j, c in enumerate( point ):

                    if c == '(':
                        stack.append( j )
                    elif c == ')':
                        start = stack.pop()
                        ref_str = point[start+1:j]
                        # Append references, but only if criteria are met
                        if (
                            len( stack ) == 0 and
                            '+' in ref_str or 'et al.' in ref_str
                        ):
                            references.append( ref_str )

                            new_point = point.replace( '(' + ref_str + ')', '' )

                new_points.append( new_point )
            points = new_points

        def store_to_data( citation_key ):
            '''Store the point to the atlas data.
            '''

            # Create or load the publication
            if citation_key in self.data.keys():
                pub = self.data[citation_key]
            else:
                pub = publication.UnofficialPublication(
                    citation_key
                )

            # Add the points
            for p in points:
                pub.process_annotation_line( p, word_per_concept=True )

            self.data[citation_key] = pub

        # Default case without references
        if len( references ) == 0:
            store_to_data( citation_key )

        # Make citation key list (for when the publication isn't the source
        # of the point, but is making the statement using references)
        else:

            # Default case
            if not parse_references:
                parsed_refs = references
            # Parse the references
            else:
                parsed_refs = []
                for refs_i in references:
                    split_refs = re.split( '; |, ', refs_i )
                    for i, ref in enumerate( split_refs ):

                        # For references that are continuations of previous ones
                        if ref.isdigit():
                            prev_ref = split_refs[i-1]
                            num_index = re.search( r'\d', prev_ref ).start()
                            ref = prev_ref[:num_index] + ref

                        # Remove the extraneous bits
                        ref = ref.replace( '+', '' )
                        ref = ref.replace( ' et al. ', '' )

                        parsed_refs.append( ref )

            # Make the list
            citation_keys = [
                '{}:{}'.format( _, citation_key ) for _ in parsed_refs
            ]

            # And store
            for citation_key in citation_keys:
                store_to_data( citation_key )

    ########################################################################
    # Atlas Operations
    ########################################################################

    def update( self, other ):
        '''Update with information from another atlas.
        When duplicates are found, the other atlas' entries are preferred.

        Modifies:
            self.data
                Adds new publications.
        '''

        new_keys = list( other.data.keys() )

        # Perform the update
        self.data._storage.update( other.data )

        # Perform pruning
        self.prune_duplicates( preferred=new_keys )

    ########################################################################

    # TODO: Delete?
    # def prune_duplicates( self, preferred=[], threshold=0.9, **vectorization_kwargs ):

    #     self.prune_duplicates_via_citation( preferred=preferred )

    #     self.prune_duplicates_via_similarity( threshold=threshold, **vectorization_kwargs )

    ########################################################################

    def prune_duplicates( self, preferred=[] ):
        '''Remove duplicate entries by comparing citation information.

        Args:
            preferred (list of strs):
                Which entries to prefer if duplicates are found.

        Modifies:
            self.data
                Drops duplicate rows.
        '''

        # Break publications into two categories sorted by preference to keep
        existing = set( self.data.keys() )
        preferred = set( preferred ).intersection( existing )
        remaining = list( existing - preferred )

        # Items to compare
        comparison_columns = [
            ( 'eprint', ),
            ( 'doi', ),
            ( 'url', ),
            ( 'title', ),
            ( 'journaltitle', 'volume', 'pages' ),
        ]

        # Make a pandas data frame to efficiently drop duplicates
        to_compare = {}

        # Loop through preferred, remaining
        for keys in [ preferred, remaining ]:

            # Loop through publications
            for key in tqdm( keys ):

                p = self.data[key]
                if not hasattr( p, 'citation' ):
                    continue

                citation = p.citation 

                # Loop through sets of columns to compare
                for columns in comparison_columns:

                    # Create if not existing
                    if columns not in to_compare:
                        to_compare[columns] = {}

                    if 'citation_key' not in to_compare[columns]:
                        to_compare[columns]['citation_key'] = []
                    to_compare[columns]['citation_key'].append( key )

                    # Loop through columns in a set
                    for column in columns:
                        if column not in to_compare[columns]:
                            to_compare[columns][column] = []

                        try:
                            value = citation[column]
                        except KeyError:
                            value = np.nan

                        to_compare[columns][column].append( value )

        # Find duplicates
        duplicates = set()
        for columns, columns_data in to_compare.items():

            # Format
            df = pd.DataFrame( data=columns_data )
            df = df.set_index( 'citation_key' )

            # Drop all-empty rows
            df = df.dropna( how='all' )

            # Find duplicates to remove
            duplicated = df.duplicated()
            duplicated_keys = df.index.values[duplicated] 
            duplicates = duplicates.union( set( duplicated_keys ) )

        # Remove duplicates
        for key in duplicates:
            del self.data[key]

    ########################################################################
    # Data Processing
    ########################################################################

    @property
    def key_concepts( self ):
        '''Easier access for key_concepts. Must be loaded for individual
        publications first.
        '''

        try:
            return self.data.notes.inner_item( 'key_concepts' )
        except KeyError:
            self.data.process_bibtex_annotations()
            return self.data.notes.inner_item( 'key_concepts' )

    ########################################################################

    @property
    def key_points( self ):
        '''Easier access for key_points. Must be loaded for individual
        publications first.
        '''

        try:
            return self.data.notes.inner_item( 'key_points' )
        except KeyError:
            self.data.process_bibtex_annotations()
            return self.data.notes.inner_item( 'key_points' )

    ########################################################################

    @property
    def all_key_concepts( self ):
        '''A set of all key concepts across publications.
        '''

        if not hasattr( self, '_all_key_concepts' ):

            # Flatten
            self._all_key_concepts = []
            for kcs in self.key_concepts.values():
                for kcs_point in kcs:
                    self._all_key_concepts += kcs_point

            self._all_key_concepts = set( self._all_key_concepts )

        return self._all_key_concepts

    ########################################################################

    def get_unique_key_concepts( self, **kwargs ):
        '''Unique key concepts, as simplified using nltk tools.
        Steps to retrieve unique key concepts:
        1. Union of the same stems.
        2. Concepts with a sufficiently low edit distance
           (accounts for mispellings)

        Optional Args:
            max_edit_distance (int):
                Maximum Levenshtein edit-distance between two concepts for them
                to count as the same concept.
        '''

        l = list( self.all_key_concepts )

        self.unique_key_concepts = utils.uniquify_words( l, **kwargs )

        return self.unique_key_concepts

    ########################################################################

    def get_data_via_api(self, *args, api_name = api.DEFAULT_API, **kwargs,):
        '''Get data for all publications via ADS or S2.
        
        Args:
            api_name (str): What api to call. Options are...
            'S2':
                Call the Semantic Scholar Academic Graph API (S2AG)
            'ADS':
                Call the NASA Astrophysics Data System API (ADS)
        '''
        api.validate_api(api_name)
        
        if api_name == api.ADS_API_NAME:
            self.get_and_store_ads_data(*args, **kwargs)
        
        if api_name == api.S2_API_NAME:
            self.get_and_store_s2_data( *args, **kwargs )

    ########################################################################

    def get_s2_data(
        self,
        fields = [ 'abstract', 'citations', 'references', 'authors', ],
        **kwargs,
    ):
        '''Get the Semantic Scholar data for all publications, from citations (the entries of a bibtex file).

        # NOTE: the reason a corresponding function doesn't exist for ADS is because we wanted to go from an ads bib to a s2 bib, but not the other way yet.
        # NOTE: this abstraction is probably useless actually

        Args:
            fields (list of strs):
                Fields to retrieve from ADS.
        
        Returns:
            {
                'papers': (list of resulting papers),
                'queries_data': dict of query data before each paper
            }
        '''
        # Collect query strings to call S2 api
        queries_data = {
            'data_keys': [],
            'ids': [],
        }        
        for key, item in self.data.items():
            # N.B.: item is a Publication, not a Paper
            paper_id = api.citation_to_s2_call( item.citation )

            queries_data['data_keys'].append( key )
            queries_data['ids'].append( paper_id )

        # Query api 
        papers = api.call_s2_api(
            paper_ids=queries_data['ids'], 
            **kwargs,
            )
        return {
            'papers': papers,
            'queries_data': queries_data,
        }

    ########################################################################

    def get_and_store_s2_data(
        self,
        fields = api.S2_QUERY_FIELDS,
        **kwargs,
    ):
        '''Get the Semantic Scholar data for all publications, from citations (the entries of a bibtex file).

        Args:
            fields (list of strs):
                Fields to retrieve from ADS.
        '''
        result = self.get_s2_data(fields, **kwargs)
        queries_data = result['queries_data']
        papers = result['papers']

        for i, key in enumerate(queries_data['data_keys']):
            # store
            atlas_pub = self.data[key]
            self.data[key] = store_s2_data( atlas_pub, papers[i] )

    ########################################################################
    # TODO: most of this is independent of atlas; should probably refactor api calling logic, and unify with other api calling parts of codebase
    def get_and_store_ads_data(
        self,
        fl = [ 'abstract', 'citation', 'reference', 'entry_date',
            'author', 'volume', 'page' ],
        publications_per_request = 300,
        characters_per_request = 2900,
        identifier = 'key_as_bibcode',
        skip_unofficial = True,
        perform_noid_queries = True,
        n_attempts_per_query = 5,
        verbose = False,
    ):
        '''Get the ADS data for all publications.

        ## API_extension::get_data_via_api
        ## We will likely want to keep this function, but also make a general function
        ## that calls the specific functions per API.

        Args:
            fl (list of strs):
                Fields to retrieve from ADS.

            publications_per_request (int):
                Maximum number of publications to request per call to ADS.
                Not as limiting as characters_per_request in most cases.

            characters_per_request (int):
                Maximum number of characters per call to ADS. This is set a bit
                below the character limit ADS seems to have.

            identifier (str):
                What identifier to use to download papers. Options are...
                'key_as_bibcode':
                    This assumes self.data.keys() are ADS bibcodes
                    and we can just use them.
                'from_citation':
                    Use the information contained in each publication's citation
                    for an identifier.
                    Requires some extra work to parse the output.

            skip_unofficial (bool):
                If True don't try to retrieve ADS data for unofficial publications.

            perform_noid_queries (bool):
                Do individual queries for the publications missing a unique ID.

            n_attempts_per_query (int):
                Number of attempts to access the API per query. Useful when experiencing
                connection issues.

            verbose (bool):
                Warn when publication data is not retrieved.
        '''

        # key_as_bibcode is an alias for bibcode
        if identifier == 'key_as_bibcode' or identifier == 'bibcode':
            identifier = 'bibcode'
            # For later downloading
            if 'bibcode' not in fl:
                fl.append( 'bibcode' )
        else:
            if 'identifier' not in fl:
                fl.append( 'identifier' )

        # Build query strings
        n_pubs = 0
        queries = []
        queries_noid = []
        query_i = {
            'search_strs': [],
            'data_keys': [],
            'ids': [],
            'identifiers': [],
        }
        for i, (key, item) in enumerate( self.data.items() ):

            # Skip unofficial publications
            if isinstance( item, publication.UnofficialPublication ) and skip_unofficial:
                pass

            # Skip publications we have data for
            elif hasattr( item, 'ads_data' ):
                pass
            elif hasattr( item, 'citations' ):
                pass

            else:
                if identifier == 'bibcode':
                    q_i = 'bibcode:"{}"'.format( key )
                    id = key
                    ident = 'bibcode'

                elif identifier == 'from_citation':
                    q_i, ident, id = api.citation_to_ads_call( item.citation )

                else:
                    raise KeyError( 'Unrecognized identifier, {}'.format( identifier ))

                # When we don't have a nice unique identifier
                # we store separately to handle later.
                if pd.api.types.is_list_like( ident ):
                    query_noid = {
                        'search_str': q_i,
                        'data_key': key,
                    }
                    queries_noid.append( query_noid )
                    continue

                # Store info about this particular query
                query_i['search_strs'].append( q_i )
                query_i['data_keys'].append( key )
                query_i['ids'].append( id )
                query_i['identifiers'].append( ident )
                n_pubs += 1

            # Break conditions
            end = i + 1 == len( self.data )
            max_pubs = n_pubs >= publications_per_request
            num_chars = len( ' OR '.join( query_i['search_strs'] ) )
            max_chars = num_chars >= characters_per_request
            if end:
                queries.append( query_i )
                break
            if max_pubs or max_chars:
                queries.append( query_i )
                n_pubs = 0
                query_i = {
                    'search_strs': [],
                    'data_keys': [],
                    'ids': [],
                    'identifiers': [],
                }
                continue

        def store_ads_data( atlas_pub, p ):

            # Store
            atlas_pub.ads_data = {}
            for f in fl:
                value = getattr( p, f )

                # Formatting choice, abstract = None replaced
                # with abstract = ''
                if f == 'abstract' and value is None:
                    value = ''

                atlas_pub.ads_data[f] = value

                attr_f = copy.copy( f )
                if attr_f == 'citation' or attr_f == 'reference':
                    attr_f += 's' # NOTE: what's this? a nuisance?
                setattr( atlas_pub, attr_f, value )

            return atlas_pub

        # Exit early if no ids to call
        if len( queries ) == 1 and len( queries[0]['search_strs'] ) == 0:
            if  len( queries_noid ) == 0 and perform_noid_queries:
                print( 'No publications to retrieve data for.' )
                return

        # Query
        print( '    Making {} ADS calls...'.format( len( queries ) ) )
        for query_i in tqdm( queries ):
            search_str = ' OR '.join( query_i['search_strs'] )
            query_dict={
                'q': search_str,
                'fl': fl,
                'rows': publications_per_request,
            }

            # Get publications out. Turned into a function and
            # wrapped to allow multiple attempts.
            @api.keep_trying( n_attempts=n_attempts_per_query )
            def get_pubs_for_query():
                ads_query = ads.SearchQuery( query_dict=query_dict )
                pubs = list( ads_query )
                return pubs
            pubs = get_pubs_for_query()

            # Identify and update
            for i, key in enumerate( query_i['data_keys'] ):
                id = query_i['ids'][i]
                atlas_pub = self.data[key]

                # Match the publication
                # NOTE: how optimized is this?
                found = False
                for p in pubs:
                    if identifier == 'bibcode':
                        if key == p.bibcode:
                            found = True
                    elif identifier == 'from_citation':
                        # Simple case
                        if not pd.api.types.is_list_like( id ):
                            for id_p in p.identifier:
                                if id.lower() in id_p.lower():
                                    found = True
                        # When there's not a single identifier
                        else:
                            pass
                    if found: break

                # Don't try to update if no matching publication was found
                if not found:
                    if verbose:
                        warnings.warn(
                            'No publications found for ' + \
                            '{}. Skipping.'.format( key )
                        )
                    continue
            
                # Store the data
                self.data[key] = store_ads_data( atlas_pub, p )

        # Query for publications without a single ID (so far)
        if perform_noid_queries:
            print(
                '    Making {} ADS calls for publications without IDs...'.format(
                    len( queries_noid ),
                )
            )
            for query_noid in tqdm( queries_noid ):

                key = query_noid['data_key']

                # Make query. Turned into a function to allow multiple attempts
                @api.keep_trying( n_attempts=n_attempts_per_query )
                def get_pubs_for_noid_query():
                    ads_query = ads.SearchQuery(
                        query_dict={
                            'q': query_noid['search_str'],
                            'fl': fl,
                            'rows': publications_per_request,
                        },
                    )
                    pubs = list( ads_query )
                    return pubs
                pubs = get_pubs_for_noid_query()

                if len( pubs ) > 1:
                    if verbose:
                        warnings.warn(
                            'Multiple publications possible ' + \
                            'for {}. Skipping.'.format( key )
                        )
                    continue
                elif len( pubs ) == 0:
                    if verbose:
                        warnings.warn(
                            'No publications found ' + \
                            'for {}. Skipping.'.format( key )
                        )
                    continue
                p = pubs[0]

                # Store the data
                self.data[key] = store_ads_data( self.data[key], p )

    ########################################################################

    def process_abstracts( self, *args, process_stemmed_content_words=True, api_name = api.DEFAULT_API, **kwargs ):
        '''Download and process the abstracts of all publications.
        Faster and with fewer API calls than for each paper individually.

        Args:
            skip_unofficial (bool):
                If True don't try to process abstracts
                of UnofficialPublications

        *Args, **Kwargs:
            Passed to self.get_ads_data
        '''

        ## API_extension::get_data_via_api
        # self.get_ads_data( *args, **kwargs )
        # self.get_data_via_api( *args, api_name = api_name, **kwargs, ) # this might be the right thing to do for ads, but it seems really inefficient for ads.
        if api_name == api.ADS_API_NAME:
            self.get_and_store_s2_data( *args, **kwargs )

        print( '    Doing NLP...' )

        # TODO: this is totally Publication specific, idk how I passed tests before. Probably in virtue of Papers getting casted to Publications without much data loss.

        n_err = 0
        for key, item in tqdm( self.data.items() ):

            # We'll skip ones that are already processed.
            # The indicator of a processed abstract is that
            # the abstract is a dictionary
            if hasattr( item, 'abstract' ):
                if isinstance( item.abstract, verdict.Dict ) or isinstance( item.abstract, dict ):
                    continue
            
            # what should be done
            # item.abstract = nlp.process_abstract( item )
            # item.stemmed_content_words = nlp.primary_stemmed_points_str(item)
            
            if isinstance( item, publication.Publication ):
                abstract_str = item.abstract_str( api_name = api_name )
                if abstract_str == '':
                    n_err += 1
                item.process_abstract( abstract_str=abstract_str, overwrite=True )
            
            if isinstance( item, api.Paper ):
                abstract_str = item.abstract
                item.abstract = {
                    'str': abstract_str,
                }
                if abstract_str is not None:
                    item.abstract['nltk'] = utils.tokenize_and_sort_text( abstract_str )

            if process_stemmed_content_words:
                item.stemmed_content_words = item.primary_stemmed_points_str()

        self.n_err_abs = n_err

    ########################################################################

    def vectorize(
        self,
        method = 'stemmed content words',
        feature_names = None,
        projection_fp = None,
        overwrite = False,
        existing = None,
        verbose = True,
        return_data = True,
        sparse = True,
    ):
        '''Project the abstract of each publication into concept space.
        In simplest form this finds all shared, stemmed nouns, verbs, and
        adjectives between all publications and counts them.

        After creation the vectorized text is saved as a sparse matrix.
        For atlases consisting of tens of thousands of entries the difference
        between saving a sparse matrix or saving a traditional matrix is
        the difference between tens of MB or tens of GB.

        Args:
            method (str):
                What code to use to perform the vectorization. Options are...
                    'stemmed content words':
                        Use stemmed nouns, verbs, adjectives, and adverbs from each publication's
                        abstract and notes. Vectorization done using scikit-learn.
                    'scikit-learn':
                        Default scikit-learn vectorization with no stemming.
                    'homebuilt':
                        Same as 'stemmed content words', but much slower.

            feature_names (array-like of strs):                                  
                Basis concepts to project onto. Defaults to all concepts across
                all publications.

            projection_fp (str):
                Location to save the vectorized text at. Defaults to
                $atlas_dir/projection.h5
                If set to 'pass' then the projection is not saved.

            overwrite (bool):
                If False then check for a cached vectorized text.

            existing (dict or None):
                Dictionary of existing result to build the projection upon.

            verbose (bool):
                If True print additional information while running.

            return_data (bool):
                If True return the resultant dictionary.

            sparse (bool):
                If True save the data as a memory-saving sparse matrix.

        Returns:
            Dictionary:
                Dictionary containing...
                vectors ((n_pub,n_features) np.ndarray of ints):
                    The value at [i,j] is the value of the projection for
                    publication for each i for each concept j.

                norms ((n_pub,) np.ndarray of floats):
                    Normalization for each publication.

                feature_names ((n_features,) np.ndarray of strs):
                    The basis concepts used. By default the union of all
                    stemmed nouns, adjectives, and verbs across all abstracts.

                publications ((n_pubs,) np.ndarray of strs):
                    The publications that are projected.

                publication_dates ((n_pubs,) np.ndarray of strs):
                    Dates of publication.

                entry_dates ((n_pubs) np.ndarray of strs):
                    Dates the database became aware of the publication.
                    Typically pre-publication, due to preprints.
        '''

        if verbose:
            print( 'Vectorizing text...' )

        # File location
        if projection_fp is None:
            projection_fp = os.path.join(
                self.atlas_dir,
                'projection.h5'
            )

        # If cached or saved and not overwriting
        if os.path.isfile( projection_fp ) and not overwrite:
            if verbose:
                print( 'Using saved vectorized text...' )
            if existing is not None:
                warnings.warn(
                    'Passing an existing vectorized text and not ' \
                    + 'overwriting. The vectorized text will fail if the ' \
                    + 'existing and new vectorized text share a save ' \
                    + 'location.'
                )
            self.projection = verdict.Dict.from_hdf5( projection_fp, sparse=sparse )

            # Old format compatbility
            if 'components' in self.projection.keys():
                self.projection['vectors'] = copy.copy( self.projection['components'] )
                del self.projection['components']
            if 'component_concepts' in self.projection.keys():
                self.projection['feature_names'] = copy.copy( self.projection['component_concepts'] )
                del self.projection['component_concepts']

            # Converting to/from sparse matrices
            if sparse:
                self.projection['vectors'] = ss.csr_matrix( self.projection['vectors'] )
            else:
                if ss.issparse( self.projection['vectors'] ):
                    self.projection['vectors'] = self.projection['vectors'].toarray()
            return self.projection

        if hasattr( self, 'projection' ) and not overwrite:
            if verbose:
                print( 'Using cached vectorized text...' )
            return self.projection

        # Set up for component calculation
        if existing is not None:

            if method != 'homebuilt':
                raise NotImplementedError( 'Cannot pass existing projection with scikit-learn-based vectorization.' )

            assert feature_names is None, "Cannot pass component " \
                + "concepts in addition to an existing projection."
            feature_names = list( existing['feature_names'] )
            vectors_list = list( existing['vectors'] )
            projected_publications = list( existing['publications'] )
            pub_date = list( existing['publication_dates'] )
            entry_date = list( existing['entry_dates'] )
        else:
            vectors_list = []
            projected_publications = []
            pub_date = []
            entry_date = []

        if method in [ 'scikit-learn', 'stemmed content words' ]:

            str_fn = {
                'scikit-learn': 'points_str',
                'stemmed content words': 'primary_stemmed_points_str',
            }
            # Compile text
            print( '    Retrieving publication data...' )
            abstracts = []
            for key, item in tqdm( self.data.items() ):
                abstracts.append( getattr( item, str_fn[method] )() )
                projected_publications.append( key )
                pub_date.append( item.publication_date )
                try:
                    entry_date.append( str( item.entry_date ) )
                except AttributeError:
                    entry_date.append( 'NaT' )

            print( '    Calculating vectorization...' )
            vectorizer = skl_text_features.CountVectorizer( token_pattern=r"(?u)\b[A-Za-z]+\b" )
            vectors = vectorizer.fit_transform( abstracts )
            feature_names = vectorizer.get_feature_names_out()

            # Calculate the norm
            norm_squared_unformatted = vectors.multiply( vectors ).sum( axis=1 )
            norm = np.sqrt( np.array( norm_squared_unformatted ).flatten() )

            # Sort vector indices.
            # Necessary for some of the c/c++ backend.
            vectors.sort_indices()

            if not sparse:
                vectors = vectors.toarray()

        elif method == 'homebuilt':

            # Loop through and calculate vectors
            for key, item in tqdm( self.data.items() ):

                # Don't reproject existing publications
                if key in projected_publications:
                    continue

                vector_i, feature_names = item.vectorize(
                    feature_names = feature_names,
                )
                vectors_list.append( vector_i )
                projected_publications.append( key )
                pub_date.append( item.publication_date )
                try:
                    entry_date.append( str( item.entry_date ) )
                except AttributeError:
                    entry_date.append( 'NaT' )

            # Format vectors
            shape = (
                len( projected_publications ),
                len( feature_names )
            )
            vectors = np.zeros( shape )
            for i, component in enumerate( vectors_list ):
                vectors[i,:len(component)] = component

            # Normalized vectors
            norm = np.linalg.norm( vectors, axis=1 )
        else:
            raise NameError( 'Unrecognized vectorize method, method={}'.format( method ) )

        # Store
        self.projection = verdict.Dict( {
            'vectors': vectors,
            'norms': norm,
            'feature_names': np.array( feature_names ).astype( str ),
            'publications': np.array( projected_publications ),
            'publication_dates': np.array( pub_date ),
            'entry_dates': np.array( entry_date ),
        } )
        if projection_fp != 'pass':
            if sparse:
                self.projection['vectors'] = ss.csr_matrix( vectors )
            self.projection.to_hdf5( projection_fp, sparse=sparse )
            # Convert back...
            if not sparse:
                if ss.issparse( vectors ):
                    vectors = vectors.toarray()
                self.projection['vectors'] = vectors

        if return_data:
            return self.projection

    ########################################################################
    # Publication-to-publication comparison
    ########################################################################

    def inner_product_custom( self, other, **kwargs ):
        '''Calculate the inner product with another object.
        This is much more customizable than inner_product, but much, much
        slower.
        '''

        inner_product = 0

        # When the other object is a publication
        # isinstance raises false exceptions
        is_pub = (
            isinstance( other, publication.Publication ) or
            str( type( other ) ) == "<class 'cc.publication.Publication'>"
        )
        if is_pub:
            for p in self.data.values():
                inner_product += other.inner_product_custom( p, **kwargs )

        # When the other object is an atlas
        elif str( type( other ) ) == "<class 'cc.atlas.Atlas'>":
            for p_self in self.data.values():
                for p_other in self.data.values():
                    inner_product += p_other.inner_product_custom( p_self, **kwargs )
        else:
            raise ValueError( "Unrecognized object for calculating the inner product, {}".format( other ) )

        if inner_product == 0:
            warnings.warn( "Inner product == 0. Did you forget to load the data?" )

        return inner_product

    ########################################################################

    def cospsi_data( self, other, **kwargs ):
        '''Calculate the cos(psi) between the atlas' data and another object.
        psi is the "angle" between two objects, defined as
        cos( psi ) = <self | other> / sqrt( <self | self> * <other | other> )

        Args:
            other:
                The other object.

            **kwargs:
                Keyword arguments passed to inner_product.

        Returns:
            cospsi (verdict.Dict of floats or ints):
                cos(psi) calculated for each item of self.data.
        '''

        ### Calculate cospsi
        # Inner products
        ip_self = {}
        ips = {}
        for key, p in self.data.items():
            ip_self[key] = p.inner_product_custom( p, **kwargs )
            ips[key] = p.inner_product_custom( other, **kwargs )
        ip_self = verdict.Dict( ip_self )
        ip_other = other.inner_product_custom( other, **kwargs )
        ips = verdict.Dict( ips )

        # Cospsi
        cospsi = ips / ( ip_self * ip_other ).apply( np.sqrt )

        return cospsi

    ########################################################################
    # Plots
    ########################################################################

    def plot_cospsi2d(
        self,
        x_key,
        y_key,
        ax = None,
        x_kwargs = {},
        y_kwargs = {},
        **kwargs
    ):
        '''Scatter plot cos(psi) of two objects calculated with all the
        publications in the library.

        Args:
            x_obj:
                The x object to calculate cos(psi) with.

            y_obj:
                The y object to calculate cos(psi) with.

            ax:
                The axis to place the plot on.

            **kwargs:
                Keyword arguments to pass to the inner products.

        Returns:
            cospsi_x:
                Dictionary of values for the x_obj.

            cospsi_y:
                Dictionary of values for the y_obj.
        '''

        ### Calculate cospsi
        used_x_kwargs = copy.deepcopy( kwargs )
        used_x_kwargs.update( x_kwargs )
        used_y_kwargs = copy.deepcopy( kwargs )
        used_y_kwargs.update( y_kwargs )
        ip_xall = self.inner_product( x_key, 'all', **used_x_kwargs )
        ip_yall = self.inner_product( y_key, 'all', **used_y_kwargs )
        ip_xs = self.inner_product( x_key, x_key, **used_x_kwargs )
        ip_ys = self.inner_product( y_key, y_key, **used_y_kwargs )
        ip_xallall = self.inner_product( 'all', 'all', **used_x_kwargs )
        ip_yallall = self.inner_product( 'all', 'all', **used_y_kwargs )
        cospsi_xs = ip_xall / np.sqrt( ip_xs * ip_xallall )
        cospsi_ys = ip_yall / np.sqrt( ip_ys * ip_yallall )

        # Setup figure
        if ax is None:
            fig = plt.figure( figsize=(8,8), facecolor='w' )
            ax = plt.gca()

        # Plot
        ax.scatter(
            cospsi_xs,
            cospsi_ys,
            color = 'k',
            s = 50,
        )

        # Labels
        ax.set_xlabel( r'$\cos \psi$(' + str( x_key ) + ')', fontsize=22 )
        ax.set_ylabel( r'$\cos \psi$(' + str( y_key ) + ')', fontsize=22 )

        # Axis tweaks
        ax.set_xlim( 0, 1 )
        ax.set_ylim( 0, 1 )
        ax.set_aspect( 'equal' )

        return cospsi_xs, cospsi_ys

    ########################################################################

    def plot_cospsi2d_custom(
        self,
        x_obj,
        y_obj,
        ax = None,
        x_kwargs = {},
        y_kwargs = {},
        **kwargs
    ):
        '''Scatter plot cos(psi) of two objects calculated with all the
        publications in the library.

        Args:
            x_obj:
                The x object to calculate cos(psi) with.

            y_obj:
                The y object to calculate cos(psi) with.

            ax:
                The axis to place the plot on.

            **kwargs:
                Keyword arguments to pass to the inner products.

        Returns:
            cospsi_x:
                Dictionary of values for the x_obj.

            cospsi_y:
                Dictionary of values for the y_obj.
        '''

        ### Calculate cospsi
        used_x_kwargs = copy.deepcopy( kwargs )
        used_x_kwargs.update( x_kwargs )
        used_y_kwargs = copy.deepcopy( kwargs )
        used_y_kwargs.update( y_kwargs )
        cospsi_xs = self.cospsi_data( x_obj, **used_x_kwargs )
        cospsi_ys = self.cospsi_data( y_obj, **used_y_kwargs )

        # Setup figure
        if ax is None:
            fig = plt.figure( figsize=(8,8), facecolor='w' )
            ax = plt.gca()

        # Plot
        xs = cospsi_xs.array()
        ys = cospsi_ys.array()
        ax.scatter(
            xs,
            ys,
            color = 'k',
            s = 50,
        )

        # Labels
        ax.set_xlabel( r'$\cos \psi$(' + str( x_obj ) + ')', fontsize=22 )
        ax.set_ylabel( r'$\cos \psi$(' + str( y_obj ) + ')', fontsize=22 )

        # Axis tweaks
        ax.set_xlim( 0, 1 )
        ax.set_ylim( 0, 1 )
        ax.set_aspect( 'equal' )

        return cospsi_xs, cospsi_ys

########################################################################

def store_s2_data( atlas_pub, paper, fields = api.S2_STORE_FIELDS ):
    '''Store a Semantic Scholar Paper as an Publication for an Atlas.

    Args:
        atlas_pub (Publication): a publication for an atlas

        paper (Paper): a Semantic Scholar Paper object

        fields (list of strs): the fields returned from a S2 query for a paper

    Returns: 
        atlas_pub
    '''

    # Store
    atlas_pub.s2_data = {}
    for field in fields:
        value = getattr( paper, field )

        # Formatting choice, abstract = None replaced
        # with abstract = ''
        if field == 'abstract' and value is None:
            value = ''

        atlas_pub.s2_data[field] = value

        attr_f = copy.copy( field )

        setattr( atlas_pub, attr_f, value ) # I'm not actually convinced that the right thing to do is to alter the Semantic Scholar paper object to have mutable fields, like title and abstract, but I'll do it until it becomes a problem.

    return atlas_pub 

########################################################################

# TODO: probably create a bibtex submodule
########################################################################

def save_ids_to_bibtex( *args, api_name = api.DEFAULT_API, **kwargs, ):
    '''Use ADS or S2 to convert publication identifiers into bibtex files.'''

    api.validate_api(api_name)

    if api_name == api.ADS_API_NAME:
        save_ads_bibcodes_to_bibtex( *args, **kwargs )
    if api_name == api.S2_API_NAME:
        save_s2_paperids_to_bibtex( *args, **kwargs )

########################################################################

def save_s2_paperids_to_bibtex( paper_ids, **kwargs, ):
    papers = api.call_s2_api( paper_ids, **kwargs )
    save_s2_papers_to_bibtex( papers, **kwargs )

########################################################################    

def save_s2_papers_to_bibtex( papers, bibtex_fp = api.S2_BIB_NAME, overwrite_bibtex = False, **kwargs, ):
    '''Extracts bibtex entries from Semantic Scholar Papers, reformatting them to store ids.
    
    Args:
        batch (bool): whether to call the SemanticScholar API using `get_paper` or `get_papers`. The latter is faster, but usually fails, and also does not have tqdm support yet.
    '''
    # Get bibtex and format
    all_entries = []

    for i, paper in enumerate(papers):

        entry = s2_paper_to_bib_entry(paper)
        
        all_entries.append(entry)
    
    db = BibDatabase()
    db.entries = all_entries

    # Save the bibtex
    writer = BibTexWriter()
    if overwrite_bibtex:
        mode = 'w'
    else:
        mode = 'a' # append
    with open( bibtex_fp, mode ) as bibfile:
        bibfile.write( writer.write(db) )

########################################################################    

def s2_paper_to_bib_entry( paper ) -> dict:
    '''Parse a Paper and fill out a bib entry.'''

    parser = BibTexParser(common_strings=False)
    parser.ignore_nonstandard_types = False
    parser.homogenize_fields = False
    parser.expect_multiple_parse = True # absolutely critical

    bibtex_str = paper.citationStyles['bibtex']

    # Format for bibtexparser
    # S2 entry types are garbage and we will replace ID anyway
    bibtex_str = "".join(['@ARTICLE{dummy_key,'] + bibtex_str.split('\n')[1:])
    db = bibtexparser.loads(bibtex_str, parser)

    if not db.entries:
        raise Exception(f'Bibtexparser could not parse:\n {bibtex_str}')
    
    entry = db.entries[-1] # one paper at a time; not optimized.
    externalIds = paper.externalIds
    date = paper.publicationDate

    # Alter the bib entry fields
    # Title
    title = entry['title']
    entry['title'] = f'{{{title}}}' # Zach puts double brackes around the title

    # ID
    entry['ID'] = paper.paperId # NOTE: possibly paper.paperId != query id!

    entry['paperid'] = paper.paperId # if it was found at all, it has a paperId, (I think)

    # Other identifiers
    xids_map = api.S2_EXTERNAL_ID_TO_BIBFIELD
    if externalIds is not None:
        for xid in xids_map:
            if xid in externalIds:
                # CorpusId is originally an int, but bibtexparser needs str
                entry[xids_map[xid]] = str(externalIds[xid])
    
    # including URL
    if paper.url is not None:
        entry[xids_map['URL']] = paper.url

    # Date
    if date is not None:
        entry['year'] = str(date.year) # year probably is there already, no harm
        entry['month'] = str(date.month)
        entry['day'] = str(date.day)

    # Abstract (useful for recovering og string later)
    if paper.abstract is not None:
        entry['abstract'] = paper.abstract

    return entry    

########################################################################

def save_ads_bibcodes_to_bibtex( 
    bibcodes, 
    bibtex_fp = api.ADS_BIB_NAME,  
    call_size=2000, 
    overwrite_bibtex = False, 
    ):
    ## API_extension::to_and_from_bibcodes

    # ADS doesn't like np arrays
    bibcodes = list( bibcodes )

    # Break into chunks
    chunked_bibcodes = api.chunk_ids(ids=bibcodes, call_size=call_size)

    bibtex_str = ''
    for bibcodes in chunked_bibcodes:

        # Retrieve data from ADS
        try:
            q = ads.ExportQuery( bibcodes )
            bibtex_str_i = q.execute()
        # Try again if there was an error on the ADS side
        except ads.exceptions.APIResponseError:
            q = ads.ExportQuery( bibcodes )
            bibtex_str_i = q.execute()

        # Reformat some lines to work with bibtexparser
        # This is not optimized.
        l = []
        for line in bibtex_str_i.split( '\n' ):
            # ADS puts quotes instead of double brackes around the title
            if 'title =' in line:
                line = line.replace( '"{', '{{' )
                line = line.replace( '}"', '}}' )
            # ADS' bib export for months doesn't have brackets around it...
            if 'month =' in line:
                line = line.replace( '= ', '= {' ).replace( ',', '},' )
            # The eprint is usually also the arxivid.
            if 'eprint =' in line:
                l.append( line.replace( 'eprint', 'arxivid' ) )
            l.append( line )
        bibtex_str += '\n'.join( l )

    # Save the bibtex
    if overwrite_bibtex:
        mode = 'w'
    else:
        mode = 'a' # append

    with open( bibtex_fp, mode ) as f:
        f.write( bibtex_str )

########################################################################

def create_random_atlas_from_pubs(
    atlas_dir, 
    pubs, 
    fl,
    api_name = api.DEFAULT_API,
) -> Atlas:
    '''Constructs an atlas from a random list of publications, by accessing API-specific data.
    
    Args:
        atlas_dir (str):
            Directory to store the atlas data in.
        
        pubs (list): 
            the publications resulting from calling util.random_publications with some API

        fl (list):
            list of fields to populate in each atlas publication from retrieved pubs
            
        api_name (str):
            the API to call. 

    Returns:
        result (Atlas) the random Atlas
    '''
    api.validate_api(api_name)
    if api_name == 'ADS':
        return create_random_atlas_from_pubs_ads( atlas_dir, pubs, fl )
    if api_name == 'S2':
        return create_random_atlas_from_pubs_s2( atlas_dir, pubs, fl )

########################################################################

def create_random_atlas_from_pubs_ads(
    atlas_dir, 
    pubs, 
    fl,
    ) -> Atlas:
    '''Constructs an atlas from a random list of publications, by accessing ADS specific data.'''

    bibcodes = [ _.bibcode for _ in pubs ] # ADS specifc
    result = Atlas.to_and_from_ids( atlas_dir, bibcodes, api_name = api.ADS_API_NAME )
    
    ## API_extension:random_publications
    ## The below code block stores already retrieved values in the atlas.
    ## The way a random atlas is made currently is to retrieve the publications,
    ## use their bibcodes to make an easy list of them,
    ## pass in and format the already retrieved data.
    # Store publication data
    for p_ads in pubs:
        
        if p_ads.bibcode not in result.data:
            continue

        p = result.data[p_ads.bibcode]

        p.ads_data = {}
        for f in fl:
            value = getattr( p_ads, f )
            p.ads_data[f] = value
            attr_f = copy.copy( f )
            if attr_f == 'citation' or attr_f == 'reference':
                attr_f += 's'
            setattr( p, attr_f, value )

    return result

########################################################################

def create_random_atlas_from_pubs_s2(*args, **kwargs) -> Atlas:
    '''Constructs an atlas from a random list of publications, by accessing S2 specific data.'''
    raise NotImplementedError