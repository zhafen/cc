'''This file contains code for interfacing cc with various APIs.

Required changes below. Search the files for the given flag name to find relevant areas that need to be changed.
The below text is not guaranteed to contain everything we need to change,
and it wouldn't be surprising if I misclassified or classified twice.

get_data_via_api:
    The most basic functionality we use.
    There will be three main types of calls I expect to show up.
    get_data_via_api_general --- the base call that is what shows up most places in the code.
    get_ads_data --- the ADS specific call. Probably should only be used by get_data_via_api_general
    get_ss_data --- The SS specific call. Same deal as ADS.

process_data:
    Take downloaded data and do something with it. Typically relies on the specific format of the downloaded data.

to_and_from_bibcodes:
    Bibcodes are the identifiers ADS uses. cc has the functionality to turn a list of bibcodes into an Atlas,
    and related functionality for turning bibcodes into a bibtex file.
    The general analog to this will probably need to be from_ids and similar.

random_publications:
    The functionality for retrieving random publications is currently ADS specific.

publication_date:
    The publication date is currently set to the date the publication was posted on arXiv.
    We'll need to make it more general.

default_bib_name:
    Change the default name of the .bib file created during algorithmic retrieval of data.
    Current name is cc_ads.bib.

maybe_unnecessary:
    This code should probably be removed after the extension is finished.

no_change:
    This code uses ADS, but should not be touched. Probably because it's not used.
'''

import ads
import semanticscholar
# import utils # literature-topography doesn't like
# from . import utils

import numpy as np
import pandas as pd

from functools import wraps
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
from tqdm import tqdm

# constants
ADS_BIB_NAME = 'cc_ads.bib'
ADS_API_NAME = 'ADS'
ADS_ALLOWED_EXCEPTIONS = (ads.exceptions.APIResponseError, )

S2_API_NAME = 'S2'
S2_BIB_NAME = 'cc_s2.bib'
import requests
S2_ALLOWED_EXCEPTIONS = (
    requests.exceptions.ReadTimeout, 
    requests.exceptions.ConnectionError,
    semanticscholar.SemanticScholarException.ObjectNotFoundExeception,
    )

########################################################################
# Semantic Scholar paper Identifiers. Please read https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_get_paper

# The following types of IDs are supported
S2_EXTERNAL_IDS  = [
    'DOI', 
    'ArXiv', 
    'CorpusId',
    'MAG', 
    'ACL', 
    'PubMed', 
    'Medline', 
    'PubMedCentral', 
    'DBLP',
    'URL',
    ]

# URLs are recognized from the following sites:
#    semanticscholar.org
#    arxiv.org
#    aclweb.org
#    acm.org
#    biorxiv.org

# For storing bib entries
S2_EXTERNAL_ID_TO_BIBFIELD = {key: key.lower() for key in S2_EXTERNAL_IDS}
# arxiv could also have value 'eprint', perhaps
S2_EXTERNAL_ID_TO_BIBFIELD['ArXiv'] = 'arxivid'

# For querying api from bib entries
S2_BIBFIELD_TO_API_QUERY = {S2_EXTERNAL_ID_TO_BIBFIELD[key]: key.upper() for key in S2_EXTERNAL_IDS}
S2_BIBFIELD_TO_API_QUERY['arxivid'] = 'ARXIV'
S2_BIBFIELD_TO_API_QUERY['pubmed'] = 'PMID' # includes Medline
S2_BIBFIELD_TO_API_QUERY['pubmedcentral'] = 'PMCID' # includes Medline


# For querying api from references/citations of Papers
S2_EXTERNAL_ID_TO_API_QUERY = {
    key: S2_BIBFIELD_TO_API_QUERY[S2_EXTERNAL_ID_TO_BIBFIELD[key]] for key in S2_EXTERNAL_IDS
}

########################################################################

# NOTE: semantic scholar will truncate total number of references, citations each at 10,000 for the entire batch.
S2_QUERY_FIELDS = [
    'abstract',
    'externalIds', # supports ArXiv, MAG, ACL, PubMed, Medline, PubMedCentral, DBLP, DOI
    'url', # as a possible external id
    'citations.externalIds',
    'citations.url',
    'references.externalIds',
    'references.url',
    'citationStyles', # supports a very basic bibtex that we will augment
    'publicationDate', # if available, type datetime.datetime (YYYY-MM-DD)
]

# for storing the results from above, we avoid dot operator to avoid attribute error, but note that everything above will be included.
S2_STORE_FIELDS = [
    'abstract',
    'externalIds', 
    'url', 
    'citations',
    'references',
    'citationStyles', 
    'publicationDate', 
]

# Attributes to save via save_data
S2_ATTRS_TO_SAVE = [
    'paper', 
    'abstract',
    'citations',
    'references',
    'bibcode',
    'entry_date',
    'notes',
    'unofficial_flag',
    'citation',
    'stemmed_content_words',
]

########################################################################

DEFAULT_BIB_NAME = ADS_BIB_NAME
DEFAULT_API = ADS_API_NAME
DEFAULT_ALLOWED_EXCEPTIONS = ADS_ALLOWED_EXCEPTIONS

########################################################################

def validate_api(api: str) -> None:
    apis_allowed = ['S2', 'ADS']
    if api not in apis_allowed:
        raise ValueError(f"No support for {api}. Allowed API options include {apis_allowed}")

########################################################################

def chunk_ids(ids: list[str], call_size = 2000):
    '''Helper function to chunk bibcodes or paperIds into smaller sublists if appropriate.'''
    # Currently ADS-specific chunking limit, but assume it is a good heuristic for S2, too.

    # Break into chunks
    assert call_size <= 2000, 'Max number of calls ExportQuery can handle at a time is 2000.'
    if len( ids ) > call_size:
        chunked_ids = [ ids[i:i + call_size] for i in range(0, len(ids), call_size) ]
    else:
        chunked_ids = [ ids, ]
    
    return chunked_ids

########################################################################

def call_s2_api(
        paper_ids, 
        batch = False,
        n_attempts_per_query = 10,
        call_size = 100,
        **kwargs,
        ):
    '''Get papers from SemanticScholar by calling the API.

    Args:

        paper_ids (list of strs): the identifiers required for querying API

        batch (bool) whether to call the SemanticScholar API using `get_paper` or `get_papers`. The latter is faster, but usually fails, and also does not have tqdm support yet.

        n_attempts_per_query (int):
            Number of attempts to access the API per query. Useful when experiencing connection issues.

        call_size: (int): maximum number of papers to call API for in one query; if less than `len(paper_ids)`, chunking will be performed.
    '''

    sch = SemanticScholar()

    paper_ids = list( paper_ids ) # in case numpy array

    chunked_paper_ids = chunk_ids(paper_ids, call_size = call_size)

    if None in paper_ids:
        # Since we should have already dropped all Nones
        raise Exception("Passed `paper_ids` contains None.")


    # how external ids should be in bibtex entry

    print( f'Querying Semantic Scholar for {len(paper_ids)} total papers.')

    papers = []
    for i, paper_ids in enumerate(chunked_paper_ids):
        
        if len(chunked_paper_ids) > 1:
            print( f'querying for {len(paper_ids)} papers; chunk {i+1} out of {len(chunked_paper_ids)}')

        @keep_trying( n_attempts=n_attempts_per_query, )
        def get_papers(batch) -> list[Paper]:
            # NOTE: risk of "requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='api.semanticscholar.org', port=443): Read timed out"
            if batch:
                # Faster to query in batch, but often fails.
                result = sch.get_papers(
                    paper_ids=paper_ids,
                    fields=S2_QUERY_FIELDS,
                )
            else:
                # typically completes about 100 queries per minute.
                result = [
                    sch.get_paper(
                    paper_id=paper_id, 
                    fields=S2_QUERY_FIELDS,
                    ) for paper_id in tqdm(paper_ids)
                ]
            return result

        papers.extend(get_papers(batch))

    return papers

########################################################################

def dict_to_s2_paper( item ) -> Paper:
    '''Parse a dict and convert to a Semantic Scholar Paper; intended for loading publications from atlas_data files.'''
    paper = Paper(item)
    if not paper._data:
        raise Exception("Empty paper.")
    return paper

########################################################################

def citation_to_s2_paper( citation: dict ) -> Paper:
    '''Parse a bibtex entry and convert to a Semantic Scholar Paper.'''

    data = {}

    if 'title' in citation:
        data['title'] = citation['title']

    if 'ID' in citation:
        data['paperId'] = citation['ID']

    data['externalIds'] = {}
    for xid in S2_EXTERNAL_ID_TO_BIBFIELD:
        if xid == 'URl':
            continue
        if S2_EXTERNAL_ID_TO_BIBFIELD[xid] in citation:
            data['externalIds'][xid] = citation[S2_EXTERNAL_ID_TO_BIBFIELD[xid]]

    if 'url' in citation:
        data['url'] = citation['url']

    if 'abstract' in citation:
        data['abstract'] = citation['abstract']

    # don't bother with datetime for now
    paper = Paper(data)
    if not paper._data:
        raise Exception("Empty paper.")

    return paper

########################################################################

def citation_to_api_call( citation: dict, api_name = DEFAULT_API ) -> tuple:
    '''Given a dictionary containing a citation return a string that, when sent to S2AG, will give a unique result.
    '''
    validate_api(api_name)
    if api_name == ADS_API_NAME:
        return citation_to_ads_call( citation)
    if api_name == S2_API_NAME:
        return citation_to_s2_call( citation )

########################################################################

def citation_to_s2_call( citation ):
    '''Given a dictionary containing a citation return a string that, when sent to S2AG, will give a unique result.

    For now, we'll only use doi, as a proof of concept.
    
    Args:
        citation (dict):
            Dictionary containing the citation information for a publication.

    Returns: 
        paper_id (str):
            String to be used for the mandatory paperId arg for S2 query.

    '''
    # Parse citation for any viable s2 identifier.

    # prioritize paperid
    if 'paperid' in citation:
        return citation['paperid'] # no need to format

    # then ArXiv
    if 'arxivid' in citation:
        return f"ARXIV:{citation['arxivid']}"

    # then DOI (I have identified some incorrect doi strings!)
    if 'doi' in citation:
        return f"DOI:{citation['doi']}"

    # search for other viable ids
    for xid in S2_BIBFIELD_TO_API_QUERY:
        if xid in citation:
            print(f'Using externalId {xid} for s2 query.')
            return f"{S2_BIBFIELD_TO_API_QUERY[xid]}:{citation[xid]}"
    
    # would return None otherwise; so what is in this entry? Probably just ads stuff?
    # breakpoint()
    raise Exception("citation has no viable s2 identifier")

########################################################################

def citation_to_ads_call( citation ):
    '''Given a dictionary containing a citation return a string that,
    when sent to ADS, will give a unique result.

    ## API_extension::get_data_via_api
    ## Need a general function and an analogous function for S2

    Args:
        citation (dict):
            Dictionary containing the citation information for a publication.

    Returns:
        q (str):
            String to be used as a query for ADS.

        ident (str):
            Type of identifier used.

        id (str):
            ID used.
    '''

    q = ''

    # Check if we should use arXiv to identify
    if 'eprint' in citation:
        use_arxiv = True
        # When we can, check that the eprint is of the correct type
        if 'eprinttype' in citation:
            use_arxiv = citation['eprinttype'] == 'arxiv'
            # if not use_arxiv:
            #     warnings.warn(
            #         'non-arxiv eprint, eprinttype={} eprint={}'.format(
            #         citation['eprinttype'],
            #         citation['eprint']
            #     )
    else:
        use_arxiv = False

    if use_arxiv:

        ident = 'arxiv'
        id = citation['eprint']

        # If an updated version of the publication
        if 'v' in id:
            id = id.split( 'v' )[0]

        # If the id has the category in it we only want the part after the /
        # but that's only if it's not the old type of arxiv ID
        id_tail = id.split( '/' )[-1]
        if '.' in id_tail:
            id = id_tail

        q = '{}:"{}"'.format( ident, id )

    elif 'doi' in citation:
        # Weird edgecase where there are extra semicolons
        id = citation['doi'].replace( ';', '' )
        ident = 'doi'
        q = '{}:"{}"'.format( ident, id )

    # Search using multiple other identifiers
    else:
        ident = []
        id = []
        if 'author' in citation:
            authors = citation['author'].split( ' and ' )
            for author in authors:
                # Handle when brackets are included
                if '{' in author and '}' in author:
                    author = author.replace( '{', '' )
                    author = author.replace( '}', '' )
                    author = author.split( ' ' )[0]

                # Space padding
                if q!= '': q += ' '

                ident.append( 'author' )
                id.append( author )
                q += 'author:"{}"'.format( author )

        if 'volume' in citation:
            # Space padding
            if q!= '': q += ' '

            ident.append( 'volume' )
            id.append( citation['volume'] )
            q += 'volume:"{}"'.format( citation['volume'] )

        if 'pages' in citation:
            # ADS only recognizes the first page.
            starting_page = citation['pages'].split( '-' )[0]

            # Space padding
            if q!= '': q += ' '

            ident.append( 'pages' )
            id.append( starting_page )
            q += 'page:"{}"'.format( starting_page )

    if q == '':
        raise Exception( 'No valid identifiers found.' )

    return q, ident, id

########################################################################

def keep_trying( n_attempts=5, allowed_exceptions = DEFAULT_ALLOWED_EXCEPTIONS, verbose=True ):
    '''Sometimes we receive server errors. We don't want that to disrupt the entire process, so this decorator allow trying n_attempts times.

    ## API_extension::get_data_via_api
    ## This decorator is general, except for the default allowed exception.

    Args:
        n_attempts (int):
            Number of attempts before letting the exception happen.

        allowed_exceptions (tuple of class):
            Allowed exception class. Set to BaseException to keep trying regardless of exception.

        verbose (bool):
            If True, be talkative.

    Example Usage:
        > @keep_trying( n_attempts=4 )
        > def try_to_call_web_api():
        >     " do stuff "
    '''

    def _keep_trying( f ):

        @wraps( f )
        def wrapped_fn( *args, **kwargs ):
            # Loop over for n-1 attempts, trying to return
            for i in range( n_attempts - 1 ):
                try:
                    result = f( *args, **kwargs )
                    if i > 0 and verbose:
                        print( 'Had to call {} {} times to get a response.'.format( f, i+1 ) )
                    return result
                except allowed_exceptions as _:
                    continue

            # On last attempt just let it be
            if verbose:
                print( 'Had to call {} {} times to get a response. Trying once more.'.format( f, n_attempts ) )
            return f( *args, **kwargs )

        return wrapped_fn

    return _keep_trying

########################################################################

def api_query(*args, api_name = DEFAULT_API, **kwargs ) -> list:
    '''Convenience wrapper for searching an API.'''
    validate_api(api_name)
    if api_name == ADS_API_NAME:
        return ads_query( *args, **kwargs )
    if api_name == S2_API_NAME:
        return s2_query( *args, **kwargs )

########################################################################

@keep_trying()
def s2_query(
    q,
    fl = ['abstract', 'citation', 'reference', 'entry_date', 'identifier' ],
    rows = 50
):
    '''Convenience wrapper for searching S2.

    Args:
        q (str):
            Call to S2.

        fl (list of strs):
            Fields to return for publications.

        rows (int):
            Number of publications to return per page.
    '''
    raise NotImplementedError

########################################################################

@keep_trying()
def ads_query(
    q,
    fl = ['abstract', 'citation', 'reference', 'entry_date', 'identifier' ],
    rows = 50
):
    '''Convenience wrapper for searching ADS.

    ## API_extension::get_data_via_api
    ## Need a general version of this function and a specific one for S2.

    Args:
        q (str):
            Call to ADS.

        fl (list of strs):
            Fields to return for publications.

        rows (int):
            Number of publications to return per page.
    '''

    ads_query = ads.SearchQuery(
        query_dict={
            'q': q,
            'fl': fl,
            'rows': rows,
        },
    )
    query_list = list( ads_query )

    return query_list

########################################################################

def random_publications(*args, api_name = DEFAULT_API, **kwargs,):
    '''Choose random publications by choosing a random date and then choosing a random publication announced on that date, via some API.'''

    validate_api(api_name)
    if api_name == ADS_API_NAME:
        return random_publications_ads(*args, **kwargs)
    
    elif api_name == S2_API_NAME:
        return random_publications_s2(*args, **kwargs)

########################################################################

def random_publications_s2(*args, **kwargs):
    '''Choose random publications by choosing a random date and then choosing a random publication announced on that date.'''
    raise NotImplementedError

########################################################################

def random_publications_ads(
    n_sample,
    start_time,
    end_time,
    fl = [ 'arxivid', 'doi', 'date', 'citation', 'reference', 'abstract', 'bibcode', 'entry_date', 'arxiv_class' ],
    arxiv_class = None,
    seed = None,
    max_loops = None,
    bad_days_of_week = [ 'Saturday', 'Sunday' ],
    n_attempts_per_query = 5,
    verbose = False,
):
    '''Choose random publications by choosing a random date and then choosing
    a random publication announced on that date.
    Note that while this means that publications announced on the same date
    as many other publications are less likely to be selected, this is not
    expected to typically be an important effect.

    ## API_extension::random_publications

    Args:
        n_sample (int):
            Number of publications to sample.

        start_time (pd.Timestamp or pd-compatible string):
            Beginning time to use for the range of selectable publications.

        end_time (pd.Timestamp or pd-compatible string):
            End time to use for the range of selectable publications.

        arxiv_class (str):
            What arxiv class the publications should belong to, if any.
            If set to 'astro-ph' it will include all subcategories of 'astro-ph'.

        seed (int):
            Integer to use for setting the random number selection.
            Defaults to not being used.

        max_loops (int):
            Number of iterations before breaking. Defaults to 20 * n_sample.

        n_attempts_per_query (int):
            Number of attempts to access the API per query. Useful when experiencing
            connection issues.

        verbose (bool):
            The usual switch to turn on/off lots of messages.

    Returns:
        pubs (list of ads queries):
            Publications selected.
    '''

    if not isinstance( start_time, pd.Timestamp ):
        start_time = pd.to_datetime( start_time )
    if not isinstance( end_time, pd.Timestamp ):
        end_time = pd.to_datetime( end_time )

    if seed is not None:
        np.random.seed( seed )

    if max_loops is None:
        max_loops = 20 * n_sample

    search_str = ''

    if arxiv_class is not None:
        search_str += 'arxiv_class:"{}"'.format( arxiv_class )
        if arxiv_class == 'astro-ph':
            subcats = [ 'GA', 'CO', 'EP', 'HE', 'IM', 'SR' ]
            for subcat in subcats:
                search_str += ' OR arxiv_class:"astro-ph.{}"'.format( subcat )
        else:
            subcats = []

    pubs = []
    n_loops = 0
    pbar = tqdm( total=n_sample, position=0, leave=True )
    empty_dates = []
    empty_abstracts = []
    no_refs_or_cits = []
    not_right_class = []
    api_response_errors = []
    while len( pubs ) < n_sample:

        # Build query
        query_dict = dict(
            fl = fl,
        )

        if n_loops > max_loops:
            tqdm.tqdm.write( 'Reached max number of loops, {}. Breaking.'.format( max_loops ) )
            break
        n_loops += 1
        
        # Generate a random datetime, skipping bad days of the week
        while True:
            random_datetime = pd.to_datetime( np.random.randint(
                    start_time.value,
                    end_time.value,
                    1,
                    dtype=np.int64
                )[0],
            )
            if random_datetime.day_name() not in bad_days_of_week:
                break
        random_date = '{}-{}-{}'.format( random_datetime.year, random_datetime.month, random_datetime.day )
        if search_str == '':
            query_dict['entdate'] = random_date

            # Get publications out. Turned into a function and
            # wrapped to allow multiple attempts.
            @keep_trying( n_attempts=n_attempts_per_query )
            def get_pubs_for_query():
                ads_query = ads.SearchQuery( **query_dict )
                query_list = list( ads_query )
                return query_list

        else:
            random_datetime_end = random_datetime + pd.DateOffset( days=1 )
            random_date_end = '{}-{}-{}'.format(
                random_datetime_end.year,
                random_datetime_end.month,
                random_datetime_end.day
            )
            query_dict['q'] = search_str + ' entdate:[{} TO {}]'.format( random_date, random_date_end )

            # Get publications out. Turned into a function and
            # wrapped to allow multiple attempts.
            @keep_trying( n_attempts=n_attempts_per_query )
            def get_pubs_for_query():
                ads_query = ads.SearchQuery( query_dict=query_dict )
                query_list = list( ads_query )
                return query_list

        query_list = get_pubs_for_query()

        if len( query_list ) == 0:
            empty_dates.append( random_datetime )
            continue

        p = np.random.choice( query_list )
        
        # Cannot do this for publications missing abstract data.
        if p.abstract is None:
            empty_abstracts.append( p )
            if verbose:
                tqdm.tqdm.write( 'Publication {} has no abstract. Continuing.'.format( p.bibcode ) )
            continue
        
        # Cannot do this for publications missing citation data.
        if p.citation is None and p.reference is None:
            no_refs_or_cits.append( p )
            if verbose:
                tqdm.tqdm.write( 'Publication {} has no references or citations. Continuing.'.format( p.bibcode ) )
            continue

        # If the *primary* class is not the target arxiv_class, continue
        if arxiv_class is not None:
            viable_classes = [ arxiv_class, ] + [ '{}.{}'.format( arxiv_class, _ ) for _ in subcats ]
            if not p.arxiv_class[0] in viable_classes:
                not_right_class.append( p )
                tqdm.write( 'Publication {} is not the right arxiv category. Continuing.'.format( p.bibcode ) )
                continue
        
        pubs.append( p )
        pbar.update( 1 )
    pbar.close()

    print( 'Retrieved {} random publications. Took {} tries'.format( len( pubs ), n_loops ) )

    return pubs

########################################################################
