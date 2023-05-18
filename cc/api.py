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
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper

from tqdm import tqdm

# constants
ADS_BIB_NAME = 'cc_ads.bib'
ADS_API_NAME = 'ADS'
ADS_ALLOWED_EXCEPTION = ads.exceptions.APIResponseError

S2_API_NAME = 'S2'
S2_BIB_NAME = 'cc_s2.bib'

########################################################################
# Semantic Scholar paper Identifiers. Please read https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_get_paper

# The following types of IDs are supported
S2_EXTERNAL_IDS  = [
    'DOI', 
    'ArXiv', 
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
S2_BIBFIELD_TO_API_QUERY = {key: key.upper() for key in S2_EXTERNAL_ID_TO_BIBFIELD}
S2_BIBFIELD_TO_API_QUERY['PUBMED'] = 'PMID' # includes Medline
S2_BIBFIELD_TO_API_QUERY['PUBMEDCENTRAL'] = 'PMCID' # includes Medline


########################################################################

DEFAULT_BIB_NAME = ADS_BIB_NAME
DEFAULT_API = ADS_API_NAME
DEFAULT_ALLOWED_EXCEPTION = ADS_ALLOWED_EXCEPTION

########################################################################

def validate_api(api: str) -> None:
    apis_allowed = ['S2', 'ADS']
    if api not in apis_allowed:
        raise ValueError(f"No support for {api}. Allowed API options include {apis_allowed}")

########################################################################

    # general method for getting data independent of atlas
def call_s2_api(paper_ids, batch = False):
    '''Get papers from SemanticScholar by calling the API.

    Args:

        paper_ids (list of strs): the identifiers required for querying API

        batch (bool) whether to call the SemanticScholar API using `get_paper` or `get_papers`. The latter is faster, but usually fails, and also does not have tqdm support yet.
    '''

    sch = SemanticScholar()

    paper_ids = list( paper_ids ) # in case numpy array

    if None in paper_ids:
        # Since we should have already dropped all Nones
        raise Exception("Passed `paper_ids` contains None.")

    # NOTE: semantic scholar will truncate total number of references, citations each at 10,000 for the entire batch.
    fields = [
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
    # how external ids should be in bibtex entry

    print( f'Querying Semantic Scholar for {len(paper_ids)} papers.')

    if batch:
        # Faster to query in batch, but often fails.
        papers = sch.get_papers(
            paper_ids=paper_ids,
            fields=fields,
        )
    else:
        # NOTE: Due to the risk of requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='api.semanticscholar.org', port=443): Read timed out. (read timeout=10), two helpful steps are
        #   (1) Batch the queries into, e.g. 10 at a time, though unclear this will help once a timeout exception is thrown
        #   (2) Wrap the api call in Zach's 'keep_trying' logic.
        papers = [
            sch.get_paper(
            paper_id=paper_id, 
            fields=fields,
            ) for paper_id in tqdm(paper_ids)
        ]
    
    return papers
