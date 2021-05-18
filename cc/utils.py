import ads
from collections import Counter 
import nltk
from nltk.metrics import edit_distance
import numpy as np
import pandas as pd
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import verdict

from . import config

########################################################################

def uniquify_words( a, **kwargs ):
    '''Find the unique words in a list
    This involves...
    1. Stemming the words in the lists
    2. Matching words with a sufficiently low edit distance
       (accounts for mispellings)

    Args:
        a (list of strs):
            The list to uniquify

    Kwargs:
        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two words for them
            to count as the same word.

        min_len_ed (int):
            Words below this length will not be considered the same even if
            they are within the given edit distance.

        stemmed (bool):
            If True the words are already stemmed.
    '''

    return match_words( a, a, **kwargs )

########################################################################

def match_words( a, b, max_edit_distance=2, min_len_ed=5, stemmed=False ):
    '''Find the matching words in two lists.
    This involves...
    1. Stemming the words in the lists
    2. Matching words with a sufficiently low edit distance
       (accounts for mispellings)

    Args:
        a (list of strs):
            The first to look for matching words in.

        b (list of strs):
            The second to look for matching words in.

        max_edit_distance (int):
            Maximum Levenshtein edit-distance between two concepts for them
            to count as the same concept.

        min_len_ed (int):
            Words below this length will not be considered the same even if
            they are within the given edit distance.

        stemmed (bool):
            If True the words are already stemmed.

    Returns:
        result (set):
            Set of matching words between the two lists.
    '''

    # Short-cut
    if ( len( a ) == 0 ) or ( len( b ) == 0 ):
        return set( [] )

    # Stem the lists first
    if not stemmed:
        sa, sb = stem( a ), stem( b )
    else:
        sa, sb = a, b

    # When not including edit distance
    if max_edit_distance is None:
        result = []
        for c in sa:
            if c in sb:
                result.append( c )
        result = set( result )
        return result

    # The words in at least one list (list b) have to be sufficiently long
    # to avoid warping one short word into another.
    word_lens = np.array([ len( _ ) for _ in sb ])
    satisfies_min_len_ed = word_lens >= min_len_ed

    # Look for concepts with a sufficiently low Levenshtein edit-distance
    result = []
    for c in sa:

        # Find matches
        edit_distances = np.array([
            edit_distance( c, c_check, )
            for c_check
            in sb
        ])

        # Matches if below minimum edit distance and not too short, or
        # if no edits are required
        is_matching = (
            ( edit_distances <= max_edit_distance ) &
            satisfies_min_len_ed
        )
        is_matching = np.logical_or( is_matching, edit_distances == 0 )
        matches = sb[is_matching]

        # Skip when there's no matches
        if len( matches ) == 0:
            continue

        # Count the matches and represent by maximum count
        count = Counter( matches )
        true_c = verdict.Dict( count ).keymax()[0]
        
        # Store
        result.append( true_c )

    result = set( result )

    return result

########################################################################

def stem( l, unique=True ):
    '''Stem the words in a list of words.
    
    Args:
        l (list of strs):
            The words to stem.

        unique (bool):
            If True return the unique, sorted stemmed words.

    Returns:
        sl (list of strs):
            List of stemmed words.
    '''
    s = nltk.stem.SnowballStemmer( 'english' )
    sl = []
    for c in l:
        words = nltk.word_tokenize( c )
        stemmed_words = [ s.stem( w ) for w in words ]
        sl.append( ' '.join( stemmed_words ) )
    if unique:
        sl = np.array( list( set( sl ) ) )
    return sl

########################################################################

def tokenize_and_sort_text( text, tag_mapping=None ):
    '''Tokenize text into words, position tag them, and then sort
    according to tag tier.

    Args:
        text (str): Text to tokenize and sort.

        tag_mapping (dict):
            How to sort the tags.
            If None uses the tag_tier in the config.

    Returns:
        result (dict)
    '''

    result = {}

    # Parse using NLTK
    sentences = nltk.sent_tokenize( text )
    sentences = [nltk.word_tokenize(sent) for sent in sentences] 
    result['all'] = [
        nltk.pos_tag(sent) for sent in sentences
    ] 

    if tag_mapping is None:
        tag_mapping = config.nltk['tag_tier']

    # Classify into primary and secondary tiers, i.e. effectively
    # nouns, verbs, and adjectives vs everything else.
    result['primary'] = []
    result['secondary'] = []
    result['primary_stemmed'] = []
    uncategorized = []
    for sent in result['all']:
        nltk1 = []
        nltk2 = []
        for word, tag in sent:
            if tag in tag_mapping[1]:
                nltk1.append( word )
            elif tag in tag_mapping[2]:
                nltk2.append( word )
            else:
                uncategorized.append( tag )
        result['primary'].append( nltk1 )
        result['secondary'].append( nltk2 )
        result['primary_stemmed'].append( stem( nltk1 ) )
    result['uncategorized'] = set( uncategorized )

    return result

########################################################################

def citation_to_ads_call( citation ):
    '''Given a dictionary containing a citation return a string that,
    when sent to ADS, will give a unique result.

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
    if 'eprint' in citation:
        # When we can, check that the eprint is of the correct type
        if 'eprinttype' in citation:
            assert citation['eprinttype'] == 'arxiv', \
                'eprinttype={} eprint={}'.format(
                citation['eprinttype'],
                citation['eprint']
            )

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

def ads_query(
    q,
    fl = ['abstract', 'citation', 'reference', 'entry_date', 'identifier' ],
    rows = 50
):
    '''Convenience wrapper for searching ADS.

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

def random_publications(
    n_sample,
    start_time,
    end_time,
    fl = [ 'arxivid', 'doi', 'date', 'citation', 'reference', 'abstract', 'bibcode', 'entry_date' ],
    arxiv_class = None,
    seed = None,
    max_loops = None,
    bad_days_of_week = [ 'Saturday', 'Sunday' ],
):
    '''Choose random publications by choosing a random date and then choosing
    a random publication announced on that date.
    Note that while this means that publications announced on the same date
    as many other publications are less likely to be selected, this is not
    expected to typically be an important effect.

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
            Number of iterations before breaking. Defaults to 10 * n_sample.

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
        max_loops = 10 * n_sample

    search_str = ''

    if arxiv_class is not None:
        search_str += 'arxiv_class:"{}"'.format( arxiv_class )
        if arxiv_class == 'astro-ph':
            search_str = 'arxiv_class:"astro-ph"'
            subcats = [ 'GA', 'CO', 'EP', 'HE', 'IM', 'SR' ]
            for subcat in subcats:
                search_str += ' OR arxiv_class:"astro-ph.{}"'.format( subcat )

    pubs = []
    n_loops = 0
    pbar = tqdm.tqdm( total=n_sample )
    bad_dates = []
    empty_dates = []
    empty_abstracts = []
    no_refs_or_cits = []
    while len( pubs ) < n_sample:


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

        # Build query
        query_dict = dict(
            fl = fl,
        )

        if search_str == '':
            query_dict['entdate'] = random_date
            ads_query = ads.SearchQuery( **query_dict )
        else:
            random_datetime_end = random_datetime + pd.DateOffset( days=1 )
            random_date_end = '{}-{}-{}'.format(
                random_datetime_end.year,
                random_datetime_end.month,
                random_datetime_end.day
            )
            query_dict['q'] = search_str + ' entdate:[{} TO {}]'.format( random_date, random_date_end )
            ads_query = ads.SearchQuery( query_dict = query_dict )

        query_list = list( ads_query )
        if len( query_list ) == 0:
            empty_dates.append( random_datetime )
            continue

        p = np.random.choice( query_list )
        
        # Cannot do this for publications missing abstract data.
        if p.abstract is None:
            empty_abstracts.append( p )
            tqdm.tqdm.write( 'Publication {} has no abstract. Continuing.'.format( p.bibcode ) )
            continue
        
        if p.citation is None and p.reference is None:
            no_refs_or_cits.append( p )
            tqdm.tqdm.write( 'Publication {} has no references or citations. Continuing.'.format( p.bibcode ) )
            continue
        
        pubs.append( p )
        pbar.update( 1 )
    pbar.close()

    print( 'Retrieved {} random publications. Took {} tries'.format( len( pubs ), n_loops ) )

    return pubs

########################################################################

def multicolor_text(
    x,
    y,
    strings,
    colors,
    fontweights = None,
    ax = None,
    fontsize = 24,
    annotated = False,
    annote_fontsize = 12,
    spacing = 1.1,
    **kwargs
):

    if ax is None:
        figure = plt.figure( figsize=(16,0.1), facecolor='w' )
        ax = plt.gca()

    n_lines = 0
    t = ax.transData
    for i, s in enumerate( strings ):

        c = colors[i]

        if fontweights is not None:
            fontweight = fontweights[i]
        else:
            fontweight = None

        text = ax.text(
            x,
            y,
            r'' + s + " ",
            color = c,
            transform = t,
            fontweight = fontweight,
            fontsize = fontsize,
            **kwargs
        )

        # Need to draw to update the text position.
        text.draw( ax.figure.canvas.get_renderer() )
        ex = text.get_window_extent()

        if annotated:
            annot_t = transforms.offset_copy(
                t,
                y = ex.height,
                units = 'dots',
            )
            ax.text(
                x,
                y,
                str( i ),
                transform = annot_t,
                color = '0.7',
                fontsize = annote_fontsize,
                **kwargs
            )

        # Wrap
        out_of_bounds = ex.transformed( ax.transAxes.inverted() ).x1 > 1.
        if not out_of_bounds:

            # Normal offset
            t = transforms.offset_copy(
                t,
                x = ex.width,
                units = 'dots',
            )

        else:

            # Wrap reset
            n_lines += 1
            t = ax.transData
            t = transforms.offset_copy(
                t,
                y = -n_lines * ex.height * ( 1. + annotated ) * spacing,
                units = 'dots',
            )

