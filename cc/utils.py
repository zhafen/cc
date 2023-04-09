import ads
from collections import Counter 
from functools import wraps
import nltk
from nltk.metrics import edit_distance
import numpy as np
import pandas as pd
import scipy
import string
import tqdm
import warnings
## API_extension::maybe_unnecessary
## Clean up imports after.

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

import verdict

from . import config
from . import api


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

def tokenize_and_sort_text( text, tag_mapping=None, primary_alphabet_only=False ):
    '''Tokenize text into words, position tag them, and then sort
    according to tag tier.

    Args:
        text (str): Text to tokenize and sort.

        tag_mapping (dict):
            How to sort the tags.
            If None uses the tag_tier in the config.

        primary_alphabet_only (bool):
            If True, demote any words with numerals or punctuation to secondary.

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

    # Characters to remove are punctuation and numbers, except hyphens
    numpun_chars = string.punctuation.replace( '-', '' ) + '0123456789'

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
                if primary_alphabet_only:
                    for char in numpun_chars:
                        if char in word:
                            break
                    if char in word:
                        nltk2.append( word )
                    else:
                        nltk1.append( word )
                else:
                    nltk1.append( word )
            elif tag in tag_mapping[2]:
                nltk2.append( word )
            else:
                uncategorized.append( tag )
        result['primary'].append( nltk1 )
        result['secondary'].append( nltk2 )
        result['primary_stemmed'].append( stem( nltk1 ) )
    result['uncategorized'] = list( set( uncategorized ) )

    return result

########################################################################

def citation_to_api_call( citation: dict, api_name = api.DEFAULT_API ) -> tuple:
    '''Given a dictionary containing a citation return a string that, when sent to S2AG, will give a unique result.
    '''
    api.validate_api(api_name)
    if api_name == api.ADS_API_NAME:
        return citation_to_ads_call( citation)
    if api_name == api.S2_API_NAME:
        return citation_to_s2_call( citation )

########################################################################

def citation_to_s2_call( citation ):
    '''Given a dictionary containing a citation return a string that, when sent to S2AG, will give a unique result.'''
    raise NotImplementedError

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

def keep_trying( n_attempts=5, allowed_exception=api.DEFAULT_ALLOWED_EXCEPTION, verbose=True ):
    '''Sometimes we receive server errors. We don't want that to disrupt the entire
    process, so this decorator allow trying n_attempts times.

    ## API_extension::get_data_via_api
    ## This decorator is general, except for the default allowed exception.

    Args:
        n_attempts (int):
            Number of attempts before letting the exception happen.

        allowed_exception (class):
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
                except allowed_exception:
                    continue

            # On last attempt just let it be
            if verbose:
                print( 'Had to call {} {} times to get a response. Trying once more.'.format( f, n_attempts ) )
            return f( *args, **kwargs )

        return wrapped_fn

    return _keep_trying

########################################################################

def api_query(*args, api_name = api.DEFAULT_API, **kwargs ) -> list:
    '''Convenience wrapper for searching an API.'''
    api.validate_api(api_name)
    if api_name == api.ADS_API_NAME:
        return ads_query( *args, **kwargs )
    if api_name == api.S2_API_NAME:
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

def random_publications(*args, api_name = api.DEFAULT_API, **kwargs,):
    '''Choose random publications by choosing a random date and then choosing a random publication announced on that date, via some API.'''

    api.validate_api(api_name)
    if api_name == api.ADS_API_NAME:
        return random_publications_ads(*args, **kwargs)
    
    elif api_name == api.S2_API_NAME:
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
    pbar = tqdm.tqdm( total=n_sample, position=0, leave=True )
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
                tqdm.tqdm.write( 'Publication {} is not the right arxiv category. Continuing.'.format( p.bibcode ) )
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


########################################################################

def plot_voronoi(
    points,
    labels = None,
    plot_cells = True,
    colors = None,
    color_default = 'none',
    edgecolors = None,
    edgecolor_default = 'k',
    cmap = 'cubehelix',
    norm = None,
    hatching = None,
    plot_label_box = False,
    ax = None,
    offset_magnitude = 5,
    qhull_options = 'Qbb Qc Qz',
    xlim = None,
    ylim = None,
    cell_kwargs = {},
    **annotate_kwargs
):

    # Convert to colors arrays
    if norm is None:
        norm = matplotlib.colors.Normalize()
    if isinstance( cmap, str ):
        cmap = matplotlib.cm.get_cmap( cmap )

    # Duplicate coordinates are not handled well
    points, unique_inds = np.unique( points, axis=0, return_index=True )
    if labels is not None:
        labels = np.array( labels )[unique_inds]
    if colors is not None:
        colors = colors[unique_inds]
        colors = cmap( norm( colors ) )
    if edgecolors is not None:
        edgecolors = edgecolors[unique_inds]
        edgecolors = cmap( norm( edgecolors ) )
    if hatching is not None:
        hatching = hatching[unique_inds]
        if ( edgecolor_default == 'none' ) or ( edgecolors is not None ):
            warnings.warn( 'Hatchcolor and edgecolor are the same in matplotlib.' )
    
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
    if xlim is None:
        xmin = points[:,0].min()
        xmax = points[:,0].max()
        xwidth = xmax - xmin
        xlim = [ xmin - 0.1 * xwidth, xmax + 0.1 * xwidth ]
    if ylim is None:
        ymin = points[:,1].min()
        ymax = points[:,1].max()
        ywidth = ymax - ymin
        ylim = [ ymin - 0.1 * ywidth, ymax + 0.1 * ywidth ]
        
    ax.set_xlim( xlim )
    ax.set_ylim( ylim )

    vor = scipy.spatial.Voronoi( points, qhull_options=qhull_options )
    
    ptp_bound = vor.points.ptp( axis=0 )
    center = vor.points.mean( axis=0 )

    for i, point in enumerate( tqdm.tqdm( points ) ):
        
        # Get data for this point
        i_region = vor.point_region[i]
        region = np.array( vor.regions[i_region] )
        is_neg = region == -1
        is_on_edge = is_neg.sum() > 0
        region = region[np.invert(is_neg)]
        vertices = vor.vertices[region]
        
        # Add additional points to the vertices for the regions
        # that are on the edge. This is taken from scipy's source code for the most part.
        if is_on_edge:
            add_vertices = []
            for j, pointidx in enumerate( vor.ridge_points ):
                simplex = np.array( vor.ridge_vertices[j] )
                if ( i in pointidx ) and ( -1 in vor.ridge_vertices[j] ):

                    ii = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    if (vor.furthest_site):
                        direction = -direction
                    far_point = vor.vertices[ii] + direction * ptp_bound.max()

                    add_vertices.append( far_point )
            # If found a vertex, add it on
            if len( add_vertices ) > 0:
                vertices = np.concatenate( [ vertices, add_vertices ], axis=0 )
            else:
                import pdb; pdb.set_trace()
            
        # Construct a shapely polygon for the region
        region_polygon = Polygon( vertices ).convex_hull
        
        # Plot the cell
        if plot_cells:
            used_cell_kwargs = {}
            used_cell_kwargs.update( cell_kwargs )
            if colors is not None:
                facecolor = colors[i]
            else:
                facecolor = color_default
            if edgecolors is not None:
                edgecolor = edgecolors[i]
            else:
                edgecolor = edgecolor_default
            if hatching is not None:
                used_cell_kwargs['hatch'] = hatching[i]
            patch = PolygonPatch(
                region_polygon,
                facecolor = facecolor,
                edgecolor = edgecolor,
                **used_cell_kwargs
            )
            ax.add_patch( patch )
            
        # Add a label, trying a few orientations
        if labels is not None:
            has = [ 'left', 'center', 'right' ]
            vas = [ 'bottom', 'center', 'top' ]
            offsets = offset_magnitude * np.array([ 1, 0, -1 ])
            break_out = False
            for iii in [ 0, 1, 2 ]:
                for jjj in [ 0, 1, 2 ]:
                    used_kwargs = dict(
                        xycoords = 'data',
                        xytext = ( offsets[iii], offsets[jjj] ),
                        textcoords = 'offset points',
                        ha = has[iii],
                        va = vas[jjj],
                    )
                    used_kwargs.update( annotate_kwargs )
                    text = ax.annotate(
                        text = labels[i],
                        xy = point,
                        **used_kwargs
                    )
                    text.set_path_effects([
                        path_effects.Stroke( linewidth=text.get_fontsize() / 5., foreground='w' ),
                        path_effects.Normal()
                    ])

                    # Create a polygon for the label
                    bbox_text = text.get_window_extent( ax.figure.canvas.get_renderer() )
                    display_to_data = ax.transData.inverted()
                    text_data_corners = display_to_data.transform( bbox_text.corners() )
                    text_data_corners = text_data_corners[[0,1,3,2],:] # Reformat
                    text_polygon = Polygon( text_data_corners )
                    
                    text.set_visible( False )
                    
                    # We'll never fit it in if it's just too large
                    if text_polygon.area > region_polygon.area:
                        break_out = True
                        break
                        
                    # If it doesn't fit in the region try again
                    if not region_polygon.contains( text_polygon ):
                        continue
                        
                    # If it doesn't fit in the bounds try again
                    if text_polygon.bounds[0] < xlim[0]:
                        continue
                    if text_polygon.bounds[1] < ylim[0]:
                        continue
                    if text_polygon.bounds[2] > xlim[1]:
                        continue
                    if text_polygon.bounds[3] > ylim[1]:
                        continue

                    # If we find a good option stop iterating
                    text.set_visible( True )
                    if plot_label_box:
                        patch = PolygonPatch(
                            text_polygon,
                            facecolor = 'none',
                            edgecolor = 'k',
                        )
                        ax.add_patch( patch )
                    break_out = True
                    break
                if break_out:
                    break

    return ax, vor
