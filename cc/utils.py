from collections import Counter 
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

from . import config # literature-topography likes this
# import config # tests like this


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
