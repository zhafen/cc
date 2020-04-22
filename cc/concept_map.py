import copy
import itertools
import nltk
import numpy as np
import palettable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms

import augment
import verdict

########################################################################

class ConceptMap( object ):

    @augment.store_parameters
    def __init__( self, concepts ):

        pass

    ########################################################################
    # Map Construction
    ########################################################################

    def start_evaluation( self, concepts=None ):
        '''Yields the concepts and relations that need to be evaluated for the
        concept map to be complete.

        Returns:
            requested_concepts (list of strs):
                Concepts missing information needed to complete the map.

            requested_relations (list of strs):
                Relations missing information needed to complete the map.
        '''

        if concepts is None:
            concepts = self.concepts

        # Don't request concepts for which data exists
        if hasattr( self, 'weights' ):
            requested_concepts = list(
                set( concepts ) - set( self.concepts )
            )
        else:
            requested_concepts = concepts

        # Evaluate relations to request
        concept_products = list(
            itertools.product( requested_concepts, self.concepts )
        )
        concept_products += list(
            itertools.product( requested_concepts, requested_concepts )
        )

        # Remove extras
        concept_products = list( set( concept_products ) )
        requested_relations = []
        for concept_product in concept_products:
            if concept_product[0] == concept_product[1]:
                continue
            elif concept_product[::-1] in requested_relations:
                continue
            else:
                requested_relations.append( concept_product )

        return requested_concepts, requested_relations

    ########################################################################

    def finish_evaluation( self, weights, relations, concepts=None, ):
        '''Store the evaluated concept map in a useful format.

        Args:
            weights (dict of floats):
                Weights of individual concepts.

            relations (dict of floats):
                Relations between concepts. A value of 1.0 means the concepts
                are identical. A value of 0.0 means the concepts are completely
                independent.

        Modifies:
            self.weights (dict of floats):
                Weights of individual concepts.

            self.relation_matrix (np.ndarray, (n_concepts,n_concepts)):
                Matrix expressing the relationship between concepts.
        '''

        # Update weight dictionary using existing data
        if hasattr( self, 'weights' ):
            for i, c in enumerate( self.concepts ):
                # Don't overwrite new weights
                if c in weights:
                    continue
                weights[c] = self.weights[i]

        # Update relation dictionary using existing data
        if hasattr( self, 'relation_matrix' ):
            for i, c_i in enumerate( self.concepts ):
                for j, c_j in enumerate( self.concepts ):
                    # Don't overwrite new relations
                    if ( c_i, c_j ) in relations or ( c_j, c_i ) in relations:
                        continue
                    relations[( c_i, c_j )] = self.relation_matrix[i,j]
        
        # Store concepts and weights
        if concepts is not None:
            self.concepts = concepts
        self.weights = np.array([ weights[c] for c in self.concepts ])

        # Store relations
        n = len( self.concepts )
        self.relation_matrix = np.full( ( n, n ), -1.0 )
        for i, c_i in enumerate( self.concepts ):
            for j, c_j in enumerate( self.concepts ):

                # Diagonals
                if i == j:
                    self.relation_matrix[i,j] = 1.
                    continue

                # Retrieve data
                try:
                    value = relations[(c_i,c_j)]
                except KeyError:
                    value = relations[(c_j,c_i)]

                self.relation_matrix[i,j] = value

    ########################################################################

    def user_evaluation( self, requested_concepts=None ):

        # Get the requested values to evaluate
        unk_concepts, unk_relations = self.start_evaluation( requested_concepts )

        # Weights
        print( '\nPlease provide weights for the following.' )
        weights = {}
        for concept in unk_concepts:
            weights[concept] = float( input( '{} :'.format( concept ) ) )

        # Relations
        print( '\nPlease provide relations for the following.' )
        relations = {}
        for relation in unk_relations:
            relations[relation] = float( input( '{} :'.format( relation ) ) )

        # Finish up
        self.finish_evaluation( weights, relations, requested_concepts, )

    ########################################################################
    # Map Functions
    ########################################################################

    def find_most_related( self, concept, n=10 ):
        '''Find the concepts most related to a given concept.

        Args:
            concept (str):
                The concept to relate to.

            n (int):
                Number of concepts to return, in order.

        Returns:
            related_concepts (np.ndarray of strs):
                The n most related concepts.
        '''

        # Find the corresponding indice for a concept
        inds = np.arange( len( self.concepts ) )
        ind = inds[np.array(self.concepts)==concept][0]

        # Sort and return
        sort_inds = np.argsort( self.relation_matrix[ind,:] )
        concept_inds = sort_inds[-n:]
        related_concepts = np.array( self.concepts )[concept_inds]

        return related_concepts

    ########################################################################
    # Data Management
    ########################################################################

    def save( self, filepath ):
        '''Save the concept map to a .hdf5 file.

        Args:
            filepath (str): Location to save the file.
        '''

        # Prep
        data = verdict.Dict( {} )
        for attr in [ 'concepts', 'weights', 'relation_matrix' ]:
            data[attr] = getattr( self, attr )

        # Save
        data.to_hdf5( filepath )

    ########################################################################

    @classmethod
    def load( cls, filepath ):
        '''Load a concept map from a .hdf5 file.

        Args:
            filepath (str): Where to load the file from.
        '''

        data = verdict.Dict.from_hdf5( filepath )

        result = ConceptMap( list( data['concepts'] ) )
        result.weights = data['weights']
        result.relation_matrix = data['relation_matrix']

        return result

    ########################################################################
    # Map Plotting
    ########################################################################

    def plot_relation_vs_weight(
        self,
        concepts = None,
        n_concepts = 10,
        y_concepts = 'most related',
        n_y_concepts = None,
        x_axis = 'weight sorted',
        ax = None,
        colors = None,
        axis_fontsize = 20,
        fontsize = 18,
        y_jitter = None,
        compress_concepts_horizontally = True,
    ):
        '''Make a plot of how linked a concept is with other concepts vs
        the weight of the concept (usually the importance of the concept in the
        context of the user).

        Args:
            concepts (None, list of strs, or str):
                Options...
                    None: All concepts.
                    list of strs: The provided concepts.
                    'most weighted': The n_concepts with highest weights.
                    a concept: The n_concepts most related to the concept.

            n_concepts (int):
                Number of x-axis concepts.

            y_concepts (str):
                Options...
                    'most related': The n_y_concepts most related to the
                        x-axis concept.
                    'x concepts': The concepts on the x-axis.

            n_concepts (int):
                Number of y-axis concepts.

            x_axis (str):
                What to plot on the x-axis. Options...
                    'weight sorted': Bar chart sorted by weight.
                    'weights': Each line is at the location of the weight.

            ax (axis object):
                Matplotlib axis to plot on, if provided.

            colors (list):
                List of colors to use.

            axis_fontsize (float):
                Fontsize for x-axis concept labels.

            fontsize (float):
                Fontsize for concept labels acting as points.

            y_jitter (float):
                If not None, randomly shift the y value by a value within
                +- y_jitter

            compress_concepts_horizontally (bool):
                If True, new words within a concept continue on the next line.
        '''

        # Default to all concepts
        if concepts is None:
            concepts = self.concepts
        elif isinstance( concepts, str ):
            if concepts == 'most weighted':
                sort_inds = np.argsort( self.weights )
                concept_inds = sort_inds[-n_concepts:]
                concepts = np.array( self.concepts )[concept_inds]
            else:
                concepts = self.find_most_related( concepts, n_concepts )

        # Number of y concepts to plot
        if y_concepts == 'most related':
            if n_y_concepts is None:
                n_y_concepts = len( concepts )

        # When no axis is provided
        if ax is None:
            fig = plt.figure( figsize=(15, 5), facecolor='w' )
            ax = plt.gca()

        # Setup an automatic colorscheme
        if colors is None:
            n_colors = len( concepts )
            colorscheme_name = 'Safe_{}'.format( n_colors )
            colorscheme = getattr(
                palettable.cartocolors.qualitative,
                colorscheme_name,
            )
            colors = {}
            for i, c in enumerate( concepts ):
                colors[c] = colorscheme.mpl_colors[i]

        weights = []
        for i, c in enumerate( self.concepts ):
            if c in concepts:
                weights.append( self.weights[i] )
        weights = np.array( weights )
        # With constant spacing
        if x_axis == 'weight sorted':
            # Sort concepts
            sorted_concepts = [
                c for _, c in 
                sorted(zip(weights, concepts))
            ]
            # Finish up with x values
            n_x = len( weights )
            values = np.arange( n_x )[::-1]
            xs = {}
            for i, c in enumerate( sorted_concepts ):
                xs[c] = values[i]
        elif x_axis == 'weights':
            xs = {}
            for i, c in concepts:
                xs[c] = weights[i]

        # Loop through and plot
        for i, c_x in enumerate( self.concepts ):

            # Skip not plotted concepts
            if c_x not in concepts:
                continue

            color = colors[c_x]

            # X position based on weights
            x = xs[c_x]

            # Change spaces to enters
            if compress_concepts_horizontally:
                c_words = nltk.word_tokenize( c_x )
                c_str = '\n'.join( c_words )

            # Axis label
            ax.annotate(
                s = c_str,
                xy = ( x, 1.0 ),
                xycoords = matplotlib.transforms.blended_transform_factory(
                    ax.transData,
                    ax.transAxes,
                ),
                xytext = ( 0., 5. ),
                textcoords = 'offset points',
                va = 'bottom',
                ha = 'center',
                fontsize = axis_fontsize,
                color = color,
            )

            # Add a line to make things more visible
            ax.axvline(
                x,
                color = color,
                linewidth = 3,
            )

            # Get y concepts
            if y_concepts == 'most related':
                used_y_concepts = self.find_most_related( c_x, n_y_concepts )
            elif y_concepts == 'x concepts':
                used_y_concepts = concepts

            # Loop through and plot
            annots = []
            c_ys = []
            ys = []
            for j, c_y in enumerate( self.concepts ):

                # Skip not plotted concepts
                if c_y not in used_y_concepts:
                    continue

                # Skip same concept
                if i == j:
                    continue

                # Change spaces to enters
                if compress_concepts_horizontally:
                    c_words = nltk.word_tokenize( c_y )
                    c_str = '\n'.join( c_words )
                else:
                    c_str = c_y

                # Get y value
                y = self.relation_matrix[i,j]

                # Induce jitter when requested
                if y_jitter is not None:
                    y += np.random.uniform( -y_jitter, y_jitter )

                # Annotate
                annot = ax.annotate(
                    s = c_str,
                    xy = ( x, y ),
                    xycoords = 'data',
                    va = 'center',
                    ha = 'left',
                    fontsize = fontsize,
                    color = 'grey',
                )

                # Draw so we can modify
                annot.draw(ax.figure.canvas.get_renderer())

                annots.append( annot )
                c_ys.append( c_y )
                ys.append( y )

            def shift_for_overlap( anns ):

                # Loop back through and avoid overlap. 
                modified = []
                for ii, annot_i in enumerate( anns ):

                    # Avoid modifying twice
                    if ii in modified:
                        continue

                    # Draw so we can modify
                    bbox_i = annot_i.get_window_extent()

                    for jj, annot_j in enumerate( anns ):

                        # Don't move if same one
                        if ii == jj:
                            continue

                        if jj in modified:
                            continue

                        bbox_j = annot_j.get_window_extent()

                        # Find overlaps
                        overlap_a = bbox_i.y0 - bbox_j.y1
                        overlap_b = bbox_j.y0 - bbox_i.y1
                        if ( overlap_a < 0 ) and ( overlap_b < 0 ):

                            # Get the appropriate transform
                            t = matplotlib.transforms.offset_copy(
                                annot_j.get_transform(),
                                y = overlap_a*1.01,
                                units = 'dots',
                            )

                            # Apply
                            annot_j.set_transform( t )
                            modified.append( jj )

                return anns, len( modified )

            # Loop until all overlaps are gone
            n_mod = len( annots )
            while n_mod > 0:
                annots, n_mod = shift_for_overlap( annots )

            # Loop through and actually draw
            for j, c_y in enumerate( c_ys ):

                annot_j = annots[j]

                # Change spaces to enters
                if compress_concepts_horizontally:
                    c_words = nltk.word_tokenize( c_y )
                    c_str = '\n'.join( c_words )
                else:
                    c_str = c_y

                try:
                    color = colors[c_y]
                except KeyError:
                    color = '0.3'

                annot = ax.annotate(
                    c_str,
                    xy = ( x, ys[j] ),
                    xycoords = annot_j.get_transform(),
                    xytext = ( 5, 0 ),
                    textcoords = 'offset points',
                    fontsize = fontsize,
                    color = color,
                    va = 'center',
                    ha = 'left',
                )

                annot_j.remove()

        # Add bars indicating importance
        x_bar = [ xs[c] for c in concepts ]
        color_bar = [ colors[c] for c in concepts ]
        heights = weights * 0.2 / weights.max()
        ax.bar(
            x_bar,
            heights,
            width = 0.4,
            align = 'edge',
            color = color_bar,
        )

        # Labels
        x_label = {
            'weight sorted': 'Importance',
            'weights': 'Weight',
        }
        ax.set_xlabel( x_label[x_axis], fontsize=22 )
        ax.set_ylabel( 'Relation', fontsize=22 )

        # Limits
        ax.set_ylim( 0., 1. )
        x_vals = list( xs.values() )
        ax.set_xlim( min( x_vals ) - 0.5, max( x_vals ) + 1 )

        # Edge tweaks
        if x_axis == 'weight sorted':
            ax.tick_params( bottom=False, labelbottom=False )

        return annot_j
