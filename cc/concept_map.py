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

    def start_evaluation( self ):
        '''Yields the concepts and relations that need to be evaluated for the
        concept map to be complete.

        Returns:
            requested_concepts (list of strs):
                Concepts missing information needed to complete the map.

            requested_relations (list of strs):
                Relations missing information needed to complete the map.
        '''

        # Evaluate relations to request
        concept_products = list(
            itertools.product( self.concepts, self.concepts )
        )
        requested_relations = []
        for concept_product in concept_products:
            if concept_product[0] == concept_product[1]:
                continue
            elif concept_product[::-1] in requested_relations:
                continue
            else:
                requested_relations.append( concept_product )

        return self.concepts, requested_relations

    ########################################################################

    def finish_evaluation( self, weights, relations ):
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

        # Store weights
        self.weights = np.array([ weights[c] for c in self.concepts ])

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

    def user_evaluation( self ):

        # Get the requested values to evaluate
        req_concepts, req_relations = self.start_evaluation()

        # Weights
        print( '\nPlease provide weights for the following.' )
        weights = {}
        for concept in req_concepts:
            weights[concept] = float( input( '{} :'.format( concept ) ) )

        # Relations
        print( '\nPlease provide relations for the following.' )
        relations = {}
        for relation in req_relations:
            relations[relation] = float( input( '{} :'.format( relation ) ) )

        # Finish up
        self.finish_evaluation( weights, relations )

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
        ax = None,
        colors = None,
        axis_fontsize = 18,
        fontsize = 16,
        y_jitter = None,
        compress_concepts_horizontally = True,
    ):
        '''Make a plot of how linked a concept is with other concepts vs
        the weight of the concept (usually the importance of the concept in the
        context of the user).

        Args:
            concepts (list of strs):
                Concepts to plot.

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

        # When no axis is provided
        if ax is None:
            fig = plt.figure( figsize=(11, 10), facecolor='w' )
            ax = plt.gca()

        # Setup an automatic colorscheme
        if colors is None:
            n_colors = len( concepts )
            colorscheme_name = 'Safe_{}'.format( n_colors )
            colorscheme = getattr(
                palettable.cartocolors.qualitative,
                colorscheme_name,
            )
            colors = colorscheme.mpl_colors

        # Loop through and plot
        for i, c_x in enumerate( concepts ):

            color = colors[i]

            # X position based on weights
            x = self.weights[i]

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

            for j, c_y in enumerate( concepts ):

                # Change spaces to enters
                if compress_concepts_horizontally:
                    c_words = nltk.word_tokenize( c_y )
                    c_str = '\n'.join( c_words )

                # Skip same concept
                if i == j:
                    continue

                # Y position based on relation
                y = self.relation_matrix[i,j]

                # Induce jitter when requested
                if y_jitter is not None:
                    y += np.random.uniform( -y_jitter, y_jitter )

                # Annotate
                ax.annotate(
                    s = c_str,
                    xy = ( x, y ),
                    xycoords = 'data',
                    xytext = ( 5, -5. ),
                    textcoords = 'offset points',
                    va = 'center',
                    ha = 'left',
                    fontsize = fontsize,
                    color = colors[j],
                )

        ax.set_xlabel( 'Weight', fontsize=22 )
        ax.set_ylabel( 'Relation', fontsize=22 )

        # Tweak
        ax.set_ylim( 0., 1. )
