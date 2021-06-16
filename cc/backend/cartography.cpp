#include <iostream>

using namespace std; // It's not clear to me when I should/shouldn't use this for ctypes...

extern "C" // required when using C++ compiler

/**
 * The basic inner product of two arrays.
 * 
 * @arr_a First array to take the inner product of.
 * @arr_b Second array to take the inner product of.
 * @size Size of the arrays.
 */
long inner_product(long arr_a[], long arr_b[], long size) {

	long i, ip = 0;
	for ( i = 0; i < size - 1; ++i ) {
		ip += arr_a[i] * arr_b[i];
	}

	return ip;
}

extern "C" // required when using C++ compiler

/**
 * The basic inner product of two arrays, but for compressed format.
 * 
 * @data_a The values of the first array.
 * @indices_a The indices the values correspond to.
 * @size_a The size of the first array.
 * @data_b The values of the second array.
 * @indices_b The indices the values correspond to.
 * @size_b The size of the second array.
 */
long inner_product_sparse( long data_a[], long indices_a[], long size_a, long data_b[], long indices_b[], long size_b ) {

	long ind_a, ind_b;
	long ip = 0, i_a = 0, i_b = 0;
	while ( i_a < size_a & i_b < size_b )
	{
		// Find out the current index
		ind_a = indices_a[i_a];
		ind_b = indices_b[i_b];

		// When they share components add to the inner product.
		// When they don't increase the location by 1.
		if ( ind_a == ind_b )
		{
			ip += data_a[i_a] * data_b[i_b];
			i_a++;
			i_b++;
		}
		else if ( ind_a > ind_b )
		{
			i_b++;
		} else
		{
			i_a++;
		}
	}
	
	return ip;
}

extern "C" // required when using C++ compiler

/**
 * Inner product of all rows in a sparse matrix with a single row from that matrix.
 * Output is an array the size of the number of rows.
 * The sparse matrix is assumed to be in compressed sparse row format.
 * 
 * @i Index of the target row.
 * @data The data values for the sparse matrix.
 * @indices The indices for each data value.
 * @data_size The total number of data values/indices.
 * @indptr Where indices for one row ends and another begins.
 * @n_rows The number of rows.
 * @result For storing the output.
 */
long* inner_product_row_all_sparse( long i, long data[], long indices[], long indptr[], long n_rows ) {

	// Create result array
	long* result;
	result = new long[n_rows];

	// Get starting ind
	long i_a = indptr[i];
	long size_a = indptr[i+1] - i_a;

	// Loop over all rows
	long i_b, size_b;
	long j = 0;
	for ( j = 0; j < n_rows; j++ ){

		// Other ind
		i_b = indptr[j];
		size_b = indptr[j+1] - i_b;
		
		// Calculate IP
		result[j] = inner_product_sparse( &data[i_a], &indices[i_a], size_a, &data[i_b], &indices[i_b], size_b );
	}

	return result;
}

extern "C" // required when using C++ compiler

/**
 * Pairwise inner product of all rows in a sparse matrix.
 * Output is a ( n_rows, n_rows ) array.
 * The sparse matrix is assumed to be in compressed sparse row format.
 * 
 * @data The data values for the sparse matrix.
 * @indices The indices for each data value.
 * @data_size The total number of data values/indices.
 * @indptr Where indices for one row ends and another begins.
 * @n_rows The number of rows.
 */
long* inner_product_matrix( long data[], long indices[], long indptr[], long n_rows ) {

	// Create result array. It's a flattened 2D matrix.
	long* result;
	result = new long [n_rows*n_rows];

	long i, j, i_a, i_b, size_a, size_b;
	for ( i = 0; i < n_rows; i++ ){

		// Get ind
		i_a = indptr[i];
		size_a = indptr[i+1] - i_a;

		for ( j = i; j < n_rows ; j++ ) {

			// Other ind
			i_b = indptr[j];
			size_b = indptr[j+1] - i_b;
		
			result[i*n_rows + j] = inner_product_sparse( &data[i_a], &indices[i_a], size_a, &data[i_b], &indices[i_b], size_b );
		}
	}

	// Reflect across the diagonal
	for ( i = 0; i < n_rows; i++ ){
		for (j = 0; j < i ; j++ ){
			result[i*n_rows + j] = result[j*n_rows + i];
		}
	}

	return result;
}

extern "C" // required when using C++ compiler

/**
 * Calculate the number of closest publications that haven't been updated since the
 * nth update, for all n updates.
 * 
 * @sorted_history The update with which the publication was added, sorted for closest.
 * @size The size of sorted_history (n_pubs).
 */
int* converged_kernel_size_row( int* sorted_history, int size, int max_update ) {

	int* result;
	result = new int [max_update];

	int update, i;
	// Loop through updates. Each update has a kernel size of convergence,
	// i.e. how many publications out have been updated at that update or less.
	for ( update = 0; update <= max_update; update++ ) {
		i = 0;
		while ( sorted_history[i] <= update ) {
			i++;
			if ( i >= size ){
				break;
			}
		}
		result[update] = i - 1;
	}

	return result;
}

int* converged_kernel_size( ) {

	int* result;
	/**
            # Loop over all publications
            full_result = []
            full_cospsi_result = []
            for pub in tqdm( publications ):

                cospsi = self.cospsi( pub, 'all' )
                sort_inds = np.argsort( cospsi )[::-1]
                sorted_cospsi = cospsi[sort_inds]
                sorted_history = self.update_history[sort_inds]

                result = []
                cospsi_result = []
                max_rank =  self.update_history.max() 
                for rank in range( max_rank ):

                    result_i = np.argmin( sorted_history <= rank ) - 1
                    result.append( result_i )
                    cospsi_result.append( sorted_cospsi[result_i] )

                full_result.append( result )
                full_cospsi_result.append( cospsi_result )

            if len( full_result ) == 1:
                return full_result[0], full_cospsi_result[0]

            return np.array( full_result ), np.array( full_cospsi_result )
	*/

	return result;
}

// The stronger test framework is setup for the frontend, but a simple test framework is found below.
int main () {
	// Inner product between two sparse rows.
	long data_a[4] = {1, 2, 3, 5};
	long indices_a[4] = { 0, 1, 4, 5};
	long data_b[5] = {-1, 2, 2, 5, 3};
	long indices_b[5] = { 0, 1, 3, 4, 17};

	long expected = -1 + 2 * 2 + 5 * 3;
	cout << "Expected is: "  << expected << endl;

	// Result value
	long ip = inner_product_sparse( data_a, indices_a, 4, data_b, indices_b, 5 );
	cout << "Inner product is: "  << ip << endl;

	// Inner product for a component matrix
	long data[9] = { 1, 2, 3, 5, -1, 2, 2, 5, 3 };
	long indices[9] = { 0, 1, 4, 5, 0, 1, 3, 4, 17 };
	long indptr[3] = { 0, 4, 9};

	long expected_norm = 1 + 2 * 2 + 3 * 3 + 5 * 5;
	cout << "Inner product row-all expected is: "  << expected_norm << " " << expected << endl;

	// Result for one row
	long* result;
	result = inner_product_row_all_sparse( 0, data, indices, indptr, 2 );
	cout << "Inner product row-all is: "  << result[0] << " " << result[1] << endl;

	long expected_all[2][2] = {
		{ expected_norm, expected, },
		{ 0, 1 * 1 + 2 * 2 + 2 * 2 + 5 * 5 + 3 * 3 },
	};
	cout << "Inner product matrix expected is: " << endl;
	cout << expected_all[0][0] << " " << expected_all[0][1] << endl;
	cout << expected_all[1][0] << "  " << expected_all[1][1] << endl;

	// Result for all
	long* result_all;
	result_all = inner_product_matrix( data, indices, indptr, 2 );
	cout << "Inner product matrix is: " << result_all << endl;
	cout << result_all[0] << "  " << result_all[1] << endl;
	cout << result_all[2] << "  " << result_all[3] << endl;
}