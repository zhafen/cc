#include <iostream>
#include <ctime>

using namespace std; // It's not clear to me when I should/shouldn't use this for ctypes...

// Widely used struct for sparse vector
struct sparse {
	long * data;
	long * indices;
	long size;
};

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
	result = new long [n_rows];

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

sparse add_sparse( sparse a, sparse b) {

	// For now we give more than enough room for the result
	// We'll cut off the extra before returning
	// long *result;
	// result = new long [a.size + b.size];
	sparse result;
	result.size = a.size + b.size;
	result.data = new long [result.size];
	result.indices = new long [result.size];

	long ind_a, ind_b;
	long i_a = 0, i_b = 0;
	while ( i_a < a.size & i_b < b.size )
	{
		// Find out the current index
		ind_a = a.indices[i_a];
		ind_b = b.indices[i_b];


		// When they share components add to the inner product.
		// When they don't increase the location by 1.
		if ( ind_a == ind_b )
		{
			result.data[i_a] = a.data[i_a] + b.data[i_b];
			i_a++;
			i_b++;
		}
		else if ( ind_a > ind_b )
		{
			result.data[i_b] = b.data[i_b];
			i_b++;
		} else
		{
			result.data[i_a] = a.data[i_a];
			i_a++;
		}
	}
	
	return result;
}

extern "C" // required when using C++ compiler

/**
 * Calculate the number of closest publications that haven't been updated since the
 * nth update, for all n updates.
 * 
 * @sorted_history_row The update with which the publication was added, sorted for closest.
 * @n_pubs The size of sorted_history_row.
 * @max_update Maximum number of updates.
 */
int* converged_kernel_size_row( int* sorted_history_row, int n_pubs, int max_update ) {

	int* result;
	result = new int [max_update];

	int update, i, history_i;
	// Loop through updates. Each update has a kernel size of convergence,
	// i.e. how many publications out have been updated at that update or less.
	for ( update = 0; update < max_update; update++ ) {
		i = 0;
		history_i = sorted_history_row[i];
		while ( history_i <= update ) {
			i++;
			history_i = sorted_history_row[i];
		}
		result[update] = i - 1;
	}

	return result;
}

extern "C" // required when using C++ compiler

/**
 * Calculate the number of closest publications that haven't been updated since the
 * nth update, for all n updates and all publications.
 * 
 * @sorted_history The update with which the publication was added, sorted for closest.
 * @n_pubs Nmber of publications
 * @max_update Maximum number of updates.
 */
int* converged_kernel_size( int* sorted_history, int n_pubs, int max_update ) {

	int* result;
	result = new int [n_pubs*max_update];

	clock_t begin = clock();

	int update, i, j, history_i;
	// Loop through publications
	for ( j = 0; j < n_pubs; j++ ){
	
		if ( j == 9 ) {
			clock_t end = clock();
			double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			cout << "Time elapsed to calculate kernel sizes for 10 publications: " << elapsed_secs << " sec" << endl;
			double expected = elapsed_secs * n_pubs / 10;
			cout << "Expected time elapsed for all " << n_pubs << " publications: " << expected << " sec" << endl;
		}

		// Loop through updates. Each update has a kernel size of convergence,
		// i.e. how many publications out have been updated at that update or less.
		for ( update = 0; update < max_update; update++ ) {
		 	i = 0;
			history_i = sorted_history[j*n_pubs + i];
		 	while ( history_i <= update ) {

		 		i++;
				history_i = sorted_history[j*n_pubs + i];
		 	}
			result[j*max_update + update] = i - 1;
		}
	}

	return result;
}

// The stronger test framework is setup for the frontend, but a simple test framework is found below.
int main () {
	// Inner product between two sparse rows.
	int i;
	long data_a[4] = {1, 2, 3, 5};
	long indices_a[4] = { 0, 1, 4, 5};
	long data_b[5] = {-1, 2, 2, 5, 3};
	long indices_b[5] = { 0, 1, 3, 4, 17};
	sparse a, b;
	a.data = data_a;
	b.data = data_b;
	a.indices = indices_a;
	b.indices = indices_b;
	a.size = 4;
	b.size = 5;

	long expected = -1 + 2 * 2 + 5 * 3;
	cout << "Expected inner product is: "  << expected << endl;

	// Result value
	long ip = inner_product_sparse( data_a, indices_a, 4, data_b, indices_b, 5 );
	cout << "Inner product is: "  << ip << endl;

	/**
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
	delete[] result;

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
	delete[] result_all;
	*/

	/**
	// Subtraction and addition
	sparse added, added_result;
	long data_e[6] = { 0, 4, 2, 10, 3 };
	added.data = data_e;
	long indices_e[6] = { 0, 1, 3, 4, 5, 17 };
	added.indices = indices_e;
	added.size = 6;
	cout << "Expected data for addition: ";
	for ( i = 0; i < 6; i++ ){
		cout << added.data[i] << " ";
	}
	cout << endl;
	cout << "Expected indices for subtraction: ";
	for ( i = 0; i < 6; i++ ){
		cout << added.indices[i] << " ";
	}
	cout << endl;
	added_result = add_sparse( a, b );
	cout << "Actual data for addition: ";
	for ( i = 0; i < 6; i++ ){
		cout << added_result.data[i] << " ";
	}
	cout << endl;
	cout << "Expected indices for subtraction: ";
	for ( i = 0; i < 6; i++ ){
		cout << added_result.indices[i] << " ";
	}
	cout << endl;
	*/

	/**
	// Converged kernel
	int sorted_history[] = {
		0, 0, 1, 4, 1, 1, 3, 1, 2, 3,
		2, 1, 1, 2, 3, 4, 3, 0, 1, 1,
		0, 0, 1, 4, 1, 1, 3, 1, 2, 3,
		2, 1, 1, 2, 3, 4, 3, 0, 1, 1,
		0, 0, 1, 4, 1, 1, 3, 1, 2, 3,
		2, 1, 1, 2, 3, 4, 3, 0, 1, 1,
		0, 0, 1, 4, 1, 1, 3, 1, 2, 3,
		2, 1, 1, 2, 3, 4, 3, 0, 1, 1,
		0, 0, 1, 4, 1, 1, 3, 1, 2, 3,
		2, 1, 1, 2, 3, 4, 3, 0, 1, 1,
	};
	int expected_kernel[] = {		
		1, 2, 2, 2,
		-1, -1, 3, 4,
	};
	int* kernel;
	kernel = converged_kernel_size( sorted_history, 10, 4 );
	cout << "Converged kernel size row 1: ";
	for ( i = 0; i < 4 ; i++ ) {
		cout << kernel[i] << " ";
	}
	cout << endl;
	cout << "Converged kernel size row 2: ";
	for ( i = 4; i < 8 ; i++ ) {
		cout << kernel[i] << " ";
	}
	cout << endl;
	cout << "Expected kernel size row 1: ";
	for ( i = 0; i < 4 ; i++ ) {
		cout << expected_kernel[i] << " ";
	}
	cout << endl;
	cout << "Expected kernel size row 2: ";
	for ( i = 4; i < 8 ; i++ ) {
		cout << expected_kernel[i] << " ";
	}
	cout << endl;
	delete kernel;
	*/
}