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
int inner_product(int arr_a[], int arr_b[], int size) {

	int i, ip = 0;
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
int inner_product_sparse( int data_a[], int indices_a[], int size_a, int data_b[], int indices_b[], int size_b ) {

	int ind_a, ind_b;
	int ip = 0, i_a = 0, i_b = 0;
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
void inner_product_row_all_sparse( int i, int data[], int indices[], int indptr[], int n_rows, int result[] ) {

	// Get starting ind
	int i_a = indptr[i];
	int size_a = indptr[i+1] - i_a;

	// Loop over all rows
	int i_b, size_b;
	int j = 0;
	for ( j = 0; j < n_rows; j++ ){

		// Other ind
		i_b = indptr[j];
		size_b = indptr[j+1] - i_b;
		
		// Calculate IP
		result[j] = inner_product_sparse( &data[i_a], &indices[i_a], size_a, &data[i_b], &indices[i_b], size_b );
	}
}

void inner_product_matrix( int data[], int indices[], int indptr[], int n_rows, int **result ) {

	int i, j, i_a, i_b, size_a, size_b;
	for ( i = 0; i < n_rows; i++ ){

		// Get ind
		int i_a = indptr[i];
		int size_a = indptr[i+1] - i_a;

		for ( j = i; j < n_rows ; j++ ) {

			// Other ind
			i_b = indptr[j];
			size_b = indptr[j+1] - i_b;
		
			result[i][j] = inner_product_sparse( &data[i_a], &indices[i_a], size_a, &data[i_b], &indices[i_b], size_b );
		}
	}
}

// The stronger test framework is setup for the frontend, but a simple test framework is found below.
int main () {
	// Inner product between two sparse rows.
	int data_a[4] = {1, 2, 3, 5};
	int indices_a[4] = { 0, 1, 4, 5};
	int data_b[5] = {-1, 2, 2, 5, 3};
	int indices_b[5] = { 0, 1, 3, 4, 17};

	int expected = -1 + 2 * 2 + 5 * 3;
	cout << "Expected is: "  << expected << endl;

	// Result value
	int ip = inner_product_sparse( data_a, indices_a, 4, data_b, indices_b, 5 );
	cout << "Inner product is: "  << ip << endl;

	// Inner product for a component matrix
	int data[9] = { 1, 2, 3, 5, -1, 2, 2, 5, 3 };
	int indices[9] = { 0, 1, 4, 5, 0, 1, 3, 4, 17 };
	int indptr[3] = { 0, 4, 9};
	int result[2] = { 0, 0 };
	int **result_all;
	result_all = new int *[2];
	for ( int i = 0; i < 2 ; i++ ) {
		result_all[i] = new int[2];
	}

	int expected_norm = 1 + 2 * 2 + 3 * 3 + 5 * 5;
	cout << "Inner product row-all expected is: "  << expected_norm << " " << expected << endl;

	// Result for one row
	inner_product_row_all_sparse( 0, data, indices, indptr, 2, result );
	cout << "Inner product row-all is: "  << result[0] << " " << result[1] << endl;

	int expected_all[2][2] = {
		{ expected_norm, expected, },
		{ 0, 1 * 1 + 2 * 2 + 2 * 2 + 5 * 5 + 3 * 3 },
	};
	cout << "Inner product matrix expected is: " << endl;
	cout << expected_all[0][0] << " " << expected_all[0][1] << endl;
	cout << expected_all[1][0] << "  " << expected_all[1][1] << endl;

	// Result for all
	inner_product_matrix( data, indices, indptr, 2, result_all );
	cout << "Inner product matrix is: " << endl;
	cout << result_all[0][0] << "  " << result_all[0][1] << endl;
	cout << result_all[1][0] << "  " << result_all[1][1] << endl;
}