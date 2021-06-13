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
	while ( i_a < size_a - 1 & i_b < size_b - 1)
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

/**
 * Inner product of all rows in a sparse matrix with a single row from that matrix.
 * Output is an array the size of the number of rows.
 * The sparse matrix is assumed to be in compressed sparse row format.
 * 
 * @ind The target row.
 * @data The data values for the sparse matrix.
 * @indices The indices for each data value.
 * @data_size The total number of data values/indices.
 * @indptr Where indices for one row ends and another begins.
 * @n_rows The number of rows.
 */
//int * inner_product_row_all_sparse(int ind, int data[], int indices[], int data_size, int indptr[], int n_rows, int result[] ) {
//
//	// Retrieve the target row indices and values.
//	// This is python thinking, and will break.
//	// I'll need to do otherwise, basically...
//	//row_data = data[indptr[ind]:indptr[ind+1]]
//	//row_indices = indices[indptr[ind]:indptr[ind+1]]
//
//	// int i, ip = 0;
//	// int start_ind = indptr[ind];
//	// int end_ind = indptr[ind+1];
//
//	// Loop over all rows
//	int i = 0;
//	for ( i = 0; i < n_rows - 1; i++ ){
//		
//		// Calculate IP
//		int ip = 0;
//	}
//
//	return ip;
//}

/**
int main () {
	int ip;

	int data_a[4] = {1, 2, 3, 5};
	int indices_a[4] = { 0, 1, 4, 5};
	int data_b[5] = {-1, 2, 2, 5, 3};
	int indices_b[5] = { 0, 1, 3, 4, 17};

	int expected = -1 + 2 * 2 + 5 * 3;

	ip = inner_product_sparse( data_a, indices_a, 4, data_b, indices_b, 5 );

	// Return value
	cout << "Inner product is: "  << ip << endl;
	cout << "Expected is: "  << expected << endl;
}
*/