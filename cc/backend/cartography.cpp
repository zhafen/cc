// using namespace std; // It's not clear to me when I should/shouldn't use this for ctypes...

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
	for (i = 0; i < size - 1; ++i ) {
		ip += arr_a[i] * arr_b[i];
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
int inner_product_row_all_sparse(int ind, int data[], int indices[], int data_size, int indptr[], int n_rows ) {

	// Retrieve the target row indices and values.
	// This is python thinking, and will break.
	// I'll need to do otherwise, basically...
	row_data = data[indptr[ind]:indptr[ind+1]]
	row_indices = indices[indptr[ind]:indptr[ind+1]]
}

/**
int main () {
	int ip;

	int a[4] = {1, 2, 3, 5};
	int b[4] = {-1, 2, 2, 5};

	ip = inner_product( a, b, 5 ) ;

	// Return value
	cout << "Inner product is: "  << ip << endl;
}
*/