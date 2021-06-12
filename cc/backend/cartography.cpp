// using namespace std; // It's not clear to me when I should/shouldn't use this for ctypes...

extern "C" // required when using C++ compiler

int inner_product(int arr_a[], int arr_b[], int size) {

	int i, ip = 0;
	for (i = 0; i < size - 1; ++i ) {
		ip += arr_a[i] * arr_b[i];
	}

	return ip;
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