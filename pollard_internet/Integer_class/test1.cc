/*******************************************************
*	test1.cc - sample program to show how to use the
*		   Integer class I wrote for the gnu gmp
*		   library.  Have fun!
*
*	Compile:
*	g++ -O4 -s -o test1 -I. test1.cc libgmp.a
*
*	Changelog:
*	971703: Created - Paul Herman <a540pau@pslc.ucla.edu>
*********************************************************/

#include "Integer.h"

int main(int argc, char **argv) {
	Integer a, b, c;

	if (argc != 3) {	// this program expects two arguments
		cerr << "Usage: " << argv[0] << " num1 num2" << endl;
		return -1;
		}

	a = argv[1];	// casting from char* to Integer is
	b = argv[2];	// done automaticaly

	c = a+b;
	cout << a << " + " << b << " = " << c << endl;

	c = a*b;
	cout << a << " x " << b << " = " << c << endl;

	c = a%b;
	cout << a << " % " << b << " = " << c << endl;

	c = gcd(a, b);
	cout << "gcd(" << a << ", " << b << ") = " << c << endl;

	c = lcm(a, b);
	cout << "lcm(" << a << ", " << b << ") = " << c << endl;
}
