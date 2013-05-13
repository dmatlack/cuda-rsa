/******************************************************
 * Integer.h -- C++ Integer class for the gmp integer library (from the
 *   GNU project.) Allows you to use the arbitrary precision integers
 *   from gmp as if they were plain old (int)'s or (long int)'s.  Just
 *   include this file in your source code.  Example main():
 *
 *    #include "Integer.h"
 *    int main(int argc, char **argv) {
 *	Integer a, b, c;
 *
 *	a = argv[1];		// automaticaly converts a string
 *	b = argv[2];		//  into an Integer.
 *	c = a*b;
 *	cout << a << " x " << b << " = " << c << endl;
 *    }
 *
 ********
 * To Do:
 *   Rewrite all additions to check for signed (int)s  (Done! with 000605)
 * ChangeLog:
 *   970304 - Created    -- Paul Herman <a540pau@pslc.ucla.edu>
 *   980804 - Cleaned up -- Paul Herman <a540pau@pslc.ucla.edu>
 *   000605 - bug fixes  -- Paul Herman <pherman@frenchfries.net>
 *******************************************************/


/***************************** BEGIN ********************************/

#include <stdlib.h>

#ifndef		FALSE
#define		FALSE 0
#endif // FALSE
#ifndef		TRUE
#define		TRUE 1
#endif // TRUE

#ifndef __GMP_INTEGER_CLASS__
#define __GMP_INTEGER_CLASS__

using namespace std;

#include <gmp.h>		// provides the mpz integer routines
#include <iostream>		// for cin & cout istreams & ostreams
#include <time.h>		// use time as seed for random numbers

#ifndef		ABSmodN
#define		ABSmodN(x, n) (x%n)	// in gmp lib, (-x)%n = (n-x)%n
#endif // ABSmodN


/**************************** INTEGER ***************************************/

class Integer {
  private:
	mpz_t _x;
	int base;
  public:
//
// Constructors
//
    Integer(void);
    Integer(int a);
    Integer(unsigned int a);
    Integer(const Integer &a);
    Integer(char * a);
    Integer(mpz_t a);
//
// Destructor
//
    ~Integer(void);
//
// Simple access functions
//
    int		getbase(void);
    void	setbase(int b);
//
// Casting	(changing Integer to another type)
//
    operator bool(void);	// (bool) i
    operator int(void);		// (int)  i
    operator unsigned int(void);// (unsigned int) i
    operator float(void);	// (float) i
    operator double(void);	// (double)i
//
// Assignments	(Setting Integer equal to something else)
//
    void operator =(Integer a);		// c = (Integer) a;
    void operator =(int a);		// c = (int) a;
    void operator =(unsigned int a);	// c = (unsigned int) a;
    void operator =(char *a);		// c = a (a string);
    void operator =(const char *a);	// c = "1231231";
    void operator =(mpz_t a);		// c = a (a mpz number);
//
// Binary (one sided) operators
//
    Integer operator -(void);		// c = -a;
    bool operator !(void);		// c = (bool) (!a)
    void operator +=(Integer a);	// c += a;
    void operator +=(int a);		// c += a;
    void operator +=(unsigned int a);	// c += a;
    void operator -=(Integer a);	// c -= a;
    void operator -=(int a);		// c -= a;
    void operator -=(unsigned int a);	// c -= a;
    void operator *=(Integer a);	// c *= a;
    void operator *=(int a);		// c *= a;
    void operator *=(unsigned int a);	// c *= a;
    void operator /=(Integer a);	// c /= a;
    void operator /=(int a);		// c /= a;
    void operator /=(unsigned int a);	// c /= a;
    void operator %=(Integer a);	// c %= a;
    void operator %=(int a);		// c %= a;
    void operator ^=(int a);		// c ^= a;
// WARNING: No ^ with type (Integer) as exponent

    Integer operator ++(void);		// c = a++;
    Integer operator --(void);		// c = a--;
    Integer operator ++(int from);	// c = ++a;
    Integer operator --(int from);	// c = --a;

//
// The following operators must be friendly because the are binary
//
    friend Integer operator +(Integer a, Integer b);	// a + b;
    friend Integer operator +(Integer a, int b);	// a + b;
    friend Integer operator +(int a, Integer b);	// a + b;
    friend Integer operator -(Integer a, Integer b);	// a - b;
    friend Integer operator -(Integer a, int b);	// a - b;
    friend Integer operator -(int a, Integer b);	// a - b;
    friend Integer operator *(Integer a, Integer b);	// a * b;
    friend Integer operator *(Integer a, int b);	// a * b;
    friend Integer operator *(int a, Integer b);	// a * b;
    friend Integer operator /(Integer a, Integer b);	// a / b;
    friend Integer operator /(Integer a, int b);	// a / b;
    friend Integer operator /(int a, Integer b);	// a / b;
    friend Integer operator %(Integer a, Integer b);	// a % b;
    friend Integer operator %(Integer a, int b);	// a % b;
    friend Integer operator %(int a, Integer b);	// a % b;
    friend Integer operator ^(Integer a, int b);	// a ^ b;
    friend Integer operator &(Integer a, Integer b);	// a & b;
    friend Integer operator |(Integer a, Integer b);	// a | b;

    friend bool operator ==(Integer a, Integer b);	// (c == a)?
    friend bool operator ==(Integer a, int b);		// (c == a)?
    friend bool operator ==(int a, Integer b);		// (c == a)?
    friend bool operator !=(Integer a, Integer b);	// (c != a)?
    friend bool operator !=(Integer a, int b);		// (c != a)?
    friend bool operator !=(int a, Integer b);		// (c != a)?
    friend bool operator <(Integer a, Integer b);	// (c < a)?
    friend bool operator <(Integer a, int b);		// (c < a)?
    friend bool operator <(int a, Integer b);		// (c < a)?
    friend bool operator <=(Integer a, Integer b);	// (c <= a)?
    friend bool operator <=(Integer a, int b);		// (c <= a)?
    friend bool operator <=(int a, Integer b);		// (c <= a)?
    friend bool operator >(Integer a, Integer b);	// (c > a)?
    friend bool operator >(Integer a, int b);		// (c > a)?
    friend bool operator >(int a, Integer b);		// (c > a)?
    friend bool operator >=(Integer a, Integer b);	// (c >= a)?
    friend bool operator >=(Integer a, int b);		// (c >= a)?
    friend bool operator >=(int a, Integer b);		// (c >= a)?
    friend ostream& operator <<(ostream &s, Integer a);		// cout << a;
    friend istream& operator >>(istream &s, Integer &a);	// cin >> a;
    friend Integer& operator <<(Integer a, int b);		// a << b;
    friend Integer& operator >>(Integer a, int b);		// a >> b;
//
// These functions are friendly to speed up access to the class'
//  private parts.
//
    friend void		Int2a(Integer i, char *string);
    friend Integer	gcd(Integer a, Integer b);
    friend Integer	exgcd(Integer a, Integer b, Integer &c, Integer &d);
    friend Integer	InvModN(Integer a, Integer n);
    friend Integer	PowModN(Integer base, Integer exp, Integer n);
    friend Integer	PowModN(Integer base, int exp, Integer n);
    friend bool		isprime(Integer a);
    friend bool		issquare(Integer a);
    friend bool		testbit(Integer a, unsigned long int bit_number);
    friend Integer	sqrt(Integer a);
    friend int		digits(Integer a, int base);
    friend int		Jacobi(Integer m, Integer n);
    friend int		Legendre(Integer m, Integer n);
    friend Integer	random(Integer range);
}; // end 'class Integer'
//
//  Misc Number theory functions (non-friendly)
//
Integer	lcm(Integer a, Integer b);
Integer	LCM(Integer a);		// LCM(a) = lcm[1, 2, 3, ..., a]
int	logn(Integer a, int base);
int	log2(Integer a);
Integer	fact(Integer a);
Integer P_n_k(Integer n, Integer k);
Integer C_n_k(Integer n, Integer k);
Integer	nextprime(Integer a);
Integer	prevprime(Integer a);
Integer	SqrtModN(Integer x, Integer p);

#endif	//	__GMP_INTEGER_CLASS__
