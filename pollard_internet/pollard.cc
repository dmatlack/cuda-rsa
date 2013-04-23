/***************************************************************************
 *	pollard.cc -- Use Pollard's p-1 Algorithm to factor a large integer
 * 
 *	Uses Pollard's algorithm to (sometimes) factor a large integer into
 *	smaller pieces.  Note this algorithm has a rather high failure rate,
 *	and is only used as an intro to Lenstra's Elliptic Curve Algorithm.
 *	The algorithm was taken from "Rational Points on Elliptic Curves", by
 *	Silverman/Tate, pages 130-132.
 *
 *	Compile:
 *	 g++ -s -O4 -o pollard pollard.cc
 *	Invoke:
 *	 ./pollard NumberToFactor A K
 *	Where A is the base for the a^k-1 calculation (usually A=2).
 *	  and K is a small number, s.t. LCM{1,2,...,K} is product of small
 *	  primes to small powers.
 *
 * ChangeLog:
 *  950516 -- Created by Ami Fischman <fischman@math.ucla.edu>
 *  970301 -- minor fixes -- Paul Herman <a540pau@pslc.ucla.edu>
 *  970324 -- added iteration condition -- Paul
 **************************************************************************/

#include "Integer_class/Integer.h"

#define max_iter 80	// number of iterations before trying a new K
// this, I think, should be a factor of the size of n

Integer factor(Integer n, Integer a, Integer K) {
  /* Return a non-trivial factor of N using Pollard's p-1 algorithm */
  /* Assumes 1<a<n, and k=LCM[1,2,3,4...K], some K */
  Integer d, k, t;
  int iteration = 0;

  k = LCM(K);
  for (;;iteration++) {
    if (gcd(a, n) > 1) return gcd(a, n);	// See if we get a freebee.
    // (Besides, gcd(a,n) must
    //   be 1 for the rest to work)

    t = PowModN(a, k, n);			// We will allways have 1<t<n
    d = gcd(t-1, n);			//   ...so no div-by-zero.
    if ( (1<d) && (d<n) ) return d;		// We have a divisor!
    if ( (0==(K%60)) || (d==n && a+1<n ) || iteration>max_iter) {
      k = LCM(++K);			// Change exponent
      a %= n; iteration = 0;
      //	    cout << "Trying new K = " << K << endl;
    }
    else if ( d==1 ) {
      a++;				// Try another base
      //	    cout << "Trying new a = " << a << endl;
    }
    else break;
  }
  cout << "K is " << K << " -- a is " << a << " -- n is " << n << " -- d is "<< d << endl;
  return 0;
}

int main(int argc, char *argv[]) {
  Integer n, pf, a, k;

  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " NumberToFactor [K]" << endl;
    return 1;
  }

  n = argv[1];
  a = 2;
  k = 2;
  if (argc == 3) k = argv[2];

  cout << n << ": " << flush;

  while(!isprime(n)) {
    pf = factor(n, a, k);
    cout << pf << " " << flush;
    n /= pf;
  }
  cout << n << endl;
  return 0;
}



