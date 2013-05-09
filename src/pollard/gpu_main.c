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
#include <stdlib.h>
#include <stdio.h>

#include "kernel.h"

int main(int argc, char *argv[]) {
  (void) argc;
  (void) argv;

  unsigned *d_table;
  if (0 != generate_prime_table(&d_table)) {
    return 0;
  }

  UL to_factor[] = { 500L, 1982301930192L };
  unsigned length = 2;

  unsigned num_to_factor;
  for (num_to_factor = 0; num_to_factor < length; num_to_factor ++) {

    UL n = to_factor[num_to_factor];
    printf("%lu: ", n);

    mpz_t factor;
    if (0 == factorize(n, d_table, &factor)) {
      char *factor_str = mpz_get_str(&factor, NULL, 0);
      printf("%s\n", factor_str);
      free(factor_str);
    }
    else {
      printf("unable to find factor\n");
    }
  }

  return 0;
}



