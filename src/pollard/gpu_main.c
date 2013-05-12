/**
 * @brief TODO
 *
 * @author AJ Kaufmann
 * @author David Matlack
 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include "kernel.h"

int main(int argc, char *argv[]) {
  unsigned *d_table;
  if (0 != generate_prime_table(&d_table)) {
    return 0;
  }

  mpz_t factor;
  mpz_init(&factor);

  struct timeval start, end;

  unsigned num_to_factor;
  for (num_to_factor = 1; (int) num_to_factor < argc; num_to_factor ++) {

    UL n = (UL) atol(argv[num_to_factor]);
    printf("%lu: ", n);

    gettimeofday(&start, NULL);

    if (0 == factorize(n, d_table, &factor)) {
      char factor_str[1024];

      gettimeofday(&end, NULL);
      long elapsed_us = (end.tv_sec * 1000 * 1000 + end.tv_usec) -
        (start.tv_sec * 1000 * 1000 + start.tv_usec);

      mpz_get_str(&factor, factor_str, 1024);
      printf("%lu (in %ld us)\n", strtol(factor_str, NULL, 16), elapsed_us);
    }
    else {
      printf("unable to find factor\n");
    }

  }

  return 0;
}



