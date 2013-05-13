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

  mpz_t n;
  mpz_t factor;

  mpz_init(&factor);
  mpz_init(&n);

  struct timeval start, end;

  unsigned num_to_factor;
  for (num_to_factor = 1; (int) num_to_factor < argc; num_to_factor ++) {
    mpz_init(&n);

    char *num_str = argv[num_to_factor];
    mpz_set_str(&n, num_str);
    
    printf("Factoring 0x%s: ", num_str);

    gettimeofday(&start, NULL);

    if (0 == factorize(&n, d_table, &factor)) {
      char factor_str[1024];

      gettimeofday(&end, NULL);
      long elapsed_us = (end.tv_sec * 1000 * 1000 + end.tv_usec) -
        (start.tv_sec * 1000 * 1000 + start.tv_usec);

      mpz_get_str(&factor, factor_str, 1024);
      printf("0x%s (in %ld us)\n", factor_str, elapsed_us);
    }
    else {
      printf("unable to find factor\n");
    }

  }

  return 0;
}



