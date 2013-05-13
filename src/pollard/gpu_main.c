/**
 * @brief TODO
 *
 * @author AJ Kaufmann
 * @author David Matlack
 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "kernel.h"

int serial_factorize(UL N, unsigned *primes, unsigned num_primes,
                      mpz_t *result) {
  (void) num_primes;
  mpz_t n, a, d, p, e, b, tmp, tmp_2, MPZ_ONE;
  mpz_init(&n);
  mpz_init(&a);
  mpz_init(&d);
  mpz_init(&p);
  mpz_init(&e);
  mpz_init(&b);
  mpz_init(&tmp);
  mpz_init(&tmp_2);

  mpz_init(&MPZ_ONE);
  mpz_set_i(&MPZ_ONE, 1);

  mpz_set_lui(&n, N);

  int count = 0;

  unsigned B;
  const unsigned B_MAX = TABLE_SIZE;

  for (B = B_START; B < B_MAX; B *= 2) {
    unsigned it;
    unsigned max_it = 80;
    unsigned p_i;

    mpz_set_lui(&e, (UL) 1);

    for (p_i = 0; primes[p_i] < B; p_i ++) {
      unsigned prime_ul = (UL) primes[p_i];

      mpz_set_lui(&p, prime_ul);

      mpz_set_lui(&tmp_2, (UL) (log((double) B) / log((double) prime_ul)));

      // tmp_2 = floor(log b / log p)
      mpz_powmod(&tmp, &p, &tmp_2, &n); // tmp = (p ** tmp_2) % n
      mpz_mult(&tmp_2, &tmp, &e);       // tmp_2 = tmp * e
      mpz_set(&e, &tmp_2);              // e = tmp_2
    }

    if (mpz_equal(&e, &MPZ_ONE)) continue;

    // try a variety of a values
    mpz_set_lui(&a, 2);
    for (it = 0; it < max_it; it ++) {
      // printf("it = %d\n", it);
      count ++;

      // check for a freebie
      mpz_gcd(&d, &a, &n);
      if (mpz_lt(&MPZ_ONE, &d)) {
        *result = d;
        // printf("Ran in %d iterations.\n", count);
        return 0;
      }

      mpz_powmod(&b, &a, &e, &n);  // b = (a ** e) % n
      mpz_sub(&tmp, &b, &MPZ_ONE); // tmp = b - 1
      mpz_gcd(&d, &tmp, &n);       // d = gcd(tmp, n)

      // success!
      if (mpz_lt(&MPZ_ONE, &d) && mpz_lt(&d, &n)) {
        *result = d;
        // printf("Ran in %d iterations.\n", count);
        return 0;
      }

      // otherwise get a new value for a
      // mpz_mult(&tmp, &a, &a);               // tmp = a ** 2
      // mpz_set_lui(&a, (UL) (i + it + tid)); // a = i + it + tid
      // mpz_add(&tmp_2, &tmp, &a);            // tmp_2 = &tmp + a
      // mpz_div(&tmp, &a, &tmp_2, &n);        // a = tmp_2 % n
      mpz_add(&tmp, &a, &MPZ_ONE);
      mpz_set(&a, &tmp);
    }
  }
  // couldn't find anything... :(
  // printf("Ran in %d iterations (and failed).\n", count);
  return -1;
}

int main(int argc, char *argv[]) {
  if (1 == argc) {
    fprintf(stderr, "Usage: %s <numbers to factor>\n", argv[0]);
    return -1;
  }

  int (*factorize)(UL, unsigned *, unsigned, mpz_t *) =
    ((strcmp(argv[1], "-s")) ? parallel_factorize : serial_factorize);
  unsigned num_to_factor = ((strcmp(argv[1], "-s")) ? 1 : 2);

  unsigned num_primes = TABLE_SIZE;
  unsigned *h_table = (unsigned *) malloc(num_primes * sizeof(unsigned));
  if (0 != generate_prime_table(&h_table, num_primes)) {
    return 0;
  }

  mpz_t factor;
  mpz_init(&factor);

  struct timeval start, end;

  for (; (int) num_to_factor < argc; num_to_factor ++) {

    UL n = (UL) atol(argv[num_to_factor]);
    printf("%lu: ", n);

    gettimeofday(&start, NULL);

    if (0 == factorize(n, h_table, num_primes, &factor)) {
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

  free(h_table);

  return 0;
}



