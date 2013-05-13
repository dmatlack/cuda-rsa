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
#include <iostream>
#include <sys/time.h>

#include "kernel.h"

using namespace std;

int serial_factorize(mpz_t n, unsigned *primes, unsigned num_primes,
                      mpz_t *result) {
  (void) num_primes;
  /* unsigned tid = 0; */
  /* unsigned threads = 1; */
  /* unsigned i = 0; */

  mpz_t a, d, e, b, tmp, tmp_2;
  mpz_init(&a);
  mpz_init(&d);
  mpz_init(&e);
  mpz_init(&b);
  mpz_init(&tmp);
  mpz_init(&tmp_2);

  int count = 0;

  unsigned B;
  const unsigned B_MAX = TABLE_SIZE;

  // try a variety of a values
  mpz_set_lui(&a, 3);

  for (B = B_START; B < B_MAX; B ++) {
    unsigned it;
    unsigned max_it = 80;

    unsigned p_i;
    unsigned power;
    unsigned prime_ul = (UL) primes[0];
    mpz_set_lui(&e, (UL) 1);
    for (p_i = 0; prime_ul < B; p_i ++) {

      power = (unsigned) (log((double) B) /
                          log((double) prime_ul));

      mpz_mult_u(&tmp_2, &e, pow(prime_ul, power)); // tmp_2 = (p ** power) * e
      mpz_set(&e, &tmp_2);        // e = tmp_2

      prime_ul = primes[p_i + 1];
    }

    if (mpz_equal_one(&e)) continue;

    for (it = 0; it < max_it; it ++) {
      // printf("it = %d\n", it);
      count ++;

      // check for a freebie
      mpz_gcd(&d, &a, &n);
      if (mpz_gt_one(&d)) {
        *result = d;
        // printf("Ran in %d iterations.\n", count);
        return 0;
      }

      mpz_powmod(&b, &a, &e, &n); // b = (a ** e) % n
      mpz_addeq_i(&b, -1);        // b -= 1
      mpz_gcd(&d, &b, &n);        // d = gcd(b, n)

      /* char buf[1024]; */
      /* mpz_get_str(&b, buf, 1024); */
      /* cout << "b = " << buf; */
      /* mpz_get_str(&a, buf, 1024); */
      /* cout << ", a = " << buf; */
      /* mpz_get_str(&e, buf, 1024); */
      /* cout << ", e = " << buf; */
      /* mpz_get_str(&n, buf, 1024); */
      /* cout << ", n = " << buf << endl; */

      // success!
      if (mpz_gt_one(&d) && mpz_lt(&d, &n)) {
        *result = d;
        // printf("Ran in %d iterations.\n", count);
        return 0;
      }

      // otherwise get a new value for a
#if 0
      mpz_mult(&tmp, &a, &a);               // tmp = a ** 2
      mpz_set_lui(&a, (UL) (i + it + tid)); // a = i + it + tid
      mpz_add(&tmp_2, &tmp, &a);            // tmp_2 = &tmp + a
      mpz_div(&tmp, &a, &tmp_2, &n);        // a = tmp_2 % n
#else
      mpz_addeq_i(&a, 1);       /* a += 1 */
#endif
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

  int (*factorize)(mpz_t, unsigned *, unsigned, mpz_t *) =
    ((strcmp(argv[1], "-s")) ? parallel_factorize : serial_factorize);
  unsigned num_to_factor = ((strcmp(argv[1], "-s")) ? 1 : 2);

  unsigned num_primes = TABLE_SIZE;
  unsigned *h_table = (unsigned *) malloc(num_primes * sizeof(unsigned));
  if (0 != generate_prime_table(&h_table, num_primes)) {
    return 0;
  }

  mpz_t n;
  mpz_t factor;

  mpz_init(&factor);
  mpz_init(&n);

  struct timeval start, end;

  for (; (int) num_to_factor < argc; num_to_factor ++) {

    char *num_str = argv[num_to_factor];
    mpz_set_str(&n, num_str);

    printf("Factoring 0x%s: ", num_str);

    gettimeofday(&start, NULL);

    if (0 == factorize(n, h_table, num_primes, &factor)) {
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

  free(h_table);

  return 0;
}



