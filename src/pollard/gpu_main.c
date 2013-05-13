/**
 * @brief TODO
 *
 * @author AJ Kaufmann
 * @author David Matlack
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>


#include "kernel.h"

using namespace std;

void lcm(mpz_t *lcm, mpz_t *a, mpz_t *b) {
  mpz_t prod;
  mpz_t gcd;
  mpz_t tmp;

  mpz_init(&prod);
  mpz_init(&gcd);
  mpz_init(&tmp);

  mpz_mult(&prod, a, b);
  mpz_gcd(&gcd, a, b);

  mpz_div(lcm, &tmp, &prod, &gcd);
}

void LCM(mpz_t *e, unsigned *primes, unsigned B, mpz_t *tmp) {
  (void) primes;
  (void) tmp;

  if (B < 3) mpz_set_lui(e, 2);
  else {
    mpz_t b;
    mpz_t l;

    mpz_init(&b);
    mpz_init(&l);

    mpz_set_lui(&b, B);

    LCM(&l, primes, B-1, tmp);
    lcm(e, &l, &b);
  }
}

void PP(mpz_t *e, unsigned *primes, unsigned B, mpz_t *tmp) {
  unsigned p_i;
  unsigned power;
  unsigned prime_ul = (UL) primes[0];

  mpz_set_lui(e, 1);

  for (p_i = 0; prime_ul < B; p_i ++) {
    power = (unsigned) (log((double) B) / log((double) prime_ul));

    mpz_mult_u(tmp, e, pow(prime_ul, power));
    mpz_set(e, tmp);

    prime_ul = primes[p_i + 1];
  }

}

int cpu_factor(mpz_t n, unsigned *primes, unsigned num_primes,
                      mpz_t *result) {
  (void) num_primes;

  unsigned B;
  unsigned max_it;
  unsigned iteration;
  mpz_t a, d, e, b, tmp;

  mpz_init(&a);
  mpz_init(&d);
  mpz_init(&e);
  mpz_init(&b);
  mpz_init(&tmp);

  mpz_set_lui(&a, 2);
  B = 2;
  max_it = 2000;

  PP(&e, primes, B, &tmp);
  for (;;iteration++) {
    mpz_gcd(&d, &a, &n);

    if (mpz_gt_one(&d)) {
      *result = d;
      return 0;
    }

    mpz_powmod(&b, &a, &e, &n); // b = (a ** e) % n
    mpz_addeq_i(&b, -1);        // b -= 1
    mpz_gcd(&d, &b, &n);        // d = gcd(b, n)

    if (mpz_gt_one(&d) && mpz_lt(&d, &n)) {
      *result = d;
      return 0;
    }

    // tmp = a+1
    mpz_set(&tmp, &a);
    mpz_addeq_i(&tmp, 1);

    if ( (0 == (B%60)) || (mpz_equal(&d, &n) && mpz_lt(&tmp, &n)) ||
         (iteration > max_it) ) {
      PP(&e, primes, ++B, &tmp);

      // a %= n
      mpz_div(&tmp, &d, &a, &n);
      mpz_set(&a, &d);
      
      iteration = 0;
    }
    else if (mpz_equal_one(&d)) {
      mpz_addeq_i(&a, 1);
    }
    else break;
  }

  // couldn't find anything... :(
  // printf("Ran in %d iterations (and failed).\n", count);
  return -1;
}

int serial_factorize(mpz_t n, unsigned *primes, unsigned num_primes,
                      mpz_t *result) {
  struct timeval start, end;
  int ret;

  gettimeofday(&start, NULL);

  ret = cpu_factor(n, primes, num_primes, result);

  gettimeofday(&end, NULL);

  long elapsed_us = (end.tv_sec * 1000 * 1000 + end.tv_usec) -
    (start.tv_sec * 1000 * 1000 + start.tv_usec);

  printf("(in %lu us) ", elapsed_us);

  return ret;
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
  mpz_t mod;
  mpz_t div;

  mpz_init(&factor);
  mpz_init(&n);
  mpz_init(&div);
  mpz_init(&mod);

  for (; (int) num_to_factor < argc; num_to_factor ++) {

    char *num_str = argv[num_to_factor];
    mpz_set_str(&n, num_str);

    printf("Factoring 0x%s: ", num_str);
    fflush(stdout);

    factorize(n, h_table, num_primes, &factor);

    char factor_str[1024];
    mpz_get_str(&factor, factor_str, 1024);
    printf("%s ", factor_str);

    mpz_div(&div, &mod, &n, &factor);
    mpz_set_lui(&div, 0);
    if (mpz_equal(&mod, &div)) { // mod == 0
      printf("correct");
    }
    else {
      printf("incorrect!");
    }

    printf("\n");
  }

  free(h_table);

  return 0;
}



