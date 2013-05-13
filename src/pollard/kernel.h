#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "mpz.h"

#define RESULTS_PER_THREAD 2
#define TABLE_SIZE (200 * 1000 * 1000)
#define B_START 2

#define UL unsigned long

int parallel_factorize(mpz_t n, unsigned *table, unsigned num_primes, mpz_t *factor);

int generate_prime_table(unsigned **d_table, unsigned num_primes);

#endif /* __KERNEL_H__ */
