#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "mpz.h"

#define RESULTS_PER_THREAD 2

#define UL unsigned long

int factorize(UL n, unsigned *table, mpz_t *factor);

int generate_prime_table(unsigned **d_table);

#endif /* __KERNEL_H__ */
