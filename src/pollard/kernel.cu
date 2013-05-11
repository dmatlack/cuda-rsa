#include <stdio.h>

#include "kernel.h"
#include "primegen.h"

__global__
void factorize_kernel(UL N, unsigned *primes, mpz_t *results) {
  UL K = 2;
  mpz_t n, a, d, k, t, tmp, tmp_2, MPZ_ONE;

  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;

  mpz_init(&n);
  mpz_init(&a);
  mpz_init(&d);
  mpz_init(&k);
  mpz_init(&t);
  mpz_init(&tmp);
  mpz_init(&tmp_2);

  mpz_init(&MPZ_ONE);
  mpz_set_i(&MPZ_ONE, 1);

  mpz_set_lui(&n, N);
  mpz_set_lui(&a, 2);

  unsigned max_it = 80;
  unsigned it = 0;

  mpz_set_lui(&k, (UL) primes[K]);
  for (;; it ++) {
    mpz_gcd(&d, &a, & n);
    if (mpz_equal(&MPZ_ONE, &d)) {
      results[tid] = d;
      return;
    }

    mpz_powmod(&t, &a, &k, &n);
    mpz_sub(&tmp, &t, &MPZ_ONE);
    mpz_gcd(&d, &t, &n);
    if (mpz_lt(&MPZ_ONE, &d) && mpz_lt(&d, &n)) {
      results[tid] = d;
      return;
    }

    mpz_add(&tmp, &a, &MPZ_ONE);
    if ((mpz_equal(&d, &n) && mpz_lt(&tmp, &n)) ||
        (it > max_it)) {
      mpz_set_lui(&k, (UL) primes[K ++]);
      mpz_div(&tmp_2, &tmp, &a, &n);
      mpz_set(&a, &tmp);
      it = 0;
    }
    else if (mpz_equal(&d, &MPZ_ONE)) {
      mpz_add(&tmp, &a, &MPZ_ONE);
      mpz_set(&a, &tmp);
    }
    else {
      results[tid] = MPZ_ONE;
      return;
    }
  }
}

int factorize(UL n, unsigned *primes, mpz_t *factor) {
  unsigned blocks = 1;
  unsigned threads_per_block = 1;
  unsigned threads = blocks * threads_per_block;

  size_t results_bytes = threads * sizeof(mpz_t);

  mpz_t *d_results;
  if ((cudaSuccess != cudaMalloc((void **) &d_results, results_bytes)) ||
      (cudaSuccess != cudaMemset(d_results, 0L, results_bytes))) {
    fprintf(stderr, "Unable to allocate results table!\n");
    return -1;
  }

  factorize_kernel<<<blocks, threads_per_block>>>(n, primes, d_results);

  mpz_t *tmp_results = (mpz_t *) malloc(results_bytes);
  if (NULL == tmp_results) {
    fprintf(stderr, "Error allocating temporary result storage!\n");
    return -1;
  }
  if (cudaSuccess != cudaMemcpy(tmp_results, d_results, results_bytes,
                                cudaMemcpyDeviceToHost)) {
    fprintf(stderr, "Unable to retrieve results from host!\n");
    return -1;
  }

  mpz_t MPZ_ZERO;
  mpz_init(&MPZ_ZERO);
  mpz_set_i(&MPZ_ZERO, 0);

  unsigned thread;
  for (thread = 0; thread < threads; thread ++) {
    if (!mpz_equal(&MPZ_ZERO, &tmp_results[thread])) {
      mpz_set(factor, &tmp_results[thread]);
      return 0;
    }
  }

  return -1;
}

void get_prime_table(unsigned *table, unsigned n) {
  primegen pg;

  primegen_init(&pg);

  for (unsigned i = 0; i < n; i ++) {
    table[i] = (unsigned) primegen_next(&pg);
  }
}

int generate_prime_table(unsigned **d_table) {
  /* approximately the number of unsigned 32-bit primes       */
  /* (actual number is 203,280,221)                           */
  /* paper claimed to use ~170,000,000 primes for experiments */
  /* may want to write this to disk at some point...          */
  unsigned primes = 200 * 1000 * 1000;
  unsigned *h_table = (unsigned *) malloc(primes * sizeof(unsigned));
  if (NULL == h_table) {
    fprintf(stderr, "Unable to allocate host prime table!\n");
    return -1;
  }

  printf("Generating prime table... "); fflush(stdout);
  get_prime_table(h_table, primes);
  printf("done!\n"); fflush(stdout);

  /* move prime table to the gpu */
  printf("Transferring table to the gpu... "); fflush(stdout);
  if ((cudaSuccess != cudaMalloc((void **) d_table,
                                 primes * sizeof(unsigned))) ||
      (cudaSuccess != cudaMemcpy((void *) *d_table, (void *) h_table,
                                 primes * sizeof(unsigned),
                                 cudaMemcpyHostToDevice))) {
    fprintf(stderr, "Unable to allocate device prime table!\n");
    return -1;
  }
  printf("done!\n"); fflush(stdout);

  free(h_table);

  return 0;
}
