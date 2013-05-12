#include <stdio.h>

#include "kernel.h"
#include "primegen.h"

#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 1

#define B_START 2
#define TABLE_SIZE (200 * 1000 * 1000)

__global__
void parallel_factorize_kernel(UL N, unsigned *primes, bool *finished,
                               mpz_t *result) {
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
  unsigned max_B = ((N < TABLE_SIZE) ? N : TABLE_SIZE);
  for (B = B_START; B < max_B; B *= 2) {
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned threads = gridDim.x * blockDim.x;
    unsigned i = blockIdx.x * blockDim.x;

    unsigned it;
    unsigned max_it = 80;

    mpz_set_lui(&e, (UL) 1);
    unsigned p_i;
    for (p_i = tid; primes[p_i] < B; p_i += threads) {
      mpz_set_lui(&p, (UL) primes[p_i]);
      // TODO: replace MPZ_ONE with logB / logp
      mpz_set_lui(&tmp_2, (UL) (log((double) B) / log((double) primes[p_i])));
      // tmp_2 = floor(log b / log p)
      mpz_powmod(&tmp, &p, &tmp_2, &n); // tmp = (p ** tmp_2) % n
      mpz_mult(&tmp_2, &tmp, &e);       // tmp_2 = tmp * e
      mpz_set(&e, &tmp_2);              // e = tmp_2
    }

    if (mpz_equal(&e, &MPZ_ONE)) return;

    // char *e_str = mpz_get_str(&e, NULL, 0);
    // printf("\tUsing e = %s\n", e_str);
    // free(e_str);

    // try a variety of a values
    mpz_set_lui(&a, 2 + tid);
    for (it = 0; it < max_it; it ++) {
      count ++;
      if (*finished) {
        printf("Ran in %d iterations.\n", count);
        return;
      }
      // char *a_str = mpz_get_str(&a, NULL, 0);
      // printf("\t\tUsing a = %s\n", a_str);
      // free(a_str);
      // check for a freebie
      mpz_gcd(&d, &a, &n);
      if (mpz_lt(&MPZ_ONE, &d)) {
        *result = d;
        *finished = true;
        printf("Ran in %d iterations.\n", count);
        return;
      }

      mpz_powmod(&b, &a, &e, &n);  // b = (a ** e) % n
      mpz_sub(&tmp, &b, &MPZ_ONE); // tmp = b - 1
      mpz_gcd(&d, &tmp, &n);       // d = gcd(tmp, n)

      // success!
      if (mpz_lt(&MPZ_ONE, &d) && mpz_lt(&d, &n)) {
        *result = d;
        *finished = true;
        printf("Ran in %d iterations.\n", count);
        return;
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
  printf("Ran in %d iterations (and failed).\n", count);
}

int factorize(UL n, unsigned *primes, mpz_t *factor) {
  unsigned blocks = NUM_BLOCKS;
  unsigned threads_per_block = THREADS_PER_BLOCK;
  unsigned threads = blocks * threads_per_block;

  size_t result_bytes = sizeof(mpz_t);

  mpz_t *d_result;
  bool *d_finished;
  if ((cudaSuccess != cudaMalloc((void **) &d_result, result_bytes)) ||
      (cudaSuccess != cudaMalloc((void **) &d_finished, sizeof(bool))) ||
      (cudaSuccess != cudaMemset(d_result, 0L, result_bytes)) ||
      (cudaSuccess != cudaMemset(d_finished, false, sizeof(bool)))) {
    fprintf(stderr, "Unable to allocate device memory!\n");
    return -1;
  }

  parallel_factorize_kernel<<<blocks, threads_per_block>>>
    (n, primes, d_finished, d_result);

  if (cudaSuccess != cudaMemcpy(factor, d_result, result_bytes,
                                cudaMemcpyDeviceToHost)) {
    fprintf(stderr, "Unable to retrieve result from host!\n");
    return -1;
  }

  cudaFree(primes);
  cudaFree(d_result);
  cudaFree(d_finished);

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
  unsigned primes = TABLE_SIZE;
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
