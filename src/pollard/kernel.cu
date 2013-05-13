#include <stdio.h>

#include "kernel.h"
#include "primegen.h"

#include <time.h>
#include <sys/time.h>

#define NUM_BLOCKS 64
#define THREADS_PER_BLOCK 32

__constant__ unsigned c_table[TABLE_SIZE];

__global__
void parallel_factorize_kernel(mpz_t n, unsigned *primes, volatile bool *finished,
                               mpz_t *result) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned threads = gridDim.x * blockDim.x;
  const unsigned bid = blockIdx.x;
  // unsigned i = blockIdx.x * blockDim.x;

  const unsigned max_it = 400000 / threads;

  const unsigned b_start = B_START + bid;//bid * blockDim.x / max_it;
  const unsigned b_inc = gridDim.x;//threads / max_it;

  unsigned B;
  const unsigned B_MAX = TABLE_SIZE;
  unsigned it;
  unsigned p_i;
  unsigned power;
  unsigned prime_ul;

  mpz_t a, d, e, b, tmp;

  mpz_init(&a);
  mpz_init(&d);
  mpz_init(&e);
  mpz_init(&b);
  mpz_init(&tmp);

  // try a variety of a values
  mpz_set_ui(&a, (UL) tid + 2);

  for (B = b_start; B < B_MAX; B += b_inc) {

    /* Compute e as a product of prime powers */
    prime_ul = (UL) c_table[0];
    mpz_set_lui(&e, (UL) 1);
    for (p_i = 0; prime_ul < B; p_i ++) {
      if (*finished) return;

      power = (unsigned) (log((double) B) / log((double) prime_ul));
      mpz_mult_u(&tmp, &e, (unsigned) pow((double) prime_ul, (double) power));

      if (*finished) return;

      mpz_set(&e, &tmp);
      prime_ul = c_table[p_i + 1];
    }

    if (mpz_equal_one(&e)) continue;
    // if (*finished) return;

    for (it = 0; it < max_it; it ++) {
      // printf("it = %d\n", it);

      if (*finished) return;

      // check for a freebie
      mpz_gcd(&d, &a, &n);
      // if (*finished) return;
      if (mpz_gt_one(&d)) {
        *result = d;
        *finished = true;
      }
      // if (*finished) return;

      mpz_powmod(&b, &a, &e, &n);  // b = (a ** e) % n
      mpz_addeq_i(&b, -1); // b -= 1
      mpz_gcd(&d, &b, &n);       // d = gcd(tmp, n)

      // if (*finished) return;

      // success!
      if (mpz_gt_one(&d) && mpz_lt(&d, &n)) {
        *result = d;
        *finished = true;
      }

      // if (*finished) return;
      // otherwise get a new value for a
#if 0
      mpz_mult(&tmp, &a, &a);               // tmp = a ** 2
      mpz_set_lui(&a, (UL) (i + it + tid)); // a = i + it + tid
      mpz_add(&tmp_2, &tmp, &a);            // tmp_2 = &tmp + a
      mpz_div(&tmp, &a, &tmp_2, &n);        // a = tmp_2 % n
#else
      mpz_addeq_i(&a, threads * max_it);       // a += 1
#endif
    }
  }
  // couldn't find anything... :(
  // printf("Ran in %d iterations (and failed).\n", count);
}

int parallel_factorize(mpz_t n, unsigned *h_table, unsigned num_primes,
                       mpz_t *factor) {
  unsigned blocks = NUM_BLOCKS;
  unsigned threads_per_block = THREADS_PER_BLOCK;
  //unsigned threads = blocks * threads_per_block;

  size_t result_bytes = sizeof(mpz_t);

  /* move prime table to the gpu */
  // unsigned *d_table;
  if (cudaSuccess != cudaMemcpyToSymbol(c_table, h_table,
                                        num_primes * sizeof(unsigned))) {
    fprintf(stderr, "Unable to allocate device prime table!\n");
    return -1;
  }

  mpz_t *d_result;
  bool *d_finished;
  if ((cudaSuccess != cudaMalloc((void **) &d_result, result_bytes)) ||
      (cudaSuccess != cudaMalloc((void **) &d_finished, sizeof(bool))) ||
      (cudaSuccess != cudaMemset(d_result, 0L, result_bytes)) ||
      (cudaSuccess != cudaMemset(d_finished, false, sizeof(bool)))) {
    fprintf(stderr, "Unable to allocate device memory!\n");
    return -1;
  }

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  struct timeval start, end;

  gettimeofday(&start, NULL);

  parallel_factorize_kernel<<<blocks, threads_per_block>>>
    (n, NULL, d_finished, d_result);

  cudaDeviceSynchronize();

  gettimeofday(&end, NULL);

  long elapsed_us = (end.tv_sec * 1000 * 1000 + end.tv_usec) -
    (start.tv_sec * 1000 * 1000 + start.tv_usec);

  printf("(in %ld us) ", elapsed_us);

  if (cudaSuccess != cudaMemcpy(factor, d_result, result_bytes,
                                cudaMemcpyDeviceToHost)) {
    fprintf(stderr, "Unable to retrieve result from host!\n");
    return -1;
  }

  cudaFree(d_result);
  cudaFree(d_finished);

  return 0;
}

void get_prime_table(unsigned *table, unsigned n) {
  primegen pg;

  primegen_init(&pg);

  for (unsigned i = 0; i < n; i ++) {
    table[i] = (unsigned) primegen_next(&pg);
  }
}

int generate_prime_table(unsigned **h_table, unsigned num_primes) {
  /* approximately the number of unsigned 32-bit primes       */
  /* (actual number is 203,280,221)                           */
  /* paper claimed to use ~170,000,000 primes for experiments */
  /* may want to write this to disk at some point...          */
  *h_table = (unsigned *) malloc(num_primes * sizeof(unsigned));
  if (NULL == *h_table) {
    fprintf(stderr, "Unable to allocate host prime table!\n");
    return -1;
  }

  printf("Generating prime table... "); fflush(stdout);
  get_prime_table(*h_table, num_primes);
  printf("done!\n"); fflush(stdout);

  return 0;
}
