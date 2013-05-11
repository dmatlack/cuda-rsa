#include <stdio.h>

#include "kernel.h"
#include "primegen.h"

__global__
void parallel_factorize_kernel(UL N, unsigned B, unsigned *primes,
                               mpz_t *results) {
  mpz_t n, a, d, p, e, b, tmp, tmp_2, MPZ_ONE;

  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned threads = gridDim.x * blockDim.x;
  unsigned i = blockIdx.x * blockDim.x;

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

  unsigned it;
  unsigned max_it = 80;

  mpz_set_lui(&e, (UL) 1);
  unsigned p_i;
  for (p_i = tid; primes[p_i] < B; p_i += threads) {
    mpz_set_lui(&p, (UL) primes[p_i]);
    // TODO: replace MPZ_ONE with logB / logp
    mpz_powmod(&tmp, &p, &MPZ_ONE, &n); // tmp = (p ** 1) % n
    mpz_mult(&tmp_2, &tmp, &e);         // tmp_2 = tmp * e
    mpz_div(&tmp, &e, &tmp_2, &n);      // e = tmp_2 % n
  }

  char *e_str = mpz_get_str(&e, NULL, 0);
  printf("\tUsing e = %s\n", e_str);
  free(e_str);

  // try a variety of a values
  mpz_set_lui(&a, 2 + tid);
  for (it = 0; it < max_it; it ++) {
    char *a_str = mpz_get_str(&a, NULL, 0);
    printf("\t\tUsing a = %s\n", a_str);
    free(a_str);
    // check for a freebie
    mpz_gcd(&d, &a, &n);
    if (mpz_lt(&MPZ_ONE, &d)) {
      results[tid] = d;
      return;
    }

    mpz_powmod(&b, &a, &e, &n);  // b = (a ** e) % n
    mpz_sub(&tmp, &b, &MPZ_ONE); // tmp = b - 1
    mpz_gcd(&d, &tmp, &n);       // d = gcd(tmp, n)

    // success!
    if (mpz_lt(&MPZ_ONE, &d) && mpz_lt(&d, &n)) {
      results[tid] = d;
      return;
    }

    // otherwise get a new value for a
    mpz_mult(&tmp, &a, &a);               // tmp = a ** 2
    mpz_set_lui(&a, (UL) (i + it + tid)); // a = i + it + tid
    mpz_add(&tmp_2, &tmp, &a);            // tmp_2 = &tmp + a
    mpz_div(&tmp, &a, &tmp_2, &n);        // a = tmp_2 % n
  }
  // couldn't find anything... :(
}

__global__
void serial_factorize_kernel(UL N, unsigned *primes, mpz_t *results) {
  mpz_t n, a, d, p, e, b, tmp, tmp_2, MPZ_ONE;

  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;

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

  unsigned it;
  unsigned max_it = 80;

  unsigned B;
  unsigned max_B = ((N < 1000 * 1000) ? N : 1000 * 1000);
  unsigned p_i = 0;
  mpz_set_lui(&e, (UL) 1);
  // we'll abort at about 1 million
  for (B = 2; B <= max_B; B *= 2) {
    // get a new e
    for (; primes[p_i] <= B; p_i ++) {
      mpz_set_lui(&p, (UL) primes[p_i]);
      // TODO: replace MPZ_ONE with logB / logp
      mpz_powmod(&tmp, &p, &MPZ_ONE, &n); // tmp = (p ** 1) % n
      mpz_mult(&tmp_2, &tmp, &e);         // tmp_2 = tmp * e
      mpz_div(&tmp, &e, &tmp_2, &n);      // e = tmp_2 % n
    }

    char *e_str = mpz_get_str(&e, NULL, 0);
    printf("Using B = %u (e = %s)\n", B, e_str);
    free(e_str);

    // try a variety of a values
    // right now, just start at two and increment
    mpz_set_lui(&a, 2);
    for (it = 0; it < max_it; it ++) {
      char *a_str = mpz_get_str(&a, NULL, 0);
      printf("\tUsing a = %s\n", a_str);
      free(a_str);
      // check for a freebie
      mpz_gcd(&d, &a, &n);
      if (mpz_lt(&MPZ_ONE, &d)) {
        results[tid] = d;
        return;
      }

      mpz_powmod(&b, &a, &e, &n);  // b = (a ** e) % n
      mpz_sub(&tmp, &b, &MPZ_ONE); // tmp = b - 1
      mpz_gcd(&d, &tmp, &n);       // d = gcd(tmp, n)

      // success!
      if (mpz_lt(&MPZ_ONE, &d) && mpz_lt(&d, &n)) {
        results[tid] = d;
        return;
      }

      // if gcd(a ** e - 1, n) == 1, get a new B
      if (mpz_equal(&d, &MPZ_ONE)) {
        break;
      }
      // otherwise gcd(a ** e - 1, n) == n - 1 --> get a new a
      mpz_add(&tmp, &a, &MPZ_ONE); // tmp = a + 1
      mpz_set(&a, &tmp);           // a = tmp
    }
  }
  // couldn't find anything... :(
}

int factorize(UL n, unsigned *primes, mpz_t *factor) {
  unsigned blocks = 1024;
  unsigned threads_per_block = 64;
  unsigned threads = blocks * threads_per_block;

  size_t results_bytes = threads * sizeof(mpz_t);

  mpz_t *d_results;
  if ((cudaSuccess != cudaMalloc((void **) &d_results, results_bytes)) ||
      (cudaSuccess != cudaMemset(d_results, 0L, results_bytes))) {
    fprintf(stderr, "Unable to allocate results table!\n");
    return -1;
  }

  unsigned B;
  unsigned max_B = ((n < 1000 * 1000) ? n : 1000 * 1000);
  for (B = 2; B < max_B; B *= 2) {
    printf("Using B = %u\n", B);
    parallel_factorize_kernel<<<blocks, threads_per_block>>>(n, B, primes, d_results);

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
  unsigned primes = // 200 *
    1000 * 1000;
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
