#include "kernel.h"
#include <stdio.h>

__global__
void factorize_kernel(UL n, unsigned *table, UL *results) {
  return;
}

int factorize(UL n, unsigned *table, UL **h_results) {
  unsigned blocks = 1;
  unsigned threads_per_block = 1;
  unsigned threads = blocks * threads_per_block;

  unsigned results_per_thread = max(RESULTS_PER_THREAD, 1000 / threads);
  unsigned max_results = threads * results_per_thread;
  size_t results_bytes = max_results * sizeof(UL);

  UL *d_results;
  if ((cudaSuccess != cudaMalloc((void **) &d_results, results_bytes)) ||
      (cudaSuccess != cudaMemset(d_results, -1L, results_bytes))) {
    fprintf(stderr, "Unable to allocate results table!\n");
    return -1;
  }

  factorize_kernel<<<blocks, threads_per_block>>>(n, table, d_results);

  UL *tmp_results = (UL *) malloc(results_bytes);
  if (NULL == tmp_results) {
    fprintf(stderr, "Error allocating temporary result storage!\n");
    return -1;
  }
  if (cudaSuccess != cudaMemcpy(tmp_results, d_results, results_bytes,
                                cudaMemcpyDeviceToHost)) {
    fprintf(stderr, "Unable to retrieve results from host!\n");
    return -1;
  }

  unsigned to = 0;
  unsigned from;
  for (from = 0; from < max_results; from ++) {
    if (0L != tmp_results[from]) {
      tmp_results[to] = (UL) tmp_results[from];
    }
    continue;
  }

  *h_results = (UL *) malloc(to * sizeof(UL));
  if (NULL == *h_results) {
    fprintf(stderr, "Error allocating temporary result storage!\n");
    free(tmp_results);
    return -1;
  }
  memcpy(*h_results, tmp_results, to * sizeof(UL));

  free(tmp_results);

  return to;
}
