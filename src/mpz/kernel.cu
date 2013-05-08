/**
 * @file kernel.cu
 *
 * @brief The CUDA kernel that will run.
 */
#include "kernel.h"

#include "mpz.h"    // multiple precision cuda code
#include "cuda_string.h"
#include <stdio.h>

char *devA;
char *devB;
char *devC;

/**
 * for i in 0...count:
 *    devC[i] = devA[i] + devB[i]
 */
__global__ void additionKernel(char *devA, char *devB, char *devC, unsigned count) {
  char str[STRING_MAX_SIZE];
  char *global_str;
  int threadId = threadIdx.x;
  int numThreads = blockDim.x;
  int index;
  mpz_t sum;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&op1);  
  mpz_init(&op2);  
  mpz_init(&sum);

  for (index = threadId; index < count; index += numThreads) {
    mpz_set_str(&op1, devA + (index * STRING_MAX_SIZE));
    mpz_set_str(&op2, devB + (index * STRING_MAX_SIZE));

    mpz_add(&sum, &op1, &op2);

    mpz_get_str(&sum, str, STRING_MAX_SIZE); 

    global_str = devC + (index * STRING_MAX_SIZE); 

    memcpy(global_str, str, cuda_strlen(str) + 1);
  }

  mpz_destroy(&sum);
  mpz_destroy(&op1);
  mpz_destroy(&op2);
}

void run_addition_kernel(char *A, char *B, char *C, unsigned num_strings) {
  size_t size = num_strings * STRING_MAX_SIZE;

  cudaMalloc(&devA, size);
  cudaMalloc(&devB, size);
  cudaMalloc(&devC, size);

  cudaMemset(&devA, 0, size);
  cudaMemset(&devB, 0, size);
  cudaMemset(&devC, 0, size);

  cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(devB, B, size, cudaMemcpyHostToDevice);

  additionKernel<<<1,32>>>(devA, devB, devC, num_strings);

  cudaMemcpy(C, devC, size, cudaMemcpyDeviceToHost);
}
