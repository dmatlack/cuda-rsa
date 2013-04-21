/**
 * @file string.cu
 *
 * @brief String functions to call from cuda code.
 */

__device__ int cudaStrlen(char *str) {
  int len = 0;

  while (str[len] != (char) 0) len++;

  return len;
}
