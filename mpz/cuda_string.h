/**
 * @file cuda_string.h
 *
 * @brief String functions to call from cuda code.
 */
#ifndef __418_CUDA_STRING_H__
#define __418_CUDA_STRING_H__

inline __device__ __host__ int cuda_strlen(const char *str) {
  int len = 0;

  while (str[len] != (char) 0) len++;

  return len;
}

#endif /* __418_CUDA_STRING_H__ */
