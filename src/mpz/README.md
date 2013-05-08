MPZ
===

MPZ stands for Multiple Precision Integer (Z = integers in math).

The MPZ code is structured so that it can be compiled by `nvcc` for use in CUDA 
kernels, or by `g++` for use on the cpu.

To use the MPZ code, just `#include "mpz.h"` in your source file. Note that the
MPZ source code is distributed across `mpz.h`, `digit.h`, and `cuda_string.h`.

The available mpz functions are documented in mpz.h.

For examples on how to run on the GPU and CPU, see the Makefile.
