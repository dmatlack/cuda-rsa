/*
B is 32 times X.
Total memory use for one generator is 2B bytes = 64X bytes.
Covers primes in an interval of length 1920X.
Working set size for one generator is B bits = 4X bytes.

Speedup by a factor of 2 or 3 for L1 cache instead of L2 cache.
Slowdown by a factor of roughly n for primes past (nB)^2.

Possible choices of X:
  2002 to fit inside an 8K L1 cache (e.g., Pentium).
  4004 to fit inside a 16K L1 cache (e.g., Pentium II).
  64064 to fit inside a 256K L2 cache.

There are various word-size limits on X; 1000000 should still be okay.
*/

#define B32 PRIMEGEN_WORDS
#define B (PRIMEGEN_WORDS * 32)
