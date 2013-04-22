/**
 * @file cpu_main.cpp
 */
#include "mpz.h"
#include <string.h>
#include <stdio.h>

void test_count_digits(mpz_t *mpz, char *str) {
  unsigned count;

  printf("Testing mpz_count_digits(\"%s\") ... ", str);

  mpz_set_str(mpz, str);

  count = mpz_count_digits(mpz);

  if (count == strlen(str)) {
    printf("PASS\n");
  }
  else {
    printf("FAIL (Expected: %lu, Got: %u)\n", strlen(str), count);
  }

}

int main(int argc, char **argv) {
  mpz_t z;

  (void)argc;
  (void)argv;

  mpz_init(&z);

  test_count_digits(&z, "0");
  test_count_digits(&z, "2339487239847298374928734");
  test_count_digits(&z, "9999");
  test_count_digits(&z, "1000000000000");
  

  return 0;
}
