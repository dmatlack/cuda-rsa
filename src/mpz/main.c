/**
 * @file main.c
 *
 * Test harness for mpz code.
 *
 * @author David Matlack (dmatlack)
 */
#include "mpz.h"

#include <stdio.h>
#include <stdlib.h>

void test_add(char *str_sum, char *str_op1, char *str_op2);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  // test_add(a, b, c) checks that a == b + c
  
  test_add("88888888888888888888888888",
           "22222222222222222222222222", "66666666666666666666666666");

  test_add("123492005954673966549998101013",
           "273492736498276349827364", "123491732461937468273648273649");

  test_add("99987987987988000307227123526",
           "99987987987987987987987999799", "12319239123727");

  return 0;
}

void test_add(char *str_sum, char *str_op1, char *str_op2) {
  mpz_t expected;
  mpz_t actual;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&expected);
  mpz_init(&actual);
  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&expected, str_sum);
  mpz_set_str(&op1, str_op1);
  mpz_set_str(&op2, str_op2);

  mpz_add(&actual, &op1, &op2);

  printf("Testing %s + %s ... ", str_op1, str_op2);
  if (mpz_equal(&actual, &expected)) {
    printf("Passed Test!\n");
  }
  else {
    char *str_actual;

    printf("Failed! \n");

    str_actual = mpz_get_str(&actual);

    printf("\tExpected %s\n", str_sum);
    printf("\tGot      %s\n", str_actual);

    free(str_actual);
  }

  mpz_destroy(&expected);
  mpz_destroy(&actual);
  mpz_destroy(&op1);
  mpz_destroy(&op2);
}

