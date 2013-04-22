/**
 * @file main.cpp
 */
#include "kernel.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NUM_STRINGS 4

// C[i] should equal A[i] + B[i]
char A[NUM_STRINGS][STRING_MAX_SIZE];
char B[NUM_STRINGS][STRING_MAX_SIZE];
char C[NUM_STRINGS][STRING_MAX_SIZE];

char correct[NUM_STRINGS][STRING_MAX_SIZE];

int main(int argc, char **argv) {
  int i;

  (void) argc;
  (void) argv;

  /************************************************/
  /* Addition Test Cases                          */
  /************************************************/

  strcpy(A[0],       "100");
  strcpy(B[0],       "200");  
  strcpy(correct[0], "300");

  strcpy(A[1],       "398459384593847593475938475");
  strcpy(B[1],       "71247619234761928374618923746");  
  strcpy(correct[1], "71646078619355775968094862221");

  strcpy(A[2],       "4444444444444444444444444444444444444444444444444444");
  strcpy(B[2],       "5555555555555555555555555555555555555555555555555555");  
  strcpy(correct[2], "9999999999999999999999999999999999999999999999999999");

  strcpy(A[3],       "22340872041208341098457029387450982734509872093458729034857");
  strcpy(B[3],       "2934785298374650283475092384750928374593874598");  
  strcpy(correct[3], "22340872041211275883755404037734457826894623021833322909455");

  run_addition_kernel(A[0], B[0], C[0], NUM_STRINGS);

  for (i = 0; i < NUM_STRINGS; i++) {
    if (!strcmp(correct[i], C[i])) {
      printf(".");
    }
    else {
      printf("\nFAIL ");
      printf("%s + %s = ", A[i], B[i]);
      printf("[Expected: %s, ", correct[i]);
      printf("Got: %s]\n", C[i]);
    }
  }

  printf("\n");
  return 0;
}
