#ifndef PRIMEGEN_H
#define PRIMEGEN_H

#include "uint32.h"
#include "uint64.h"

#define PRIMEGEN_WORDS conf-words

typedef struct {
  uint32 buf[16][PRIMEGEN_WORDS];
  uint64 p[512]; /* p[num-1] ... p[0], in that order */
  int num;
  int pos; /* next entry to use in buf; WORDS to restart */
  uint64 base;
  uint64 L;
} primegen;

extern void primegen_sieve(primegen *);
extern void primegen_fill(primegen *);

extern void primegen_init(primegen *);
extern uint64 primegen_next(primegen *);
extern uint64 primegen_peek(primegen *);
extern uint64 primegen_count(primegen *,uint64 to);
extern void primegen_skipto(primegen *,uint64 to);

#endif
