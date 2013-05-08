#include <math.h>
#include "primegen.h"

primegen pg;

void main()
{
  uint64 p;
  uint64 lastp;
  uint32 diff;
  uint32 maxdiff;

  primegen_init(&pg);

  primegen_next(&pg);
  lastp = primegen_next(&pg);
  maxdiff = 0;

  for (;;) {
    p = primegen_next(&pg);
    diff = p - lastp;
    if (diff > maxdiff) {
      printf("%.0f %.0f %f\n"
	,(double) p
	,(double) diff
	,log((double) diff)/log(log((double) p))
	);
      maxdiff = diff;
    }

    lastp = p;
  }
}
