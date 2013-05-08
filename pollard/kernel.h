#ifndef __KERNEL_H__
#define __KERNEL_H__

#define RESULTS_PER_THREAD 2

#define UL unsigned long

int factorize(UL n, unsigned *table, UL **h_results);

#endif /* __KERNEL_H__ */
