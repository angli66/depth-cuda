/**
 * By Ang Li, Jul 2022
 */

#ifndef COST_AGGREGATION_GENERAL_H_
#define COST_AGGREGATION_GENERAL_H_

#include <stdint.h>

__global__
void CostAggregationKernelUpToDownGeneral(uint8_t* d_cost, uint8_t *d_L,
                                            const int P1, const int P2,
                                            const int rows, const int cols, const int maxDisp);

#endif /* COST_AGGREGATION_GENERAL_H_ */
