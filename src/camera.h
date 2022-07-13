/**
 * By Ang Li, Jul 2022
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include <stdint.h>
#include <cmath>

__global__
void remap(const float *mapx, const float *mapy,
            const uint32_t rows, const uint32_t cols,
            const uint8_t *src, uint8_t *dst);

__global__
void disp2Depth(const uint8_t *disp, float *depth,
                    float focalLen, float baselineLen,
                    float minDepth, float maxDepth);

#endif /* CAMERA_H_ */