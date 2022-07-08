/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

    Modified by Ang Li, Jul 2022

**/

#ifndef DEPTH_METHOD_H_
#define DEPTH_METHOD_H_

#include <stdint.h>
#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "cost_aggregation.h"

template <class T>
class Mat2d {
public:
    Mat2d(size_t rows, size_t cols, T *data) {
		m_rows = rows;
		m_cols = cols;
        m_data = data;
    }
	T *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows, m_cols;
    T *m_data;
};

void init_depth_method(const uint8_t _p1, const uint8_t _p2);
Mat2d<uint8_t> compute_depth_method(Mat2d<uint8_t> left, Mat2d<uint8_t> right);
void finish_depth_method();
static void free_memory();

#endif /* DEPTH_METHOD_H_ */
