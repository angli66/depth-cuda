#!/usr/bin/env bash

mkdir build
cd build
cmake ..
make
cp engine.so ../dpcuda
cd ..
rm -rf build