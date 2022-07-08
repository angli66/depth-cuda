#!/usr/bin/env bash

mkdir build
cd build
cmake ../src
make
cp engine.so ../dpcuda
cd ..
rm -rf build