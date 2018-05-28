#!/bin/bash

source activate intel
rm -rf build
if [ ! -f ./build ]; then
    mkdir build
fi
source activate intel
LDSHARED="icc -shared" CC=icc python3.6 setup.py build_ext --inplace
