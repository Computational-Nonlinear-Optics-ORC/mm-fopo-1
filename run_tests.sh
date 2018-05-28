#!/bin/bash

cd src/cython_files
rm -rf build *so cython_integrand.c *html
python setup.py build_ext --inplace
cd ../..

pytest testing/*.py
