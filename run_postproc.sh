#!/bin/bash
# Simple postprocessing code

cd src/cython_files
rm -rf build *so cython_integrand.c *html
cython -a cython_integrand.pyx
python setup.py build_ext --inplace
cd ../..
python src/Conversion_efficiency_post_proc.py 
