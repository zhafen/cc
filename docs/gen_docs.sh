#!/bin/bash

# Generate docs using the docstrings
sphinx-apidoc -o ./source ../cc -e -f

# Make the files
make clean
make html
