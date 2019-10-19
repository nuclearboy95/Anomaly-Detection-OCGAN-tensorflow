#!/bin/bash

rm build/*
rm dist/*
python setup.py bdist_wheel
twine upload dist/ocgan-*.whl