#/bin/bash


for req in $(cat requirements.txt); do
    pip install ${req};
done

cd BreakoutDetection/Python

swig -python -c++ breakout_detection.i
python setup.py build_ext -I../src build
python setup.py build_ext -I../src install

cd ../..