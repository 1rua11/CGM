source activate zbf_pytorch171
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
