#!/bin/bash

FILE_NAME=${1}
NUM_EPOCH=${2}

echo "== start sh =="

echo "exec 'jupyter nbconvert --to notebook --execute app/${FILE_NAME}.ipynb --output result/${FILE_NAME}_epoch_${NUM_EPOCH}.nbconvert.ipynb'"

jupyter nbconvert --to notebook --execute app/${FILE_NAME}.ipynb --output ../result/${FILE_NAME}_epoch_${NUM_EPOCH}.nbconvert.ipynb

echo "== end sh =="
