#!/bin/bash

FILE_NAME=${1}
NUM_EPOCH=${2}

APP_PATH="$(dirname $(pwd))"

echo "== start sh =="

echo "exec 'jupyter nbconvert --to notebook --execute $APP_PATH/app/${FILE_NAME}.ipynb --output $APP_PATH/result/${FILE_NAME}_epoch_${NUM_EPOCH}.nbconvert.ipynb'"

jupyter nbconvert --to notebook --execute $APP_PATH/app/${FILE_NAME}.ipynb --output $APP_PATH/result/${FILE_NAME}_epoch_${NUM_EPOCH}.nbconvert.ipynb

echo "== end sh =="
