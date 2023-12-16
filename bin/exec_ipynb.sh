#!/bin/bash

FILE_NAME=${1}
NUM_EPOCH=${2}

# APP_PATH="$(dirname $(pwd))"
APP_PATH=$(pwd)

echo $APP_PATH

echo "== start sh =="

if [ -z $NUM_EPOCH ]; then

  echo "exec 'jupyter nbconvert --to notebook --execute $APP_PATH/app/${FILE_NAME}.ipynb --output $APP_PATH/result/${FILE_NAME}.nbconvert.ipynb'"

  jupyter nbconvert --to notebook --execute $APP_PATH/app/${FILE_NAME}.ipynb --output $APP_PATH/result/${FILE_NAME}.nbconvert.ipynb

else

  echo "exec 'jupyter nbconvert --to notebook --execute $APP_PATH/app/${FILE_NAME}.ipynb --output $APP_PATH/result/${FILE_NAME}_epoch_${NUM_EPOCH}.nbconvert.ipynb'"

  jupyter nbconvert --to notebook --execute $APP_PATH/app/${FILE_NAME}.ipynb --output $APP_PATH/result/${FILE_NAME}_epoch_${NUM_EPOCH}.nbconvert.ipynb
fi

echo "== end sh =="
