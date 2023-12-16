#!/bin/bash

echo "== start setup =="

# APP_PATHをPYTHONPATHに追加
export PYTHONPATH=$PYTHONPATH:$(pwd)
# export PYTHONPATH=$PYTHONPATH:$(dirname $(pwd))

echo "== end setup =="
