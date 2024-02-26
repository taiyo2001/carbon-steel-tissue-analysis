#!/bin/bash

# NOTE: exec with source command. If you don't, you may get an error
echo "== start setup =="

CURRENT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
APP_PATH=$(dirname "$CURRENT_DIR")

if [ -e "$APP_PATH/bin/exec_ipynb.sh" ]; then
  chmod +x $APP_PATH/bin/exec_ipynb.sh
  alias ipynb="sh $APP_PATH/bin/exec_ipynb.sh"
  echo "exec sh exec_ipynb.sh at $APP_PATH/bin = ipynb [file_name] [epochs]"
else
  echo "Error: exec_ipynb.sh not found at $APP_PATH/bin"
  exit 1
fi

# set module import source
export PYTHONPATH=$PYTHONPATH:$APP_PATH

chmod +x $APP_PATH/bin/exec_ipynb.sh
alias ipynb="sh $APP_PATH/bin/exec_ipynb.sh"


if [ ! -d "$APP_PATH/tmp" ]; then
    mkdir "$APP_PATH/tmp"
fi

if [ ! -d "$APP_PATH/data" ]; then
    mkdir "$APP_PATH/data"
fi

if [ ! -d "$APP_PATH/result/services" ]; then
    mkdir "$APP_PATH/result/services"
fi

echo "== end setup =="
