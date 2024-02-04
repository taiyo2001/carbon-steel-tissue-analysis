#!/bin/bash

OPTION=${1}

echo "== start formatter =="

if [ $OPTION = "check" ]; then
  echo "Running in check mode. Checking formatting..."
  black --check ./
else
  echo "Running in normal mode. Formatting files..."
  black ./
fi

echo "== end formatter =="
