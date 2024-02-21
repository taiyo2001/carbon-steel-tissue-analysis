#!/bin/bash

CURRENT_DIR=$(pwd)
APP_NAME=$(basename "$CURRENT_DIR")

echo "setup $APP_NAME"

if [ ! $(docker image inspect -f '{{.Id}}' $APP_NAME 2>/dev/null) ]; then
    echo "Building Docker image..."
    docker build -t $APP_NAME .
fi

echo "Running Docker container..."
docker run -it --rm --gpus all -v $(pwd):/carbon-steel-tissue-analysis $APP_NAME
