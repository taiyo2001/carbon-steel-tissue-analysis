#!/bin/bash

CURRENT_DIR=$(pwd)
APP_NAME=$(basename "$CURRENT_DIR")

# echo "アプリ名: $APP_NAME"

if ! docker image inspect "$APP_NAME" &> /dev/null; then
    echo "Building Docker image..."
    docker build -t $APP_NAME .
fi

echo "Running Docker container..."
docker run -it --rm --gpus all -v $(pwd):/app --name $APP_NAME
