#!/bin/bash
echo "Running data generation with hyperparameters from file: $1"

uv run --env-file "$1" -- python ./GenTrainData.py
# uv run --env-file "$1" -- python ./hello_world.py