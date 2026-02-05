#!/bin/bash
echo "Running training with hyperparameters from file: $1"

uv run --env-file "$1" -- python ./TrainDiffusion.py