#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher none ${@:3}