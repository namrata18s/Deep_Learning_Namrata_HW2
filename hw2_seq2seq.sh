#!/bin/bash
DATA_DIR=$1
OUTPUT_FILE=$2

python3 ./model_eq2seq.py "$DATA_DIR" "$OUTPUT_FILE"
