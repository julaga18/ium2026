#!/bin/bash
set -e

LIMIT=${1:-15}

head -n 1 data/diabetes.csv > data/dataset.csv
tail -n +2 data/diabetes.csv | shuf | head -n "$LIMIT" >> data/dataset.csv