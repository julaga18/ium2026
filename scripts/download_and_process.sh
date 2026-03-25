#!/bin/bash
CUTOFF=$1
mkdir -p data

seq 1 100 > data/raw_dataset.txt
echo "Pobrano zbiór danych do data/raw_dataset.txt"

shuf data/raw_dataset.txt | head -n $CUTOFF > data/subset.txt
echo "Utworzono podzbiór $CUTOFF losowych elementów w data/subset.txt"

head -n $CUTOFF data/subset.txt > data/final_dataset.txt
echo "Utworzono finalny zbiór danych z $CUTOFF przykładami w data/final_dataset.txt"