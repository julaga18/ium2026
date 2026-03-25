#!/bin/bash

mkdir -p data

seq 1 100 > data/raw_dataset.txt
echo "Pobrano zbiór danych do data/raw_dataset.txt"

shuf data/raw_dataset.txt | head -n 20 > data/subset.txt
echo "Utworzono podzbiór 20 losowych elementów w data/subset.txt"

head -n 10 data/subset.txt > data/final_dataset.txt
echo "Utworzono finalny zbiór danych w data/final_dataset.txt"