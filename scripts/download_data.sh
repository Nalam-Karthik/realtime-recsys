#!/usr/bin/env bash
# Fetch MovieLens ml-latest-small (100k ratings, 610 users, 9.7k movies) from GroupLens.
set -euo pipefail
mkdir -p data
curl -sSL -o data/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip -q -o data/ml-latest-small.zip -d data
rm data/ml-latest-small.zip
echo "data/ml-latest-small ready"
