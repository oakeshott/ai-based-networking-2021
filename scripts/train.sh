#!/usr/bin/env bash

set -Ceuo pipefail

function error_handler() {
  set +x
  echo "something went wrong" >&2
  exit 1
}

: "start" && {
  echo "start..."
  trap error_handler ERR
}

: "train" && {
  echo "train..."
  INPUT_DIR=similarity_measures/train
  KFOLD=5
  python src/estimation.py -i ${INPUT_DIR} --train -l logs -o models -b 32 --log-seq 1 --fold ${KFOLD} --seed 1
}

: "done" && {
  set +x
  echo "successful"
}


