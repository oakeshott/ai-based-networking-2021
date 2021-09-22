#!/usr/bin/env bash

set -Ceuo pipefail

KFOLD=4
function error_handler() {
  set +x
  echo "something went wrong" >&2
  exit 1
}

: "start" && {
  echo "start..."
  trap error_handler ERR
  if [ ! -d logs ]; then
    mkdir logs
  fi
  if [ -e logs/result.txt ]; then
    rm logs/result.txt
    touch result.txt
  fi
}

: "test" && {
  echo "testing..."
  INPUT_DIR=similarity_measures/test/issue
  for i in `seq 0 9`
  do
    for k in `seq 1 ${KFOLD}`
    do
      python src/estimation.py -i ${INPUT_DIR}/similarity_${i}.json --model-path models/fold${k}_1000.mdl --test
    done
  done
}

: "done" && {
  set +x
  echo "successful"
}
