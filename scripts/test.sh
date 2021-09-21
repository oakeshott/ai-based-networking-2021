#!/usr/bin/env bash

set -Ceuo pipefail

K=5
function error_handler() {
  set +x
  echo "something went wrong" >&2
  exit 1
}

: "start" && {
  echo "start..."
  trap error_handler ERR
}

: "test" && {
  echo "testing..."
  INPUT_DIR=similarity_measures/test/issue
  for i in `seq 0 9`
  do
    for k in `seq 1 ${K}`
    do
      python src/estimation.py -i ${INPUT_DIR}/similarity_${i}.json --model-path models/fold${k}_1000.mdl --test
    done
  done
}

: "done" && {
  set +x
  echo "successful"
}
