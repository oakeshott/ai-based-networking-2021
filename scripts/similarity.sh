#!/usr/bin/env bash

set -Ceuo pipefail

function error_handler() {
  set +x
  echo "something went wrong" >&2
  exit 1
}

: "start" && {
  echo "start preprocessing..."
  trap error_handler ERR
}


: "similarity measures" && {
  SOURCE_PATH=./img/dataset/
  RESDIR=similarity_measures/train
  python src/similarity.py --original ${SOURCE_PATH}/original --received ${SOURCE_PATH}/received/ -o ${RESDIR} --n-jobs 10 --grayscale
}

: "similarity measures" && {
  SOURCE_PATH=./img/issue/
  RESDIR=similarity_measures/test
  for i in `seq 0 4`
  do
    python src/similarity.py --original ${SOURCE_PATH}/original/0-4 --received ${SOURCE_PATH}/received/${i} -o ${RESDIR} --test-data --grayscale
  done
  for i in `seq 5 9`
  do
    python src/similarity.py --original ${SOURCE_PATH}/original/5-9 --received ${SOURCE_PATH}/received/${i} -o ${RESDIR} --test-data --grayscale
  done
}

: "done" && {
  set +x
  echo "successful"
}
