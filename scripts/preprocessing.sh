#!/usr/bin/env bash

set -Ceuo pipefail

SOURCE=dataset_and_issue
TARGET=img
RESIZE=0.2

function error_handler() {
  set +x
  echo "something went wrong" >&2
  exit 1
}

: "start" && {
  echo "start preprocessing..."
  trap error_handler ERR
  set -x
}

: "preprocess original video data" && {
  SOURCE_PATH=${SOURCE}/dataset
  TARGET_PATH=${TARGET}/dataset
  python src/preprocessing.py -i ${SOURCE_PATH}/original -o ${TARGET_PATH}/original/ --resize ${RESIZE} -j 3  --grayscale
}

: "preprocessing received video data" && {
  SOURCE_PATH=${SOURCE}/dataset
  TARGET_PATH=${TARGET}/dataset
  python src/preprocessing.py -i ${SOURCE_PATH}/received -o ${TARGET_PATH}/received/ --resize ${RESIZE} -j 5 --grayscale
}

: "preprocess original video data" && {
  SOURCE_PATH=${SOURCE}/issue
  TARGET_PATH=${TARGET}/issue
  python src/preprocessing.py -i ${SOURCE_PATH}/original -o ${TARGET_PATH}/original/ --resize ${RESIZE} -j 1 --grayscale
}

: "preprocessing received video data" && {
  SOURCE_PATH=${SOURCE}/issue
  TARGET_PATH=${TARGET}/issue
  python src/preprocessing.py -i ${SOURCE_PATH}/received -o ${TARGET_PATH}/received/ --resize ${RESIZE} -j 2 --grayscale
}

: "done" && {
  set +x
  echo "successful"
}

