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
  set -x
}

: "kill docker process" && {
  echo "kill docker process..."
  PID=`docker ps -a | grep ai-based-networking | awk '{print $1}'`
  docker rm ${PID} -f
}

: "kill docker image" && {
  echo "kill docker image..."
  PID=`docker images -a | grep ai-based-networking | awk '{print $3}'`
  docker rmi ${PID} -f
}

: "done" && {
  set +x
  echo "successful"
}

