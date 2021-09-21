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
  set -x
}

: "ffprobe original video" {
  RESDIR=probe_info/dataset/original
  if [ ! -d $RESDIR ] ; then
    mkdir -p $RESDIR
  fi
  files=`find dataset_and_issue/dataset/original -type f -name "*.mp4"`
  for f in $files
  do
    basename=`echo $f | awk -F/ '{print $4}' | awk -F. '{print $1}'`
    ffprobe -show_frames -select_streams v -hide_banner -print_format json ${f} > ${RESDIR}/${basename}.json
  done
}

: "ffprobe received video" {
  RESDIR=probe_info/dataset/received
  if [ ! -d $RESDIR ] ; then
    mkdir -p $RESDIR
  fi
  files=`find dataset_and_issue/dataset/received -type f -name "*.mp4"`
  for f in $files
  do
    basename=`echo $f | awk -F/ '{print $4$5$6}' | awk -F'.' '{print $1}'`
    ffprobe -show_frames -select_streams v -hide_banner -print_format json ${f} > ${RESDIR}/${basename}.json
  done
}

: "done" && {
  set +x
  echo "successful"
}
