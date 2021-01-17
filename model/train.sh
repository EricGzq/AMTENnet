#!/usr/bin/env sh
set -e

TOOLS=/home/caffegpu/caffe-master/build/tools
GLOG_log_dir='/home/caffegpu/Desktop/model/log/' \
$TOOLS/caffe train \
  --solver=/home/caffegpu/Desktop/model/solver_model.prototxt \
  --gpu=0,1
  

