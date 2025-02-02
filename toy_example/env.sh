#!/bin/bash
# if necessary, load conda environment
ROOT_PATH=`pwd`

export PS_PATH=$ROOT_PATH
export PYTHONPATH=${PS_PATH}:$PYTHONPATH:$MODULES_PATH:${S3PRL_PATH}
export LC_ALL=C
export PYTHONUNBUFFERED=1
