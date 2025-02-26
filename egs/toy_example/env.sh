#!/bin/bash
# if necessary, load conda environment
ROOT_PATH=`pwd`

export PS_PATH=$ROOT_PATH
export TOOL_PATH=${ROOT_PATH}/../../tools

export PYTHONPATH=${PS_PATH}:${ROOT_PATH}:$PYTHONPATH
export LC_ALL=C
export PYTHONUNBUFFERED=1
