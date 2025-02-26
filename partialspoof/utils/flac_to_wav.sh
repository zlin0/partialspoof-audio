#!/bin/bash

# Copyright 2021 National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.
#
# Convert flac to wav, and normalize wavs by using sv56


set -e
set -x

# Convert FLAC files to WAV.
if [ -z `which sox` ]; then
    echo "SoX not installed. Please install SoX before proceeding. E.g.:"
    echo ""
    echo "    apt-get install sox"
fi
FLAC_SCP=$1
WAV_DIR=$2
SV56_PATH=$3

if [[ ! -d $WAV_DIR ]]; then
    mkdir $WAV_DIR
fi
#for flac_fn in `ls $FLAC_DIR/*.flac`; do
while IFS=' ' read -r uttid flac_fn; do
    bn=${flac_fn##*/}
    recid=${bn%%.*}
    wf=$WAV_DIR/${recid}.wav
    #sox $flac_fn $wf

    partialspoof/utils/norm_sv56.sh $flac_fn $wf ${SV56_PATH}

done < ${FLAC_SCP}
