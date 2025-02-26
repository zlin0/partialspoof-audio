#/bin/bash
#
# Extracting VAD prediction using ptannote pretrained VAD model.
#

nj=40
#sad_decode_stage=0
DATA_PATH=source_data/train
MODEL_NAME=sad_ami

. ./tools/parse_options.sh

##for dset in train dev eval; do
#for dset in train ; do
#
#DATA_PATH=../data/${dset}
#
#segmentation/detect_speech_activity.sh \
#  --nj $nj --stage $sad_decode_stage \
#  data/dihard3_${dset} exp/dihard3_sad_tdnn_stats \
#  mfcc exp/dihard3_sad_tdnn_stats_decode_${dset} \
#  data/dihard3_${dset}_seg
#done



#for dset in train dev eval; do
while true
do
stat2=$(gpustat | awk '{print $9}' | sed -n '2p')
echo $stat2
if [[ "${stat2}" -le 1000   ]]; then
    WAV_SCP=${DATA_PATH}/wav.scp
    OUT_NPY=${DATA_PATH}/vad/pyannote_${MODEL_NAME}

    if [[ ! -d ${DATA_PATH}/vad ]]; then
	    mkdir ${DATA_PATH}/vad
    fi

    python utils/pyannote_vad_infer.py \
    	--wav_scp ${WAV_SCP} --out_npy ${OUT_NPY} --gpu 0 \
           	--frame_shift 0.01 --model ${MODEL_NAME}     

#    OUT_NPY=${DATA_PATH}/vad/pyannote_sad_dihard
#    python ./vad/pyannote_vad_infer.py --wav_scp ${WAV_SCP} --out_npy ${OUT_NPY} --gpu 0\
#    	--frame_shift 0.01 --model sad_dihard     #--hubconf_dir ${PYANNOTE_PATH}
fi || exit 1
done

