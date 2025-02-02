#!/bin/bash

# Copyright 2021-2025 National Institute of Informatics (authors: Lin Zhang)
# Author: Lin Zhang
# Initial version Date: 2021/03/01
# Updated on 2025/01 for release, using ASVspoof2019 LA train as a toy example.
#
# This script demonstrates how to run vad, segmentation and concatenate
# For creating database to explore Partial Spoof scenario

stage=$1
set -x
#set -e 
. ../env.sh


###############################################################################
# Configuration for directories and tools 
###############################################################################
# 1. Setup directory
if [[ $(hostname -f) == *smg.nii.ac.jp  ]]; then  #nii 
   HOME_DIR=/home/smg/zhanglin
elif [[ $(hostname -f) == *fit.vutbr.cz  ]]; then  #debug in but 
   HOME_DIR=/mnt/matylda4/qzhang
else                                              #tsubame
   HOME_DIR=/gs/hs0/tgh-20IAA/zhanglin
fi
MAIN_DIR=${HOME_DIR}/workspace/03PS/partialspoof-audio/egs/toy_example

# 2. setup path to installed kaldi and sv56
#    kaldi is used to provide some function like wav-to-duration
#    sv56 is used for normalization  
# kaldi
KALDI_PATH=${HOME_DIR}/software/kaldi
# SV56_PATH=${HOME_DIR}/software/sv56demo-master/src/sv56demo
SV56_PATH=${HOME_DIR}/software/STL-2009/src/sv56/sv56demo

# 3. Setup subset (sub-folder) to process
# 4. Setup source dir (SOURCEWAV_DIR) and export dir (EXP_DIR).
MODE="train"
TYPE="bonafide spoof"
SOURCEDATA_DIR=${MAIN_DIR}/source_data # Prepare needed source data files.
SOURCEWAV_DIR=${MAIN_DIR}/source_wav_norm   # Save normalized wav to this folder.
EXP_DIR=exp_toy_v0

###############################################################################
# Stage 0. Preparing normalized source wav.
#          Convert flac to wav and normalize the wav
###############################################################################
# Before running, we need to prepared ${SOURCEWAV_DIR} for info of source wav: 
# make sure you have files: flac.scp, . 
if [ $stage -le 0 ]; then
    for dset in ${MODE}; do

        FLAC_SCP=${SOURCEDATA_DIR}/${dset}/flac.scp 
        WAV_PATH=${SOURCEWAV_DIR}/${dset}/ 

        if [ -d ${WAV_PATH}  ]; then
            rm -rf ${WAV_PATH}
        fi
        mkdir -p ${WAV_PATH}
        if [ ! -d log  ]; then
            mkdir log
        fi

        # 2. convert flac to wav, and nomalize the wav.
        ./local/flac_to_wav.sh $FLAC_SCP $WAV_PATH ${SV56_PATH} 
        
        # Prepare wav.scp
        awk -vpath=${WAV_PATH} '{print $1" "path$1".wav"}' ${FLAC_SCP} \
		> ${SOURCEDATA_DIR}/${dset}/wav.scp
	## Alternative way to prepare wav.scp:
        # find ${WAV_PATH}/*.wav > data/${dset}/wav.list 
        # cat data/${dset}/wav.list | xargs -i basename {} .wav |\
	#     paste -d' ' - data/${dset}/wav.list > data/${dset}/wav.scp 
        # rm -rf data/${dset}/wav.list
        echo "Finish preparing source wav for "${SOURCEDATA_DIR}"/"${dset}
    done
# Finally, we will get: source_wav_norm, wav.scp
fi || exit 1


###############################################################################
# Stage 1. Find homegenous speech region using VAD/ASR.
###############################################################################
# Need to prepare: 
# 1) data/wav.scp in <uttid> <path>
# 2) data/label/label2num in <label_name> <label_id> 
# 3) protocols to specify each sample belings to which class
# Configuration
MIN_SIL_MS=0     # Treat nonspeech part as speech if it is shorter than
MIN_SEG_MS=0     # Treat a speech segment as nonspeech if it is shorter than 
SHIFT_STEP_MS=10 # 
STANDARDVAD_NAME=vad-stand-ms-sil${MIN_SIL_MS}-seg${MIN_SEG_MS}
#
if [ $stage -le 1 ]; then
    for dset in ${MODE}; do
	LABEL2NUM=${SOURCEDATA_DIR}/label/label2num_mul

        # 1. Calculating VAD, Currently we support kaldi_based VAD, rms VAD, pyannote
	# 1) Use some pretrained models to extract VAD
        python local/ 
        # TODO: Please refer to pyannote to get pretrained vad model.
	# 1. 
	#
	#
        # Do VAD and get vad.scp
	# zlnpy is similar to kaldi's style but in python
	# multi can implement kaldi, rms, pyannote
        # output: vad_dict.npy
        python partialspoof/utils/cal_vad.py \
		--WAV_SCP ${SOURCEDATA_DIR}/${dset}/wav.scp \
	       	--VAD_DIR ${SOURCEDATA_DIR}/${dset}/vad \
		--VAD_TYPE multi \
	      	--pyannote_pretrain_vad ${SOURCEDATA_DIR}/${dset}/vad/pyannote_sad_dihard.npy || exit 1
	
#        # Modified the 5th column: Replace - as bonafide for bonafide samples
#        PROTOCOL=${SOURCEWAV_DIR}/protocols/ASVspoof2019.LA.cm.${dset}.trn.txt
#        NEW_PROTOCOL=${SOURCEWAV_DIR}/protocols/zlASVspoof2019.LA.cm.${dset}.trn.txt
#        awk '{if($5=="bonafide"){print $1" "$2" "$3" bonafide "$5}
#	      else print $0}' ${PROTOCOL} > ${NEW_PROTOCOL}

#    and convert it to standard.
        # Convert the npy version into <start-time> <end-time> <label> in ms for each wav.
#        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}
#        if [ -d ${STANDARDVAD_DIR}  ]; then
#            rm -rf ${STANDARDVAD_DIR}
#        fi
#        mkdir ${STANDARDVAD_DIR}/utt-vad
#        python  ./local/get_standard_vad.py --TYPE "multi"\
#                                           --VAD_FILE data/${dset}/vad/vad_dict.npy \
#                                           --VAD_SAVE_DIR ${STANDARDVAD_DIR}/utt-vad \
#                                           --LABEL2NUM ${LABEL2NUM}  \
#                                           --WAV2TYPE ${NEW_PROTOCOL} \
#					   --RECO2DUR data/${dset}/reco2dur \
#	                                   --shift_step_ms ${SHIFT_STEP_MS} --min_sil_ms ${MIN_SIL_MS} --min_seg_ms ${MIN_SEG_MS}  || exit 1
#
#        #create vad.scp for standard vad
#        awk -vstvadpath=${STANDARDVAD_DIR}/utt-vad/ '{print $1, stvadpath$1".vad" }' data/${dset}/wav.scp > ${STANDARDVAD_DIR}/vad.scp
#        sort -n ${STANDARDVAD_DIR}/vad.scp -o ${STANDARDVAD_DIR}/vad.scp
#      
#	#generate rttm for original wav/
#	#we will use this to analysis later
#        ./local/vad2rttm_ms.sh ${STANDARDVAD_DIR} data/label/num2label_all       
      done
fi || exit 1
exit 1

#############
##2.generate spoof/genuine pool for each speaker
##also utt2spk, type_vad.scp, type_wav.scp 
############
if [ $stage -le 2 ]; then
    for dset in ${MODE}; do

        PROTOCOL=${SOURCEWAV_DIR}/protocols/ASVspoof2019.LA.cm.${dset}.trl.txt
        NEW_PROTOCOL=${SOURCEWAV_DIR}/protocols/zlASVspoof2019.LA.cm.${dset}.trl.txt
        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}

        #utt2spk
        cut -d' ' -f2 ${PROTOCOL} | paste -d' ' - ${PROTOCOL} | cut -d' ' -f1,2 - > data/${dset}/utt2spk
	#spk2utt
	${KALDI_PATH}/egs/wsj/s5/utils/utt2spk_to_spk2utt.pl data/${dset}/utt2spk > data/${dset}/spk2utt

        for type in ${TYPE}; do
            #create vad file and its scp in person level
            PERSON_VAD_DIR=${STANDARDVAD_DIR}/${type}-person-vad/  # Output directory for personvad
            PERSON_VAD_SCP=${STANDARDVAD_DIR}/${type}-person-vad.scp

            if [ -d ${PERSON_VAD_DIR}  ]; then
                rm -rf ${PERSON_VAD_DIR}
            fi
            mkdir -p ${PERSON_VAD_DIR}

            rm -rf ${PERSON_VAD_SCP}
            
            awk  -vtype=${type} -vvadpath=${STANDARDVAD_DIR}/utt-vad/ -vperson_vad_dir=${PERSON_VAD_DIR}/ -vperson_vad_scp=${PERSON_VAD_SCP} '{
                if ( $5==type ){
                    cmd="cat "vadpath$2".vad | sed \047s/^/" $2 " /g\047 >> "person_vad_dir$1".personvad"; system(cmd);
            }} ' ${NEW_PROTOCOL}

            find ${PERSON_VAD_DIR}*.personvad | xargs -i basename {} .personvad | awk -vpath=${PERSON_VAD_DIR}/ '{dir=path$1".personvad";print $1" "dir}' - > ${PERSON_VAD_SCP}
        
            awk -vtype=${type} '(NR==FNR){TYPE[$2]=$5}(NR!=FNR){if(TYPE[$1]==type){ print} }' ${NEW_PROTOCOL} data/${dset}/wav.scp  > data/${dset}/${type}_wav.scp
            awk -vtype=${type} '(NR==FNR){TYPE[$2]=$5}(NR!=FNR){if(TYPE[$1]==type){ print} }' ${NEW_PROTOCOL} ${STANDARDVAD_DIR}/vad.scp  > ${STANDARDVAD_DIR}/${type}_vad.scp
            sort -n data/${dset}/${type}_wav.scp -o data/${dset}/${type}_wav.scp
            sort -n ${STANDARDVAD_DIR}/${type}_vad.scp -o ${STANDARDVAD_DIR}/${type}_vad.scp
        done

    done
    echo "finished prepare vad for person and utt"
fi

#
#
#
########################################
##3. concatenate
##concatenate wav in CON_WAV_DIR
##and generate wav.scp, rttm, sysdur statisticf in CON_DATA_DIR
########################################
ALLOW_REUSE=False
if [ $stage -le 3 ]; then

    for dset in ${MODE}; do
        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}

	PREFIX_NAME=`tr '[a-z]' '[A-Z]' <<< CON_${dset:0:1}`

        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        LOG_DIR=${EXP_DIR}/${dset}_log
        echo "start to concatenate wav in "${CON_WAV_DIR}
        
        for file in ${CON_WAV_DIR} ${CON_DATA_DIR} ${LOG_DIR} ; do
            if [ -d ${file}  ]; then
                rm -rf ${file}
            fi
            mkdir -p ${file}
        done
        
        con_count=0
        
        #genertae concatenate wavfile in ${CON_WAV_DIR} 
        #and return the total number of concatenate.
        #and generate correspondding vad file in ${CON_WAV_DIR}


####To keep the ratio, we do not use spf to gen 
#NO, let's try 2 direction.
        python local/main_concatenate.py  \
            --ori_vad_scp ${STANDARDVAD_DIR}/bonafide_vad.scp \
            --insf_spk_vad_scp ${STANDARDVAD_DIR}/spoof-person-vad.scp \
            --utt2spk data/${dset}/utt2spk \
            --concatenate_wav_dir ${CON_WAV_DIR} \
            --wav_scp data/${dset}/wav.scp \
            --insert_log ${LOG_DIR}/${dset}_spf2gen_insert.log \
            --overlap_rate 0.5 \
            --counter ${con_count} \
	    --prefix_name ${PREFIX_NAME}\
	    --similar_length True\
	    --must_insert True\
	    --insert_label_type spf2gen 
	   

         #+++check the number between original number(ori_num) and generated number(wav_num)
	cd ${CON_WAV_DIR}
	ls CON*.wav > ${MAIN_DIR}/${CON_DATA_DIR}/list
	cd ${MAIN_DIR}
        wav_num=$(< ${CON_DATA_DIR}/list wc -l)
        rm -rf ${CON_DATA_DIR}/list
        ori_num=$(< ${STANDARDVAD_DIR}/bonafide_vad.scp wc -l)

#        echo "1/2 in "${CON_WAV_DIR}","
        echo "Finished generate "${wav_num}" wav by insert spoof to "${ori_num}" bonafide" 
        if [ ${wav_num} != ${ori_num} ]; then
            echo "WARNING: Need to Check, number != ori_num" 
        fi
        #++++finish check
####To keep the ratio, we do not use spf to gen 
#NO, let's try 2 direction.

#        wav_num=0
        python local/main_concatenate.py  \
            --ori_vad_scp ${STANDARDVAD_DIR}/spoof_vad.scp \
            --insf_spk_vad_scp ${STANDARDVAD_DIR}/bonafide-person-vad.scp \
            --utt2spk data/${dset}/utt2spk \
            --concatenate_wav_dir ${CON_WAV_DIR} \
            --wav_scp data/${dset}/wav.scp \
            --insert_log ${LOG_DIR}/${dset}_gen2spf_insert.log \
            --overlap_rate 0.5 \
            --counter ${wav_num} \
	    --prefix_name ${PREFIX_NAME}\
	    --similar_length True\
	    --must_insert False\
	    --insert_label_type gen2spf 
	    #--rand_seed 21\
        
        #+++check the number between original number(ori_num) and generated number(wav_num)
	cd ${CON_WAV_DIR}
	ls CON*.wav >  ${MAIN_DIR}/${CON_DATA_DIR}/wav.list
	cd ${MAIN_DIR}
        wav_num_tol=$(< ${CON_DATA_DIR}/wav.list wc -l)
        wav_num2=$[wav_num_tol - wav_num]
        ori_num2=$(< ${STANDARDVAD_DIR}/spoof_vad.scp wc -l)
#
#        echo "2/2 in "${CON_WAV_DIR}","
        echo "Finished generate "${wav_num2}" wav by insert bonafide to "${ori_num2}" spoof" 
        if [ ${wav_num2} != ${ori_num2} ]; then
            echo "WARNING: Need to Check, number != ori_num" 
        fi
#        #+++finish check
        echo "Total "${wav_num_tol}" generated in "${CON_WAV_DIR}":"
#        echo ${wav_num}" wav by insert spoof to "${ori_num}" bonafide," 
        echo ${wav_num2}" wav by insert bonafide to "${ori_num2}" spoof." 
        echo "--------------------------------"
#        
        
        #generate wav.scp 
        sed 's/\.wav$//g' ${CON_DATA_DIR}/wav.list | awk -vpath=${CON_WAV_DIR}/ '{; print $1, path$1".wav"}' - > ${CON_DATA_DIR}/wav.scp
        rm -rf ${CON_DATA_DIR}/wav.list 
        sort -n -k1 ${CON_DATA_DIR}/wav.scp -o ${CON_DATA_DIR}/wav.scp

##	#generate reco2dur
##	wav-to-duration scp:${CON_DATA_DIR}/wav.scp ark,t:${CON_DATA_DIR}/reco2dur
	${KALDI_PATH}/src/featbin/wav-to-duration scp:${CON_DATA_DIR}/wav.scp ark,t:${CON_DATA_DIR}/reco2dur

	#I forgot this, so we need to do this in by hand. TODO
	###wrong budui
	head -n wav_num ${CON_DATA_DIR}/reco2dur > ${CON_DATA_DIR}/reco2dur_spf2gen
	tail -n wav_num2 ${CON_DATA_DIR}/reco2dur > ${CON_DATA_DIR}/reco2dur_gen2spf
        #generate rttm
	#${CON_DATA_DIR} must contain wav.scp
	#*.wav located in the same place with *.vad
        ./local/vad2rttm.sh ${CON_DATA_DIR} data/label/num2label_all       

	if [ ! -d ${CON_DATA_DIR}/sil ]; then
		mkdir -p ${CON_DATA_DIR}/sil
		mkdir -p ${CON_DATA_DIR}/nosil
	fi
#
        sort -n -k2 -k3 ${CON_DATA_DIR}/rttm -o ${CON_DATA_DIR}/rttm
	cp ${CON_DATA_DIR}/rttm ${CON_DATA_DIR}/sil/rttm
        grep -v '> nonspeech <' ${CON_DATA_DIR}/sil/rttm > ${CON_DATA_DIR}/nosil/rttm 
	awk '(NR==FNR){NEW[$1]=$2}(NR!=FNR){$8=NEW[$8]; 
	printf "SPEAKER %s 1 %7.3f %7.3f <NA> <NA> %s <NA> <NA>\n", $2, $4, $5, $8 }' data/label/multi2bin_name ${CON_DATA_DIR}/sil/rttm > ${CON_DATA_DIR}/sil/rttm_bin
	awk '(NR==FNR){NEW[$1]=$2}(NR!=FNR){$8=NEW[$8]; 
	printf "SPEAKER %s 1 %7.3f %7.3f <NA> <NA> %s <NA> <NA>\n", $2, $4, $5, $8 }' data/label/multi2bin_name ${CON_DATA_DIR}/nosil/rttm > ${CON_DATA_DIR}/nosil/rttm_bin


   done
fi || exit

if [ $stage -eq 4 ]; then

    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create sysdur, segments for "${CON_DATA_DIR}
#############################
        cut -d' ' -f1 ${CON_DATA_DIR}/wav.scp > ${CON_DATA_DIR}/${dset}.lst


	#4.1 generate utt2num, utt2num_spf
	#remember sort -n
	#generate sysdur statistic
        if [ ! -d ${CON_DATA_DIR}/sysdur_stat ] ; then
            mkdir -p ${CON_DATA_DIR}/sysdur_stat
	fi
	python  local/rttm_to_reco2num.py\
	       	--rttm ${CON_DATA_DIR}/sil/rttm\
	       	--save-dir ${CON_DATA_DIR}/sysdur_stat
	sort -n -k1 ${CON_DATA_DIR}/sysdur_stat/reco2num_allsys -o ${CON_DATA_DIR}/sysdur_stat/reco2num_allsys
	sort -n -k1 ${CON_DATA_DIR}/sysdur_stat/reco2num_spfsys -o ${CON_DATA_DIR}/sysdur_stat/reco2num_spfsys
	sort -n -k1 ${CON_DATA_DIR}/sysdur_stat/reco2spfdur -o ${CON_DATA_DIR}/sysdur_stat/reco2spfdur
	ln -s ${CON_DATA_DIR}/sysdur_stat ${CON_DATA_DIR}/sysdur_stat_silsil


	cut -d' ' -f1 ${CON_LOG_DIR}/${dset}_gen2spf_insert.log | sort -u > ${CON_DATA_DIR}/gen2spf.lst
	cut -d' ' -f1 ${CON_LOG_DIR}/${dset}_spf2gen_insert.log | sort -u > ${CON_DATA_DIR}/spf2gen.lst


	#For spoof detection task.
	#TODO silsil / silspf
	#when we insert geniune to spoof
        #treat silence as spoof
        if [ ! -d ${CON_DATA_DIR}/sysdur_stat_silspf ] ; then
            mkdir -p ${CON_DATA_DIR}/sysdur_stat_silspf
	fi
	python local/rttm_to_reco2num_silspf.py\
		--list ${CON_DATA_DIR}/gen2spf.lst\
	       	--rttm ${CON_DATA_DIR}/sil/rttm\
	       	--save-dir ${CON_DATA_DIR}/sysdur_stat_silspf
	sort -n -k1 ${CON_DATA_DIR}/sysdur_stat_silspf/reco2num_allsys -o ${CON_DATA_DIR}/sysdur_stat_silspf/reco2num_allsys
	sort -n -k1 ${CON_DATA_DIR}/sysdur_stat_silspf/reco2num_spfsys -o ${CON_DATA_DIR}/sysdur_stat_silspf/reco2num_spfsys
	sort -n -k1 ${CON_DATA_DIR}/sysdur_stat_silspf/reco2spfdur -o ${CON_DATA_DIR}/sysdur_stat_silspf/reco2spfdur

#
#
##	python local/convert_rttm_to_utt2spk_and_segments.py\
##	       --rttm_file ${CON_DATA_DIR}/sil/rttm \
##               --utt2spk ${CON_DATA_DIR}/sil/utt2sys\
##               --segments ${CON_DATA_DIR}/sil/segments
##        
##	python local/convert_rttm_to_utt2spk_and_segments.py\
##	       --rttm_file ${CON_DATA_DIR}/nosil/rttm \
##               --utt2spk ${CON_DATA_DIR}/nosil/utt2sys\
##               --segments ${CON_DATA_DIR}/nosil/segments
##
##
done
fi

if [ $stage -eq 5 ]; then
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  


###TODO modify
###check duration first, then select
###        python ./local/slect_bal -- 
##                 --save-file ${CON_DATA_BAL_DIR}/
#
#please find select script in the 00note/202103-spfcon/*/check_concatenate (del_recoids)
	dir=/home/smg/zhanglin/workspace/00note/202103-spfcon/v2-2d-mustinsert/
	mkdir ${CON_DATA_BAL_DIR}
	cp ${dir}/${dset}_del_recoids ${CON_DATA_BAL_DIR}
#
#########prepare CON_DATA_BAL_DIR
        for file in ${dset}.lst reco2dur ;do
	    awk '(NR==FNR){DEL[$1]=1}
		 (NR!=FNR){if($1 in DEL){}else{print}
	    }' ${CON_DATA_BAL_DIR}/${dset}_del_recoids ${CON_DATA_DIR}/${file} > ${CON_DATA_BAL_DIR}/${file} 
	done

	for sysdur_type in sysdur_stat sysdur_stat_silspf; do
            if [ ! -d ${CON_DATA_BAL_DIR}/${sysdur_type} ] ; then
                mkdir -p ${CON_DATA_BAL_DIR}/${sysdur_type}
	    fi

            for file in reco2num_spfsys reco2spfdur ;do
	        awk '(NR==FNR){DEL[$1]=1}
	    	 (NR!=FNR){if($1 in DEL){}else{print}
	        }' ${CON_DATA_BAL_DIR}/${dset}_del_recoids ${CON_DATA_DIR}/${sysdur_type}/${file} > ${CON_DATA_BAL_DIR}/${sysdur_type}/${file} 
	    done
        done


#############For spoofing detection task
#different from 1-d(gen2spf)
##
#wav.lst
	cut -d' ' -f1 data/${dset}/bonafide_wav.scp > ${CON_DATA_BAL_DIR_PLUS}/${dset}.lst
        cat ${CON_DATA_BAL_DIR}/${dset}.lst >> ${CON_DATA_BAL_DIR_PLUS}/${dset}.lst
# wav
#need to copy, because xinwang's code need prepare wav in one dir
       ASVSPF_DIR=/home/smg/zhanglin/workspace/DATA/asvspoof19_normwav
       grep 'bonafide$' ${ASVSPF_DIR}/protocols/ASVspoof2019.LA.cm.${dset}.trl.txt | awk -F' ' -vspfwavdir=${ASVSPF_DIR}/${dset}/  -vconwavdir=${CON_WAV_DIR}/ '{cmd="cp "spfwavdir$2".wav "conwavdir; system(cmd)}' 
#### reco2dur
##       ###mkdir ${CON_DATA_BAL_DIR_PLUS}/old
##       ###mv ${CON_DATA_BAL_DIR_PLUS}/reco2dur* ${CON_DATA_BAL_DIR_PLUS}/old/ 
##
#	# generate reco2dur, bonafide_reco2dur, spoof_reco2dur
	cp ${CON_DATA_BAL_DIR}/reco2dur ${CON_DATA_BAL_DIR_PLUS}/reco2dur_spoof
	#we use the original bonafide, LA_*
	cp data/${dset}/reco2dur_bonafide ${CON_DATA_BAL_DIR_PLUS}/reco2dur_bonafide
	cp data/${dset}/reco2dur_bonafide ${CON_DATA_BAL_DIR_PLUS}/reco2dur
	cat ${CON_DATA_BAL_DIR}/reco2dur >> ${CON_DATA_BAL_DIR_PLUS}/reco2dur


	#################sysdur_stat
	for sysdur_type in sysdur_stat sysdur_stat_silspf; do
	#for sysdur_type in sysdur_stat ; do
            if [ ! -d ${CON_DATA_BAL_DIR_PLUS}/${sysdur_type} ] ; then
                mkdir -p ${CON_DATA_BAL_DIR_PLUS}/${sysdur_type}
            fi
    
            #reco2num_spfsys
            awk '{print $1" 0"}' data/${dset}/bonafide_wav.scp >  ${CON_DATA_BAL_DIR_PLUS}/${sysdur_type}/reco2num_spfsys
            cat ${CON_DATA_BAL_DIR}/${sysdur_type}/reco2num_spfsys >> ${CON_DATA_BAL_DIR_PLUS}/${sysdur_type}/reco2num_spfsys
            #reco2spfdur
            #'recoid','spoof_system_num','spfdur','ratio_spfall','ratio_genall'
            awk '{print $1" 0 0.0 0.0 1.0" }' data/${dset}/bonafide_wav.scp >  ${CON_DATA_BAL_DIR_PLUS}/${sysdur_type}/reco2spfdur
            cat ${CON_DATA_BAL_DIR}/${sysdur_type}/reco2spfdur >> ${CON_DATA_BAL_DIR_PLUS}/${sysdur_type}/reco2spfdur
        done

done
fi

if [ $stage -eq 6 ]; then
    for config_type in config_con config_con_silsil;do
        BASE=${MAIN_DIR}/../01asvspoof-silsil/${config_type}/
        cd ../${ASVSPF_DIR}/${config_type}

        ##if [ -d ${EXP_DIR}  ]; then
            rm -r ${EXP_DIR}
        ##fi
        ##echo ${EXP_DIR}
        ##mkdir -p ${EXP_DIR}
        
        for dset in ${MODE}; do
            x=config_con_test_on_${dset}.py    
            path=${MAIN_DIR}/${EXP_DIR}/    
            echo ${path}
            awk -vpath=$path -vexpname=${EXP_DIR} '{
                      if($0~/^CON_DATA_PATH/){print "CON_DATA_PATH = '\''"path"'\''"}
            	  else if($0~/^trn_set_name/){print "trn_set_name = '\''"expname"_trn'\''" }
            	  else if($0~/^val_set_name/){print "val_set_name = '\''"expname"_val'\''" }
            	  else if($0~/^test_set_name/){print "test_set_name = '\''"expname"_'\'' + set_type" }
            	  else{print} }' ${BASE}/${x} >  ${x}	    
            	  #else{print} }' ${BASE}/${x} >  ${EXP_DIR}/${x}	    
            sed -i 's/con_data_plus/con_data_bal_plus/g' ${x} 
        done
	#base/00_run.sh
	#sed -i 's/exp//g' ${x}
        #sed -i 's/con_data//g' model_con.py
	#nn_manage->zl, trn_lst  main.py


        cd ${MAIN_DIR}
    done
fi



SEG_LEN=1
SEG_OVERLAP_UNIT=0
SEG_UNIT_SEC=0.16


#con_data_bal_plus
if [ $stage -eq 66 ]; then
    set -e
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  

        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}
        #rttm
        awk '(NR==FNR){RECO[$1]=$2}
        (NR!=FNR){if($2 in RECO){print}}' data/${dset}/reco2dur_bonafide ${STANDARDVAD_DIR}/rttm > ${STANDARDVAD_DIR}/rttm_bonafide


        for file in rttm ;do
	        awk '(NR==FNR){DEL[$1]=1}
		     (NR!=FNR){if($2 in DEL){}else{print}
	        }' ${CON_DATA_BAL_DIR}/${dset}_del_recoids ${CON_DATA_DIR}/${file} > ${CON_DATA_BAL_DIR}/${file} 
    	done

#	    # generate rttm, bonafide_rttm, spoof_rttm
	    cp ${STANDARDVAD_DIR}/rttm_bonafide ${CON_DATA_BAL_DIR_PLUS}/rttm_bonafide
	    cp ${STANDARDVAD_DIR}/rttm_bonafide ${CON_DATA_BAL_DIR_PLUS}/rttm
	    cat ${CON_DATA_BAL_DIR}/rttm >> ${CON_DATA_BAL_DIR_PLUS}/rttm

         python   ./local/seglab_extraction.py \
                --rttm-file ${CON_DATA_BAL_DIR_PLUS}/rttm \
                --reco2dur ${CON_DATA_BAL_DIR_PLUS}/reco2dur \
                --label2num data/label/label2num_all \
                --seg-len ${SEG_LEN} --seg-overlap-unit ${SEG_OVERLAP_UNIT}  --seg-unit-sec ${SEG_UNIT_SEC} \
                --gen2spf-list ${CON_DATA_DIR}/gen2spf.lst \
                --seglab-save-file ${CON_DATA_BAL_DIR_PLUS}/seglab_${SEG_UNIT_SEC}
                #--gen2spf-list ${CON_DATA_BAL_PATH}/gen2spf.lst\
    done

fi

#con_data_bal_plus
#creat spf in label and spoof ratio label.
if [ $stage -eq 666 ]; then
MODE=$2 #"train dev eval"
    set -e
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        #CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/fuse_spf_balcon_data #con_data_bal_plus  

        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}


#SEG_LEN=1 #for diarization label, how many embedding to be one unit.
#SEG_OVERLAP_UNIT=0

         #for unit_sec in 0.64 0.32 0.16 0.08 0.04 0.02 0.01 ; do 
         for unit_sec in 0.01 ; do 
         SEG_UNIT_SEC=${unit_sec}
         python ./local/seglab_extraction_spfin.py \
                --rttm-file ${CON_DATA_BAL_DIR_PLUS}/rttm \
                --reco2dur ${CON_DATA_BAL_DIR_PLUS}/reco2dur \
                --label2num data/label/label2num_all \
                --seg-len ${SEG_LEN} --seg-overlap-unit ${SEG_OVERLAP_UNIT}  --seg-unit-sec ${SEG_UNIT_SEC} \
                --gen2spf-list ${CON_DATA_DIR}/gen2spf.lst \
                --seglab-save-file ${CON_DATA_BAL_DIR_PLUS}/seglab_${SEG_UNIT_SEC}_spfin
                #--gen2spf-list ${CON_DATA_BAL_PATH}/gen2spf.lst\
	done
    done
fi
#con_data_bal_plus
#creat spf in label and spoof ratio label.
#create label include silence.
if [ $stage -eq 777 ]; then
MODE=$2
    set -e
    for dset in ${MODE}; do
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        IN_DATA_DIR=${EXP_DIR}/${dset}/fuse_spf_balcon_data  

        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}


#SEG_LEN=1 #for diarization label, how many embedding to be one unit.
#SEG_OVERLAP_UNIT=0

         #for unit_sec in 0.16 0.08 0.04 0.02 0.01 ; do 
         for unit_sec in 0.01 ; do 
         SEG_UNIT_SEC=${unit_sec}
         python ./local/seglab_extraction_spfin_new.py \
                --rttm-file ${IN_DATA_DIR}/rttm \
                --reco2dur ${IN_DATA_DIR}/reco2dur \
                --label2num data/finallabel/multi2bin_name.bin \
                --seg-len ${SEG_LEN} --seg-overlap-unit ${SEG_OVERLAP_UNIT}  --seg-unit-sec ${SEG_UNIT_SEC} \
                --gen2spf-list ${CON_DATA_DIR}/gen2spf.lst \
		--silence 0 \
		--class_type bin \
                --seglab-save-file ${IN_DATA_DIR}/seglab_${SEG_UNIT_SEC}_spfin
                #--seglab-save-file ${IN_DATA_DIR}/seglab_${SEG_UNIT_SEC}_spfin_1sil
                #--gen2spf-list ${CON_DATA_BAL_PATH}/gen2spf.lst\
	done
    done
fi

#no last seg
if [ $stage -eq 662 ]; then
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  

        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}

         python ./local/seglab_extraction_spfin.py \
                --rttm-file ${CON_DATA_BAL_DIR_PLUS}/rttm \
                --reco2dur ${CON_DATA_BAL_DIR_PLUS}/reco2dur \
                --label2num data/label/label2num_all \
                --seg-len ${SEG_LEN} --seg-overlap-unit ${SEG_OVERLAP_UNIT}  --seg-unit-sec ${SEG_UNIT_SEC} \
                --gen2spf-list ${CON_DATA_DIR}/gen2spf.lst \
                --seglab-save-file ${CON_DATA_BAL_DIR_PLUS}/seglab_${SEG_UNIT_SEC}_skip_spfin \
	        --skip_lastseg True
                #--gen2spf-list ${CON_DATA_BAL_PATH}/gen2spf.lst\
    done
fi


#con_data_bal_plus
if [ $stage -eq 77 ]; then
    MODE="train dev eval"
    set -e
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  

        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}

#        for file in wav.scp ;do
#	        awk '(NR==FNR){DEL[$1]=1}
#		     (NR!=FNR){if($1 in DEL){}else{print}
#	        }' ${CON_DATA_BAL_DIR}/${dset}_del_recoids ${CON_DATA_DIR}/${file} > ${CON_DATA_BAL_DIR}/${file} 
#    	done
#
	    cp data/${dset}/bonafide_wav.scp ${CON_DATA_BAL_DIR_PLUS}/bonafide_wav.scp
	    cp data/${dset}/bonafide_wav.scp ${CON_DATA_BAL_DIR_PLUS}/wav.scp
	    cat ${CON_DATA_BAL_DIR}/wav.scp >> ${CON_DATA_BAL_DIR_PLUS}/wav.scp

#        #sed -i 's/ exp-2d-sametol/ \/mingback\/zhanglin\/workspace\/03i\/PROJ\/01spf-con\/00data-prepare\/exp-2d-sametol/g' ${CON_DATA_BAL_DIR_PLUS}/wav.scp
#        awk '{if($1~/LA/){print $1" ffmpeg -v 8 -i /data1/caiwch/asvspoof2019_posteval/LA/ASVspoof2019_LA_dev/flac/"$1".flac -f wav -ar 16000 -acodec pcm_s16le -|"}else{print}}' ${CON_DATA_BAL_DIR_PLUS}/wav.scp > ${CON_DATA_BAL_DIR_PLUS}/new.wav.scp

#utt2spk
#
#        awk '(NR==FNR){SPK[$1]=$2}(NR!=FNR){print $1,SPK[$2]}' data/${dset}/utt2spk ${EXP_DIR}/${dset}_log/${dset}_spf2gen_insert.log > ${CON_DATA_DIR}/utt2spk
#        awk '(NR==FNR){SPK[$1]=$2}(NR!=FNR){print $1,SPK[$2]}' data/${dset}/utt2spk ${EXP_DIR}/${dset}_log/${dset}_gen2spf_insert.log >> ${CON_DATA_DIR}/utt2spk
#        sort -u ${CON_DATA_DIR}/utt2spk -o ${CON_DATA_DIR}/utt2spk
#
#        for file in utt2spk ;do
#	        awk '(NR==FNR){DEL[$1]=1}
#		     (NR!=FNR){if($1 in DEL){}else{print}
#	        }' ${CON_DATA_BAL_DIR}/${dset}_del_recoids ${CON_DATA_DIR}/${file} > ${CON_DATA_BAL_DIR}/${file} 
#    	done
#
#        awk '(NR==FNR){RECO[$1]=$2}
#        (NR!=FNR){if($1 in RECO){print}}' data/${dset}/reco2dur_bonafide data/${dset}/utt2spk > data/${dset}/bonafide_utt2spk
#
#	    cp data/${dset}/bonafide_utt2spk ${CON_DATA_BAL_DIR_PLUS}/bonafide_utt2spk
#	    cp data/${dset}/bonafide_utt2spk ${CON_DATA_BAL_DIR_PLUS}/utt2spk
#	    cat ${CON_DATA_BAL_DIR}/utt2spk >> ${CON_DATA_BAL_DIR_PLUS}/utt2spk
#
    done

fi


if [ $stage -eq 99  ]; then
    MODE="train dev eval"
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create protocols for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  

        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}

	# CM protocols
	target_protocols_file=${EXP_DIR}/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.${dset}.trl.txt
        grep ' bonafide' ${EXP_DIR}/protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.${dset}.trl.txt > $target_protocols_file 
        awk '{print $2,$1,"- CON spoof"}' ${CON_DATA_BAL_DIR}/utt2spk >> $target_protocols_file 
	sort -n -k1 $target_protocols_file -o $target_protocols_file  


#	# ASV protocols
#	#need todo check
#        grep ' bonafide ' ${EXP_DIR}/protocols/ASVspoof2019.LA.asv.${dset}.gi.trl.txt > ${EXP_DIR}/protocols/PartialSpoof.LA.asv.${dset}.gi.trl.txt
#        #awk '{print $2,$1,"CON spoof"}' ${CON_DATA_BAL_DIR}/utt2spk >> ${EXP_DIR}/protocols/PartialSpoof.LA.asv.${dset}.gi.trl.txt
#        for gender in male female;do
#
#            awk '(NR==FNR){GEN[$1]=1}(NR!=FNR){if($1 in GEN){print}}' ${EXP_DIR}/protocols/ASVspoof2019.LA.asv.${dset}.${gender}.trl.txt ${EXP_DIR}/protocols/PartialSpoof.LA.asv.${dset}.gi.trl.txt > ${EXP_DIR}/protocols/PartialSpoof.LA.asv.${dset}.${gender}.trl.txt
#        done



        #after trail
        

    done

fi

#generate change point detection label.
if [ $stage -eq 100 ]; then
    set -e
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  

        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}

         python ./local/seglab_extraction_spfin_cpd.py \
                --seglab-file ${CON_DATA_BAL_DIR_PLUS}/seglab_${SEG_UNIT_SEC}_spfin.npy \
                --seglab-save-file ${CON_DATA_BAL_DIR_PLUS}/seglab_${SEG_UNIT_SEC}_spfin_cpd
                #--gen2spf-list ${CON_DATA_BAL_PATH}/gen2spf.lst\
    done
fi

# check dirty data
if [ $stage -eq 111  ]; then
MODE="eval"
    for dset in ${MODE}; do
        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        CON_LOG_DIR=${EXP_DIR}/${dset}_log/  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR=${EXP_DIR}/${dset}/con_data_bal  
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  

	OLD_DIR=${CON_DATA_BAL_DIR}


        python -m pdb ./local/check_data.py \
        --old_dir ${OLD_DIR} \
	--base_file sysdur_stat_silspf/reco2num_spfsys \
        --new_out_dir ${OLD_DIR}_new
    done
fi


#long utt
#should be moved to another script
if [ $stage -eq 8888 ]; then
MODE="train dev"
FIX_SPK_NUM=2
EXP_DIR=exp-${FIX_SPK_NUM}utt-dcls-dspk  #different class diff spk
EXP_DIR=exp-${FIX_SPK_NUM}utt-scls-dspk  #same class diff spk
#EXP_DIR=exp-utts-dcls-sspk  #deffent class same spk
#EXP_DIR=exp-utts-scls-sspk  #same class same spk

    for dset in ${MODE}; do
        STANDARDVAD_DIR=data/${dset}/${STANDARDVAD_NAME}

	PREFIX_NAME=`tr '[a-z]' '[A-Z]' <<< COU_${dset:0:1}`

        CON_WAV_DIR=${EXP_DIR}/${dset}/con_wav
        CON_DATA_DIR=${EXP_DIR}/${dset}/con_data  
        LOG_DIR=${EXP_DIR}/${dset}_log
        echo "start to concatenate wav in "${CON_WAV_DIR}
        
        for file in ${CON_WAV_DIR} ${CON_DATA_DIR} ${LOG_DIR} ; do
            #if [ -d ${file}  ]; then
            #    rm -rf ${file}
            #fi
            mkdir -p ${file}
        done
        
        con_count=0

        #same_cls True  
	python ./local/main_concatenate_long.py \
            --ori_vad_scp ${STANDARDVAD_DIR}/bonafide_vad.scp \
            --utt2spk data/${dset}/bonafide_utt2spk \
            --concatenate_wav_dir ${CON_WAV_DIR} \
            --wav_scp data/${dset}/wav.scp \
            --insert_log ${LOG_DIR}/${dset}_${EXP_DIR}.log \
            --counter ${con_count} \
	    --prefix_name ${PREFIX_NAME}\
	    --allow_reuse True \
            --same_spk False \
	    --fix_concate_num ${FIX_SPK_NUM} \
	    --allow_same_utt_in_session True
    done

fi

if [ $stage -eq 999 ]; then
EXP_DIR_OLD=exp-2d-sametol
EXP_DIR=exp-2d-sametol-trim03
MODE="train"
    set -e
    for dset in ${MODE}; do
        CON_DATA_DIR_OLD=${EXP_DIR_OLD}/${dset}/con_data  
        echo "start to create balance and sysdur, segments for "${CON_DATA_DIR}
        CON_DATA_BAL_DIR_PLUS=${EXP_DIR}/${dset}/con_data_bal_plus  


#already set 
#SEG_LEN=1 #for diarization label, how many embedding to be one unit.
#SEG_OVERLAP_UNIT=0

         #for unit_sec in 0.16 0.08 0.04 0.02 0.01 ; do 
         for unit_sec in 0.32 ; do 
         SEG_UNIT_SEC=${unit_sec}
         python3 ./local/seglab_extraction_spfin.py \
                --rttm-file ${CON_DATA_BAL_DIR_PLUS}/rttm \
                --reco2dur ${CON_DATA_BAL_DIR_PLUS}/reco2dur \
                --label2num data/label/label2num_all \
                --seg-len ${SEG_LEN} --seg-overlap-unit ${SEG_OVERLAP_UNIT}  --seg-unit-sec ${SEG_UNIT_SEC} \
                --gen2spf-list ${CON_DATA_DIR_OLD}/gen2spf.lst \
                --seglab-save-file ${CON_DATA_BAL_DIR_PLUS}/seglab_${SEG_UNIT_SEC}_spfin
	done
    done
fi


if [stage -eq 9999 ]; then
    bash ./local/create_uem.sh
fi



