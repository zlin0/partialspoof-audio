#!/usr/bin/env python

"""
To calculate vad for wav_scp
saved in vad_dir

Lin Zhang

"""


import scipy.io.wavfile as sciwav
import scipy.signal
import numpy as np
import argparse
import sys
import os
from tqdm import tqdm

#vad 
sys.path.append("/home/smg/zhanglin/workspace/00lab/project-NN-Pytorch-scripts.202102")
sys.path.append("/home/smg/zhanglin/workspace/00lab")
import torch
from psf_zl import get_energy
from core_scripts.data_io.wav_tools_202103 import silence_handler
from pyannote.audio.utils.signal import Binarize

#####Argument Parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--WAV_SCP',type=str, default='data/train/wav.scp')
parser.add_argument('--VAD_DIR',type=str, default='data/train/vad')
parser.add_argument('--VAD_TYPE',type=str, choices=['kaldi', 'rms', 'pyannote', 'multi'], default='multi')
parser.add_argument('--pyannote_pretrain_vad',type=str, default='')
args = parser.parse_args()

#data 
utt2wav =  dict([line.split() for line in open(args.WAV_SCP)])


def vec_padding(vector, len_limit):
    if(len(vector) < len_limit):
        return np.hstack((vector, np.zeros(len_limit - len(vector))))
    else:
        return vector[:len_limit]

#1. calculate energy for each frame
def cal_kaldi_vad_by_energy(log_energy, 
    vad_energy_mean_scale=0.5,
    vad_energy_th=5,
    vad_frames_context=0,
    vad_proportion_th=0.6):

    """
    http://kaldi-asr.org/doc/namespacekaldi.html#a451c729dd12ccb0707dd53c9768e572d

    vad_energy_mean_scale: :obj:`float`, optional
        If this is set to s, to get the actual threshold we let m be the mean
        log-energy of the file, and use s*m + vad-energy-th
    vad_energy_th: :obj:`float`, optional
        Constant term in energy threshold for MFCC0 for VAD.
    vad_frames_context: :obj:`int`, optional
        Number of frames of context on each side of central frame,
        in window for which energy is monitored 
    vad_proportion_th: :obj:`float`, optional
        Parameter controlling the proportion of frames within the window that
        need to have more energy than the threshold

    """
    energy_threshold = vad_energy_th
    energy_threshold += vad_energy_mean_scale * log_energy.sum() / len(log_energy)


#     for (int32 t2 = t - context; t2 <= t + context; t2++) {
    vad=[]
    T = log_energy.shape[0]
    for t in np.arange(len(log_energy)):
        num_count = 0
        den_count = 0
        context = vad_frames_context

        for t2 in np.arange(t - context, t + context + 1):
            if(t2 >=0 and t2 < T):
                den_count += 1
                if (log_energy[t2] > energy_threshold):
                    num_count += 1

        if (num_count >= den_count * vad_proportion_th):
            vad.append(1.0)
        else:
            vad.append(0.0)

    # low=0.5, high=5):
    # vad= np.where((energy> low) & (energy <high), 1, 0)

    return np.array(vad)

#def cal_vad_zero_energy():
#https://github.com/Zhangtingyuxuan/voice_activity_detection/blob/master/audio_split.py
# def cal_rms()

#def cal_npyvad_by_scp(wav_scp, vadtype, winlen=0.025, winstep=0.01, low=0.5, high=5): 
#    """
#    wav_scp: <wavid> <path>
#    vadtype: 'zlnpy'
#             'multi': now implement kaldi_vad, rms_vad, pyannote
#
#    """
#    #read WAV_SCP
#    #iterate to cal energy
#    if vadtype=="zlnpy":
#        energy_dict={}
#        vad_dict = {}
#        for uttid, uttdir in utt2wav.items():
#            sr, wav = sciwav.read(uttdir)
#            energy = get_energy(wav, samplerate=sr, winlen=0.025, winstep=0.01)
#            energy_dict[uttid] = energy
#            vad_dict[uttid] = cal_kaldi_vad_by_energy(energy)
#            
#    return vad_dict

def cal_npyvad_by_scp(wav_scp, vadtype, winlen=0.025, winstep=0.01, samplerate=16000, low=0.5, high=5,vote=2): 
    """
    wav_scp: <wavid> <path>
    vadtype: 'kaldi'
             'multi': now implement kaldi_vad, rms_vad, pyannote

    """
    #init
    kaldi_vad=False
    rms_vad=False
    pyannote_dih=False
    vote=1


    #init window:
    winlen_p = int(winlen * samplerate)
    winstep_p = int(winstep * samplerate)



    if vadtype=="kaldi":
        kaldi_vad=True
    elif vadtype=="pyannote":
        pyannote_dih=True
    elif (vadtype=="multi"):
        kaldi_vad=True
        rms_vad=True
        pyannote_dih=True
        assert vote >0

    if(kaldi_vad+rms_vad+pyannote_dih==3):
        vote=2

    #prepare for model:
    if(pyannote_dih):
        if(args.pyannote_pretrain_vad):
            pyannote_vad_dihard_label_pt=np.load(args.pyannote_pretrain_vad, allow_pickle=True).tolist()
        else:
            sad_dihard_model = torch.hub.load('pyannote/pyannote-audio', 'sad_dihard')
            binarize = Binarize(offset=0.5, onset=0.5, log_scale=True,
                                            min_duration_off=0.1, min_duration_on=0.1)

    vad_dict = {}
    for uttid, uttdir in tqdm(utt2wav.items()):
        vads_dict = {}
        sr, sig = sciwav.read(uttdir)
        assert sr == samplerate

        n_frames = int(sig.shape[0] / sr / winstep)
        vads_dict[uttid]=np.zeros(n_frames)
        vad_dict[uttid]=np.zeros(n_frames)
        if(kaldi_vad):
            energy_dict={}
            energy = get_energy(sig, samplerate=sr, winlen=winlen, winstep=winstep)
            energy_dict[uttid] = energy
            vad =cal_kaldi_vad_by_energy(energy)
            vads_dict[uttid]=np.sum((vads_dict[uttid], vec_padding(vad,n_frames)),axis=0)
        if(rms_vad):
            vad = silence_handler(sig, sr, fl=winlen_p, fs = winstep_p, shortest_len_in_ms=0) # shortest_len_in_ms will be define later
            vads_dict[uttid]=np.sum((vads_dict[uttid], vec_padding(vad[2],n_frames)),axis=0)
        if(pyannote_dih):
            if(args.pyannote_pretrain_vad):
                pyannote_vad_dihard_label = pyannote_vad_dihard_label_pt[uttid][:n_frames]
            else:
                sad_dihard_scores = sad_dihard_model({'uri': uttid, 'audio': uttdir})
                speech = binarize.apply(sad_dihard_scores, dimension=1) # timeline instance
                pyannote_vad_dihard_label = np.zeros(n_frames)
                for i, speech_region in enumerate(speech):
                   st, et = '%.3lf' % speech_region.start, '%.3lf' % speech_region.end
                   sf, ef = round(speech_region.start / winstep), round(speech_region.end / winstep)
                   pyannote_vad_dihard_label[sf:ef] = 1
            vads_dict[uttid]=np.sum((vads_dict[uttid], vec_padding(pyannote_vad_dihard_label,n_frames)),axis=0)
        #vote
        vad_dict[uttid][np.where(vads_dict[uttid]>=vote)] = 1


    return vad_dict

def main():

    vad_dict = cal_npyvad_by_scp(args.WAV_SCP, args.VAD_TYPE)
    if( not os.path.exists(args.VAD_DIR) ):
        os.makedirs(args.VAD_DIR)
    np.save(args.VAD_DIR+'/vad_dict.npy', vad_dict)
    # np.load('vad_dict.npy', allow_pickle=True)
def debug():
    WAV_SCP="data/train/wav.scp"
    VAD_TYPE="multi"
    vad_dict = cal_npyvad_by_scp(WAV_SCP, VAD_TYPE)

if __name__ == '__main__':
    main()
#    debug()
