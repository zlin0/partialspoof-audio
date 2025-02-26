#!/usr/bin/env python
# Copyright 2021 National Institute of Informatics (authors: Lin Zhang, zlin@ieee.org)

"""
To calculate vad for a wav or wav_scp
Supporting: 
(1) window shift log_energy_based_VAD from kaldi.
(2) VAD Sec.5.1 of https://www.sciencedirect.com/science/article/pii/S0167639309001289

"""

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
import scipy.signal
import scipy.io.wavfile as sciwav

sys.path.append("tools")
from psf_zl import get_energy
from wav_tools import silence_handler 

def vec_padding(vector, len_limit):
    """Padding vector using 0 if it has shorter length of len_limit
    Input:
      vector: vector
      len_limit: int
    """
    if(len(vector) < len_limit):
        return np.hstack((vector, np.zeros(len_limit - len(vector))))
    else:
        return vector[:len_limit]

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

    return np.array(vad)

def cal_kaldi_vad_for_wav(sig, samplerate=16000, winlen=0.025, winstep=0.01):
    """Calculating VAD for a given signal"""
    energy = get_energy(sig, samplerate=samplerate, winlen=winlen, winstep=winstep)
    vad = cal_kaldi_vad_by_energy(energy)
    return vad

def cal_kaldivads_for_scp(wav_scp, winlen=0.025, winstep=0.01, samplerate=16000): 
    """Given a scp file, calculating energy-based VAD according kaldi

    Input:
      wav_scp: str, SCP file, with format for each line as: <wavid> <path>
      winlen: flot, frame length, default is 0.025 sec.
      winstep: float, frame shift, default is 0.01 sec.
      samplerate: float, sampling rate, default is 16kHz

    Output:
      vad_dic: <dict> {'wavid': [vad vector]; ...}
    """
    # Read wav from scp file.
    utt2wav = dict([line.split() for line in open(wav_scp)])

    #init window:
    winlen_p = int(winlen * samplerate)
    winstep_p = int(winstep * samplerate)

    vad_dict = {}
    for uttid, uttdir in tqdm(utt2wav.items()):
        sr, sig = sciwav.read(uttdir)
        assert sr == samplerate

        n_frames = int(sig.shape[0] / sr / winstep)
        vad_dict[uttid] = cal_kaldi_vad_for_wav(sig, samplerate=sr, 
                                                winlen=winlen, winstep=winstep)

    return vad_dict

def cal_maxeng_vad_for_wav(sig, samplerate, 
                           winlen_p, winstep_p, 
                           shortest_len_in_ms=0.0):
    """ Calculate max-energy-based VAD following sec. 5.1 from 
        An Overview of Text-Independent Speaker Recognition: From Features to Supervectors.
        Tomi Kinnunen, and Haizhou Li.
        https://www.sciencedirect.com/science/article/pii/S0167639309001289
        code from https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob
           /master/core_scripts/data_io/wav_tools.py#L289 
    Input:
    ------
      wav_scp: str, SCP file, with format for each line as: <wavid> <path>
      winlen_p: int, frame length, default is 400 = 0.025*16000
      winstep_p: int, frame shift, default is 160 = 0.010*16000
      samplerate: float, sampling rate, default is 16kHz
      shortest_len_in_ms: not used here.

    """
    vad = silence_handler(sig, sr=samplerate, fl=winlen_p, fs=winstep_p, 
                          shortest_len_in_ms=shortest_len_in_ms)[2] 
    #short
    return vad

def cal_maxeng_vads_for_scp(wav_scp, samplerate=16000, 
                            winlen=0.025, winstep=0.01, 
                            shortest_len_in_ms=0.0): 
    """Given a scp file, calculating max-energy-based VAD according to Tomi's review paper

    """
    # Read wav from scp file.
    utt2wav = dict([line.split() for line in open(wav_scp)])

    #init window:
    winlen_p = int(winlen * samplerate)
    winstep_p = int(winstep * samplerate)
    vad_dict = {}
    for uttid, uttdir in tqdm(utt2wav.items()):
        sr, sig = sciwav.read(uttdir)
        assert sr == samplerate

        n_frames = int(sig.shape[0] / sr / winstep)
        vad_dict[uttid] = cal_maxeng_vad_for_wav(sig, samplerate=sr, 
                                                 winlen_p=winlen_p, winstep_p=winstep_p, 
                                                 shortest_len_in_ms=shortest_len_in_ms)

    return vad_dict

def debug_kaldi_vad():
    WAV_SCP="source_data/train/wav.scp"
    vad_dict = cal_kaldivads_for_scp(WAV_SCP)
    if( not os.path.exists("exp_debug") ):
        os.makedirs("exp_debug")
    np.save('exp_debug/vad_dict.npy', vad_dict)
    
def debug_maxeng_vad():
    WAV_SCP="source_data/train/wav.scp"
    vad_dict = cal_maxeng_vads_for_scp(WAV_SCP)
    if( not os.path.exists("exp_debug") ):
        os.makedirs("exp_debug")
    np.save('exp_debug/vad_dict.npy', vad_dict)

if __name__ == '__main__':
    debug_maxeng_vad()
