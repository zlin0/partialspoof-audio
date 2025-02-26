#!/usr/bin/env python
# Copyright 2021-2025 National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

"""
To calculate vad for a wav or wav_scp
Supporting: 
(1) VAD in dict saved in npy format: {'uttid':[vad_vector]}
VAD Sec.5.1 of https://www.sciencedirect.com/science/article/pii/S0167639309001289
(2) other pretrained models to compute VAD: pyannote

"""


import os
import sys
import argparse
import numpy as np
import scipy.signal
from tqdm import tqdm
import scipy.io.wavfile as sciwav
import torch

#vad 
sys.path.append("tools")
from psf_zl import get_energy

from partialspoof.audio.vad.compute_vad import vec_padding
from partialspoof.audio.vad.compute_vad import cal_kaldi_vad_for_wav, cal_maxeng_vad_for_wav

#from pyannote.audio.utils.signal import Binarize

def get_args():
    #####Argument Parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--WAV_SCP',type=str, default='source_data/train/wav.scp',
                        help="scp file specify path to wav.")
    parser.add_argument('--VAD_DIR',type=str, default='source_data/train/vad', 
                        help="folder to save computed VAD.")
    
    #parmeters for VAD:
    parser.add_argument('--vad_pyannote', action='store_true', default=False, 
                        help="Whether to use pyannote to compute VAD.")
    parser.add_argument('--precompute_vad_pyannote',type=str, default='', 
                        help="precomputed VAD using pyannote.")
    parser.add_argument('--vad_kaldi', action='store_true', default=False, 
                        help="Whether to use window shifting energy-based VAD from kaldi.")
    parser.add_argument('--precompute_vad_kaldi',type=str, default='', 
                        help="precomputed kaldi VAD .")
    parser.add_argument('--vad_maxeng', action='store_true', default=False, 
                        help="Whether to use max-energy-based VAD from Tomi's paper.")
    parser.add_argument('--precompute_vad_maxeng',type=str, default='', 
                        help="precomputed maxeng VAD .")
    
    mse  = "This is for additional VAD besides above defined/computed one."
    mse += "It can be added in a list like: --other-precompute-vad vad1.npy vad2.npy"
    parser.add_argument('--precompute_vad_other', type=str, default='', help=mse)
    # TODO: support multiple precomputed VAD.
    #parser.add_argument('--precompute_vad_other', type=str, nargs='+', \
    #                    default=[''], help=mes)
    
    args = parser.parse_args()

    return args


def cal_npyvad_by_scp(args, vote=2, 
                      winlen=0.025, winstep=0.01, 
                      samplerate=16000, low=0.5, high=5): 
    """
    Input:
    ------
      args: argparse
      vote: int, voting to set VAD prediction.
      winlen, sinstep: float, windown length and shift
      samplerate: sampling rate

    """
    mse = "At least one method for computing VAD need to be assigned."
    assert(args.vad_kaldi or args.vad_maxeng or args.vad_pyannote), mse
    
    #init
    utt2wav = dict([line.split() for line in open(args.WAV_SCP)])
    if(not vote):
        vote = np.ceil((args.vad_kaldi+args.vad_maxeng+args.vad_pyannote)*0.5)
    #assert vote > 0

    # Init window convert from second to sample point.:
    winlen_p = int(winlen * samplerate)
    winstep_p = int(winstep * samplerate)

    # Load pretrained VAD model:
    if(args.vad_pyannote):
        if(args.precompute_vad_pyannote):
            # We can load VAD if we already have it.
            pre_vad_pyannote=np.load(args.precompute_vad_pyannote, allow_pickle=True).tolist()
        else:
            # Else we will load pretrained model from pyannote and compute VAD.
            sad_dihard_model = torch.hub.load('pyannote/pyannote-audio', 'sad_dihard') 
            # Support sad_ami, sad_dihard
            binarize = Binarize(offset=0.5, onset=0.5, log_scale=True,
                                            min_duration_off=0.1, min_duration_on=0.1)
    # Load precomputed VAD in npy:
    if(args.vad_kaldi and args.precompute_vad_kaldi):
        pre_vad_kaldi = np.load(args.precompute_vad_kaldi, allow_pickle=True)
    if(args.vad_maxeng and args.precompute_vad_maxeng):
        pre_vad_maxeng = np.load(args.precompute_vad_maxeng, allow_pickle=True)
    if(args.precompute_vad_other):
        pre_vad_other = np.load(args.precompute_vad_other, allow_pickle=True)

    vad_dict = {}
    vads_dict = {}
    for uttid, uttdir in tqdm(utt2wav.items()):
        sr, sig = sciwav.read(uttdir)
        assert sr == samplerate

        n_frames = int(sig.shape[0] / sr / winstep)
        vads_dict[uttid]=np.zeros(n_frames) # To save all VAD (sum up) from multiple systems for the final voting.
        vad_dict[uttid]=np.zeros(n_frames)  # To save the final VAD 
        if(args.vad_kaldi):
            if(args.precompute_vad_kaldi):
                vad = pre_vad_kaldi[uttid]
            else:
                vad = cal_kaldi_vad_for_wav(sig, samplerate=sr, 
                                            winlen=winlen, winstep=winstep)
            vads_dict[uttid] = np.sum((vads_dict[uttid], vec_padding(vad, n_frames)), axis=0)
        if(args.vad_maxeng):
            if(args.precompute_vad_maxeng):
                vad = pre_vad_maxeng[uttid]
            else:
                vad = cal_maxeng_vad_for_wav(sig, samplerate = sr, 
                                         winlen_p=winlen_p, winstep_p = winstep_p, 
                                         shortest_len_in_ms=0) # shortest_len_in_ms will be define later
            vads_dict[uttid]=np.sum((vads_dict[uttid], vec_padding(vad, n_frames)), axis=0) 
            # Note that if cal_maxeng_vad_for_wav return 3-dim, we need to choose[2] 
            # as silence_handler implemented in xin's code support return wav_no_sil, sil_wav, time_tag
        if(args.vad_pyannote):
            if(args.precompute_vad_pyannote):
                vad = pre_vad_pyannote[uttid][:n_frames]
            else:
                sad_dihard_scores = sad_dihard_model({'uri': uttid, 'audio': uttdir})
                speech = binarize.apply(sad_dihard_scores, dimension=1) # timeline instance
                vad = np.zeros(n_frames)
                for i, speech_region in enumerate(speech):
                   st, et = '%.3lf' % speech_region.start, '%.3lf' % speech_region.end
                   sf, ef = round(speech_region.start / winstep), round(speech_region.end / winstep)
                   vad[sf:ef] = 1
            vads_dict[uttid]=np.sum((vads_dict[uttid], vec_padding(vad, n_frames)),axis=0)

        # If more than vote times VAD detect it as speech, we assign label as speech (1) 
        vad_dict[uttid][np.where(vads_dict[uttid] >= vote)] = 1

    return vad_dict

def main():
    args = get_args()
    vad_dict = cal_npyvad_by_scp(args)
    if( not os.path.exists(args.VAD_DIR) ):
        os.makedirs(args.VAD_DIR)
    np.save(os.path.join(args.VAD_DIR,'vad_dict_vote.npy'), vad_dict)
    # np.load('vad_dict.npy', allow_pickle=True)

def define_debug_args():
    class Args:
        WAV_SCP = 'source_data/train/wav.scp'
        VAD_DIR = 'source_data/train/vad'
        vad_pyannote = False
        precompute_vad_pyannote = ''
        vad_kaldi = True
        precompute_vad_kaldi = ''
        vad_maxeng = True
        precompute_vad_maxeng = ''
        precompute_vad_other = ''
    args = Args()    
    return args
def debug():
    args = define_debug_args()
    vad_dict = cal_npyvad_by_scp(args)

if __name__ == '__main__':
    main()
#    debug()
