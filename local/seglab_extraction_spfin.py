
"""
Created on Feb 12 2021

@author: zlin

Creates "segment level embeddings with label" for uisrnn
https://github.com/google/uis-rnn

"""

import os
import sys
import math

import torch
import importlib
import numpy as np


import argparse
from tqdm import tqdm


from collections import defaultdict
#from numpy.lib.stride_tricks import sliding_window_view

#Argument Parser
def str2bool(value):
    """A function to convert string to bool value."""
    if value.lower() in {'True', 'yes', 'true', 't', 'y', '1'}:
        return True
    if value.lower() in {'False', 'no', 'false', 'f', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser('Feature Prepare: python seglab_extraction_spfin.py')

parser.add_argument('--rttm-file',type=str, default='/home/smg/zhanglin/workspace/00ad/00data_prepare/exp/train/con_data/rttm')
parser.add_argument('--reco2dur',type=str, default='/home/smg/zhanglin/workspace/00ad/00data_prepare/exp/train/con_data/reco2dur')
parser.add_argument('--label2num',type=str, default='/home/smg/zhanglin/workspace/00ad/00data_prepare/exp/train/con_data/label2num')

parser.add_argument('--seg-len',type=int, default=1) 
parser.add_argument('--seg-overlap-unit',type=int, default=0, help='the overlap number of embedding units ') 
parser.add_argument('--seg-unit-sec',type=float, default=0.16) 
parser.add_argument('--sample-rate',type=float, default=16000) 

parser.add_argument('--gen2spf-list',type=str, default='') 
parser.add_argument('--skip_lastseg',type=str2bool, default=False) 
#parser.add_argument('--spf2gen-list',type=str, default='') 

parser.add_argument('--seglab-save-file',type=str, default='exp/tmp')

args = parser.parse_args()


reco2dur = dict([line.split() for line in open(args.reco2dur)])
label2num = dict([line.split() for line in open(args.label2num)])
gen2spf = np.loadtxt(args.gen2spf_list, dtype=str)

def rttmlab2spflab(dialabvec, gen2spf_flag=False):
    """
    convert label to spoof label:
      spoof/ nospeech from spoof: 0
      geniune/ nospeech from genuine: 1

    """
    spflabvec = np.zeros_like(dialabvec, dtype=int).astype(str)
    #bonafide
    spflabvec[np.where(dialabvec=='1')]='1'

   # #spoof
   # spflabvec[np.where((dialabvec!='0') & (dialabvec != '1') )]='0'

    #nonspeech from bonafide
    if(not gen2spf_flag):
        spflabvec[np.where(dialabvec=='0')]='1'

    #else:
    #spflabvec[np.where(dialabvec=='0')]='0'

    return spflabvec


def get_rttm(rttm_file):
    rttm = defaultdict(list)
    with open(rttm_file) as f:
        for line in f.readlines():
            _, reco , channels, st, dur, _, _, lab, _, _  = line.split() 
            rttm[reco].append([lab, float(st), float(st) + float(dur) ])
    return rttm

def spflab_sp2spfinlab(args, filename, labvec_sp, num_frames=None):
    """

    skip_lastseg: do we need to label the last part, which len(last)<len(unit) 
    """
    seg_shift_sp=args.sample_rate * float( args.seg_unit_sec *  (args.seg_len - args.seg_overlap_unit ) ) 
    if(num_frames==None):
        num_frames = math.ceil(args.sample_rate * float(reco2dur[filename]) / seg_shift_sp)
    spfinlabvec = np.zeros(num_frames, dtype=int).astype(str)
    spfratiolabvec = np.zeros(num_frames, dtype=float)
    for idx in np.arange(num_frames):
        #wherther  this segment include spoof sample point.
        #when sum == len, means only contain genuine.
        sp, ep = int(idx * seg_shift_sp), min(int((idx+1)*seg_shift_sp),int(len(labvec_sp)))
        if( ((idx+1)*seg_shift_sp) > len(labvec_sp) and args.skip_lastseg):
            break

        #gen_spnum=sum(labvec_sp[sp:ep]) labvec_sp is str, cannot use sum
        gen_spnum=sum(labvec_sp[sp:ep]=='1')
        spf_spnum=(ep-sp)-gen_spnum   #ep-sp not seg_shift_sp because the last segment may < seg_shift_sp
        spfinlabvec[idx]=str(int(spf_spnum==0))   #do not have spoof frame
        spfratiolabvec[idx]=float(spf_spnum)/float(seg_shift_sp)

    return spfinlabvec, spfratiolabvec


def rttm2labvec_sp(args, filename, one_rttm, num_frames=None ): 
    """ 
    sp -> sample point
    
     Input: 
     Output: 
         get sample point label.

    """
    dur = float(reco2dur[filename])
    tol_sp = round(dur * args.sample_rate)
    #convert rttm to sample point label vec.
    labvec_sp=np.zeros(tol_sp, dtype=int).astype(str)
    for idx, (lab, st, et) in enumerate(one_rttm):
        sp, ep = round(st * args.sample_rate), round(et * args.sample_rate)
        labvec_sp[sp:ep] = str(label2num[lab])

    #convert sample point label vec to spoof label:
    #  spoof/ nospeech from spoof: 0
    #  geniune/ nospeech from genuine: 1
    spflabvec_sp=rttmlab2spflab(labvec_sp, filename in gen2spf)    

    spfinlabvec, spfratiolabvec = spflab_sp2spfinlab(args, filename, spflabvec_sp)     

    return np.array(spfinlabvec, dtype=str), np.array(spfratiolabvec, dtype=float)


def rttm2labvec(args, filename, one_rttm, num_frames=None ): 
    
    """
    rttm2labvec:
        convert rttm for one session to lab 
        
    input:
        args:      from argpase
        one_rttm:  one rttm for one session, [segment_len * 3]
                   each line <lab>  <st> <end>
    output:
        labvec:    lab in segment level: [1 * segment_len]    
        
    """
    seg_shift_sec=float( args.seg_unit_sec *  (args.seg_len - args.seg_overlap_unit ) ) 
    if(num_frames==None):
        #num_frames = int(float(reco2dur[filename]) / seg_shift_sec)
        #score.shape in the asvspf model, so here we use math.ceil
        num_frames = math.ceil(float(reco2dur[filename]) / seg_shift_sec)
    labvec = np.zeros(num_frames, dtype=int).astype(str)
    for idx, (lab, st, et) in enumerate(one_rttm):
        sf, ef = round(st / seg_shift_sec), round(et / seg_shift_sec)
        labvec[sf:ef] = str(label2num[lab])
    while(ef < num_frames):
        move = min(num_frames-ef, ef) 
        if (move==0): move =1 
        labvec[ef: ef+move] = labvec[: move ]
        ef = ef+move
        
    spflabvec=rttmlab2spflab(labvec, filename in gen2spf)    

    return np.array(spflabvec, dtype=str)
    
    

def main():
    #1. get neccessary embedding, filenames, rttm 
##    #filenames = np.load(args.filename_file, allow_pickle=True)
    rttm = get_rttm(args.rttm_file)
    
    
    sequences=[]
    cluster_ids=[]
    cluster_recoids = []
    save_scores = []
    seglab=defaultdict(list)
    seglab_ratio=defaultdict(list)

    for uttid, v in tqdm(rttm.items()):
#        if(uttid).startswith('CON'):
#            print("debug")
       
        labvec, ratiolabvec = rttm2labvec_sp(args, uttid, rttm[uttid])
        seglab[uttid]=labvec
        seglab_ratio[uttid]=ratiolabvec

    np.save(args.seglab_save_file,seglab)
    np.save(args.seglab_save_file+'_ratio',seglab_ratio)


def debug():
    rttm = get_rttm(args.rttm_file)
    uttid='CON_T_0000000'
    uttid='LA_T_9987202'
    labvec, ratiolabvec = rttm2labvec_sp(args, uttid, rttm[uttid])
    return

if __name__ == "__main__":
    main()
#    debug()


