#!/usr/bin/env python
# Copyright 2021-2025 National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.
"""
This script convert frame-level speech activity detection marks to standard vad file per audio:
For each audio, creat a stand vad like
<start_time> <end_time> <label>
<start time> and <end time> are in ms 
<label> could be in genuine 1/ spoof 2-/ nonspeech 0, etc. 

----
Created on Nov. 9 01:35 2020
Modify on March 1 2021
Author: Lin Zhang
"""
import os
import argparse
import itertools
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--TYPE',type=str, default='zlnpy', choices=['kaldi', 'zlnpy'],
                        help="format of the input vad file?")
    parser.add_argument('--VAD_FILE',type=str, default='source_data/train/kaldi_vad.scp')
    parser.add_argument('--VAD_SAVE_DIR',type=str, default='exp/train/vad_stand_ms/')

    parser.add_argument('--RECO2DUR',type=str, default='source_data/train/reco2dur')
    parser.add_argument('--LABEL2NUM',type=str, default='')
    parser.add_argument('--WAV2TYPE',type=str, default='')
    parser.add_argument('--shift_step_ms',type=float, default=10)

    # Filter too short silence/segments
    parser.add_argument('--min_sil_ms',type=float, default=0,
                         help="Minimum duration (in ms) required for a silence to be included"
                         "Silence shorter than this duration will be removed")
    parser.add_argument('--min_seg_ms',type=float, default=0,
                         help="Minimum duration (in ms) required for a segment to be included"
                         "Segments shorter than this duration will be removed")
    
    args = parser.parse_args()

    return args

args = get_args()

# Preparing
if (args.LABEL2NUM and args.WAV2TYPE):
    #as we use A0* now, delete label2unm
    label2num = dict([line.split() for line in open(args.LABEL2NUM)])
    wav2type = dict([line.split() for line in open(args.WAV2TYPE)])
reco2dur = dict([ [line.split()[0], line.split()[1] ]  for line in open(args.RECO2DUR)])

#def filter_short_sil(vec, min_sil_ms, shift_step):
#    """ NOT USE
#    detect too short silence, label them as speech.
#    inefficient, not use anymore
#    """
#
#    if(min_sil_ms <= 0):
#        return vec
#    it = 0
#    # Make an iterator that returns consecutive keys and groups from the iterable
#    # k = 0|1, v is the continous [0] or [1] vector.
#    for k,v in itertools.groupby(vec):
#        length=len(list(v))
#        if( (k == 0 or k == 0.0)
#                and length <= float(min_sil_ms)/float(shift_step)): 
#            vec[it: it + length] = 1
#        it = it + length
#
#    return vec

def ignore_short_seg(frame_tag, seg_len_thres):
    """
    Ignore some short segment/nonspeech. note that only support 0/1

    Input:
    ------
        frame_tag: np.array<int> (1,)
        seg_len_thres: int, shortest frame number

    Output:
    ------
        np.array, 
    """
    frame_tag_new = np.zeros_like(frame_tag) + frame_tag
    # boundary of each segment
    seg_bound = np.diff(np.concatenate(([0], frame_tag, [0])))
    # start of each segment
    seg_start = np.argwhere(seg_bound == 1)[:, 0]
    # end of each segment
    seg_end = np.argwhere(seg_bound == -1)[:, 0]
    assert seg_start.shape[0] == seg_end.shape[0], \
        "Fail to extract segment boundaries"
    
    # length of segment
    seg_len = seg_end - seg_start
    seg_short_ids = np.argwhere(seg_len < seg_len_thres)[:, 0]
    for idx in seg_short_ids:
        start_frame_idx = seg_start[idx]
        end_frame_idx = seg_end[idx]
        frame_tag_new[start_frame_idx:end_frame_idx] = 0
    return frame_tag_new

def vadvec2vadary(vec, shift_step, dur, new_label):
    """
    Convert vad vec [0, 0, ..., 1, 1, 0,...] 
        to array <begin_ms> <end_ms> <label>

    Input:
    ------
       vec: np.array in (1,), vad vector
       shift_step: float, in ms
       dur: float, Duration of this vec.
       new_label: Instead of speech (1), labeling using more specific labels, like A01...

    Output:
    ------
        vad_ary: np.array in dim [frame_len, 3], eahc row: [<begin_ms>, <end_ms>, <label>] 
    """
    lenvad = []
    last = 0
    for k,v in itertools.groupby(vec):
        if(args.WAV2TYPE and k != 0): 
            # If wav2type is provided, and this segment is not nonspeech(0)
            #   convert speech label into more specific label.
            k = np.int(new_label) 
        lenvad.append([len(list(v)), int(k)]) # Get homergenous part, save length and label.
    
    lenvad=np.array(lenvad)
    lenvad[:, 0] = np.cumsum(lenvad[:, 0], axis=0) # Get the end point
    lenvad_st = np.copy(lenvad[:,0]).reshape(-1,1)
    lenvad_st[1:] = lenvad_st[0:-1] # Get the start point
    lenvad_st[0, 0] = 0

    vad_time=np.hstack((lenvad_st, lenvad))
    
    # Convert frame to ms
    vad_time[:, 0:2] = vad_time[:, 0:2] * shift_step

    # For the rest duration, convert to ms
    vad_time[-1, 1] = dur * 1000 

    return vad_time


def zlnpy2standvad(TYPE, vad_file, vad_save_dir, shift_step=10, min_sil_ms=0, min_seg_ms=0):
    """
    Convert frame-level vad to standard vad in ms
    input: 
    ------
        vad_file: str, saved .npy file for vad, {'uttid':[vad_vector], ...}
        vad_save_dir: str, save processed vad into here
        shift_step: in ms, defauls = 10ms
        min_sil_ms: silence shorter than min_sil_ms will be ignore.

    """

    if os.path.exists(vad_save_dir):
        os.system("rm -rf " + vad_save_dir)
    #os.makedirs(vad_save_dir)

    if (TYPE == "kaldi"):
        vad_npy = kaldi_io.read_vec_flt_scp(vad_file)
    else:
        vad_npy = np.load(vad_file, allow_pickle=True).tolist()

    for uttid, vec in tqdm(vad_npy.items()):
        vec = vec.astype(int)
        if(min_sil_ms>0): # Non-speech
            #vec = filter_short_sil(vec, min_sil_ms, shift_step)
            # 1 - vec indicate nonspeech
            vec = ignore_short_seg(1 - vec, min_sil_ms / shift_step)
            vec = 1 - vec
        if(min_seg_ms>0): # Speech 
            vec = ignore_short_seg(vec, min_sil_ms / shift_step)


        if(args.WAV2TYPE): 
            # If wav2type is provided
            # We will use updated label (like 2 for A01, 3 for A02 ...) instead of 1 for speech 
            new_label = label2num[wav2type[uttid]] 
        else: new_label = ''
        vadary_time = vadvec2vadary(vec, shift_step, float(reco2dur[uttid]), new_label) 
        vadfile_save= os.path.join(vad_save_dir, uttid + '.vad')
        np.savetxt(vadfile_save, vadary_time, fmt='%s %s %d')
        
def main():
    zlnpy2standvad(TYPE=args.TYPE, 
                   vad_file=args.VAD_FILE, vad_save_dir=args.VAD_SAVE_DIR,
                   shift_step=args.shift_step_ms, 
                   min_sil_ms=args.min_sil_ms, min_seg_ms=args.min_seg_ms)
    

def debug():
    vad_file='source_data/train/vad/vad_dict_vote.npy'
    vad_save_dir='tmp'
    zlnpy2standvad('zlnpy', vad_file, vad_save_dir, min_sil_ms = 40)


if __name__ == '__main__':
#    main()
     debug()
