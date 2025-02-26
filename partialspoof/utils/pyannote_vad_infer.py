#!/usr/bin/env python
# Copyright 2021 National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

import os
import sys
import tqdm
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wf
sys.path.append("/home/smg/zhanglin/workspace/02diarization/pyannote-audio/pyannote")

from pyannote.audio.utils.signal import Binarize

def get_args():
    parser = argparse.ArgumentParser(description='Inference for voice activity detection')
    # data config
    parser.add_argument('--wav_scp', default='data/train/wav.scp', type=str)
    parser.add_argument('--out_segments', default=None, type=str,
            help="""Output the results as segment file, where each line in this file is: 
                        [segment_id, utt, start_of_speech, end_of_speech]""")
    parser.add_argument('--out_npy', default=None, type=str,
            help="""Output the results as a dict {utt: vad numpy array which contains 0s and 1s}. 
                        frame_shift should be specified with this parameter, default=0.01s""")
    parser.add_argument('--frame_shift', default=0.01, type=float)
    parser.add_argument('--sample_rate', default=16000, type=int)
    # model config
    parser.add_argument('--hubconf_dir', default='.', type=str)
    parser.add_argument('--model', default='sad_ami', type=str)
    parser.add_argument('--source', default='local', type=str)
    # infer config
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_args()

    return args

def get_models(args):
    ''''''
    #return torch.hub.load(args.hubconf_dir, args.model, source=args.source)
    return torch.hub.load('pyannote/pyannote-audio', args.model)  #pipeline=True


def infer(args):
    sad_model = get_models(args)
    wav_scp = [line.split() for line in open(args.wav_scp)]
    out_segments, out_npy = [], {}

    for utt, wav_file in tqdm.tqdm(wav_scp, ncols=100):
        # check sample rate
        sr, y = wf.read(wav_file)
        assert sr == args.sample_rate

        test_file = {'uri': utt, 'audio':wav_file}
        sad_scores = sad_model(test_file)
        binarize = Binarize(offset=0.52, onset=0.52, log_scale=True,
                                    min_duration_off=0.1, min_duration_on=0.1)
        speech = binarize.apply(sad_scores, dimension=1) # timeline instance

        vad_label = np.zeros(round(y.shape[0] / sr / args.frame_shift))
        for i, speech_region in enumerate(speech):
#             segment_id = '%s_%04d' % (utt, i)
            st, et = '%.3lf' % speech_region.start, '%.3lf' % speech_region.end
#             out_segments.append([segment_id, utt, st, et])
            sf = round(speech_region.start / args.frame_shift) 
            ef = round(speech_region.end / args.frame_shift)
            vad_label[sf:ef] = 1
        out_npy[utt] = vad_label

#     # write segments
#     with open(args.out_segments, 'w') as f:
#         for line in out_segments:
#             f.write('%s\n' % ('\t'.join(line)))
            
    # save vad label
    np.save(args.out_npy, out_npy)

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    infer(args)
