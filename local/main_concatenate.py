#!/usr/bin/env python
"""

-----
Create on Nov. 2020
Modify on March 6 2021
Author: Lin Zhang


"""
import scipy.io.wavfile as sciwav
import numpy as np
import argparse
import logging
import random
import sys
import os

from collections import defaultdict
from tqdm import tqdm

from bisect import bisect

#####Argument Parser
def str2bool(value):
    """A function to convert string to bool value."""
    if value.lower() in {'True', 'yes', 'true', 't', 'y', '1'}:
        return True
    if value.lower() in {'False', 'no', 'false', 'f', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--ori_vad_scp',type=str, default='data/eval/vad-stand-ms-sil0-seg0/bonafide_vad.scp')
parser.add_argument('--insf_spk_vad_scp',type=str, default='data/eval/vad-stand-ms-sil0-seg0/spoof-person-vad.scp')
parser.add_argument('--utt2spk',type=str, default='data/eval/utt2spk')
parser.add_argument('--concatenate_wav_dir',type=str, default='exp/eval/con_wav')
parser.add_argument('--wav_scp',type=str, default='data/eval/wav.scp')
# parser.add_argument('--spf2num',type=str, default='/home/smg/zhanglin/workspace/00ad/data/label2num')
# parser.add_argument('--NOTE_FILE', type=str, default='/home/smg/zhanglin/workspace/00ad/exp/concate.note') #cannot work
parser.add_argument('--insert_log',type=str, default='tmp_eval_spf2gen_insert.log')

parser.add_argument('--overlap_rate', type=float, default=0.5)
parser.add_argument('--counter', type=int, default=0)
parser.add_argument('--prefix_name', type=str, default='CON')

parser.add_argument('--rand_seed', type=int, default=18)

parser.add_argument('--similar_length', type=str2bool, default=False, help='select segment with similar length.')
parser.add_argument('--insert_label_type', type=str, choices=['gen2spf', 'spf2gen', 'common'], default='common', 
        help='inser geniune to spoof, this will reflect label.')
parser.add_argument('--min_sil_ms', type=int, default=20, help="""
        after insert, delete silence duration shorter than min_sil_len
        only implement in common.
        """)
parser.add_argument('--allow_reuse', type=str2bool, default=False, help='Allow use the same segemnts or not. line 233')
parser.add_argument('--must_insert', type=str2bool, default=False, help='Allow use the original audio or not. line ')
parser.add_argument('--save_wav', type=str2bool, default=True, help='Whether to save wav (for case re generate labels.) ')

args = parser.parse_args()

COUNTER = args.counter


#data 
ori_utt2vad=dict([ [line.split()[0], line.split()[1] ] for line in open(args.ori_vad_scp) ])
utt2spk = dict([line.split() for line in open(args.utt2spk)])
utt2wav =  dict([line.split() for line in open(args.wav_scp)])
insf_spk2vad = dict([line.split() for line in open(args.insf_spk_vad_scp)])


insert_logID=open(args.insert_log,'w')

def get_vad_by_uttid(uttid):
    vad_dir=ori_utt2vad[uttid]
    fileID=open(vad_dir,'r')
    #dtypes={'names':('st','et','lab'),'formats':(float, float, int)}
    vad=np.loadtxt(fileID, delimiter =' ', dtype=np.float)
    fileID.close()
    vad=vad.reshape(-1,3)
    return vad

#already checked in shell
# #check concatenate dir
# if os.path.exists(args.concatenate_wav_dir):
#     os.system('rm -rf {}'.format(args.concatenate_wav_dir))
# os.makedirs(args.concatenate_wav_dir)

def check_con_label(new_label):
    new_label = np.array(new_label)
    #1. dur=0 segment
    new_label = np.delete(new_label, 
            np.where(new_label[:,1]-new_label[:,0]<=1e-10), axis=0) 
    #2. continue class
    del_lst = []
    pre_cls=-1
    for idx in np.arange(new_label.shape[0]):
        # iterate check by row
        if(new_label[idx][2]==pre_cls):
            new_label[idx][0] = new_label[idx-1][0]
            del_lst.append(idx-1)
        else: 
            pass
        pre_cls = new_label[idx][2]

    new_label = np.delete(new_label, del_lst, axis=0)    

    return new_label
        

def whole_concatenate(args, ori_uttid, ori_wav, ori_sr, up_label, new_label, ins_pos, 
                      insf_spkvad, overlap_rate = 0.5):
    '''
    ori_wav is needed as we should updated iterate.

    ori_uttid 
    ori_wav, ori_sr: the orignal signal.
    # ori_vad: vad file for the orginal speech. in ms. changed to new_label, so deleted
    up_label: used to concatenate wav, (use the overlap_rate silence to do overlap-add). (the 1st version of PS).
    new_label: new vad file for the lastest concate speech, in point. we need to update it each iterate.
    ins_pos: insert position, index in the ori_vad list.
    insf_spkvad: All spoof segment of VAD file for spkid.
    overlap_rate = overlap_rate: how long short pause will be used.
    similar_length: if we want to use segment with similar length to replace the original one.
    '''
    global COUNTER

    up_label = np.array(up_label)
    new_label = np.array(new_label)
    len_label = up_label.shape[0]
    #new_label = np.array(new_label, dtype=int)



    #1.2 find the start & end [point] by overlaprate 

    ori_overlap_st_len = 0 if (ins_pos == 0) else (up_label[ins_pos-1,1]-up_label[ins_pos-1,0])* (1-overlap_rate) #or(1-overlap_rate)
    ori_overlap_end_len = 0 if (ins_pos >= len(up_label)-1) else (up_label[ins_pos+1,1]-up_label[ins_pos+1,0]) * overlap_rate   #or(1-overlap_rate)
    ori_st  = up_label[ins_pos,0] - ori_overlap_st_len   
    ori_end = up_label[ins_pos,1] + ori_overlap_end_len
    ori_len = up_label[ins_pos,1] - up_label[ins_pos,0]

    #2. select spoof
    #2.1 random select one segment (insert from insf_idx index) from spkvad
    spf_spe_idx_list = (np.asarray(np.where((insf_spkvad[:,3] != '0.0') & (insf_spkvad[:,3] != '0')))).flatten()
    spf_spe_len = (insf_spkvad[:,2].astype(float)-insf_spkvad[:,1].astype(float))* ori_sr / 1000  #ms -> point

    if(args.similar_length): #select index position with similar legth.
        ind_group_len = defaultdict(list)
        #a. sort spf by length, get the sorted array and index after sorted.
        sorted_spf_spe_len = np.sort(spf_spe_len)
        permute_index = np.argsort(spf_spe_len)
        #b. uniq length, create dict for each length. each dict[length] = all index with same length.
        spf_spe_len_uniq = np.unique(sorted_spf_spe_len)
        for le in spf_spe_len_uniq:
            group_len = permute_index[np.where(sorted_spf_spe_len == le)]
            group_len = np.intersect1d(group_len, spf_spe_idx_list)
            if(len(group_len>0)):
                ind_group_len[le] = group_len
        
        #c. get ori_len's position in the sorted_spf_len.
        #sim_len_idx = sorted_spf_spe_len.searchsorted(ori_len) 
        #sim_len_idx = min(sim_len_idx, len(sorted_spf_spe_len)-1)
        #sim_len = sorted_spf_spe_len[sim_len_idx]

        ind_group_len_key=np.array(list(ind_group_len.keys()))
        sim_len_idx = ind_group_len_key.searchsorted(ori_len) 
        sim_len_idx = min(sim_len_idx, len(ind_group_len_key)-1)
        sim_len = ind_group_len_key[sim_len_idx]

        sim_len_idx_group = ind_group_len[sim_len]
        
        #d. randomly choise one. 
        #random.seed(COUNTER)
        random.seed(abs(COUNTER-len(sim_len_idx_group)))  #different for every utt, but rebuildable.
        insf_idx = random.sample(sim_len_idx_group.tolist(), 1)[0]   
            
    else: 
        random.seed(abs(COUNTER-len(spf_spe_idx_list)))  #different for every utt, but rebuildable.
        #random.seed(COUNTER)  
        insf_idx = spf_spe_idx_list[random.randint(0,len(spf_spe_idx_list)-1)]
    print("insf_idx:",insf_idx )  #testtest
    print("insf_spkvad",insf_spkvad[insf_idx])
    spf_id=insf_spkvad[insf_idx,0]
    insf_sr, insf_wav = sciwav.read(utt2wav[spf_id])
    insf_spkvad_value = insf_spkvad[:,1:].astype(np.float)
    
    #2.2 cut the segment based on overlap rate
    ##if we select the first segment
    insf_overlap_st_len = 0 if ((insf_idx==0) 
            or (insf_spkvad[insf_idx-1, 0] != insf_spkvad[insf_idx, 0]))\
            else (insf_spkvad_value[insf_idx-1,1]-insf_spkvad_value[insf_idx-1,0])*overlap_rate * insf_sr / 1000
    ##if we select the last segment
    insf_overlap_end_len = 0 if ((insf_idx ==len(insf_spkvad)-1 )or
            (insf_spkvad[insf_idx+1, 0] != insf_spkvad[insf_idx, 0]))\
            else (insf_spkvad_value[insf_idx+1,1]-insf_spkvad_value[insf_idx+1,0])*overlap_rate * insf_sr / 1000

    insf_st = insf_spkvad_value[insf_idx,0] * insf_sr / 1000 - insf_overlap_st_len
    insf_end = insf_spkvad_value[insf_idx,1] * insf_sr / 1000 + insf_overlap_end_len 

    spf_seg = insf_wav[int(insf_st):int(insf_end)]
    
    con_ID=args.prefix_name +'_{:07d}'.format(COUNTER)

    insert_info=[con_ID, ori_uttid, int(ori_st), int(ori_end), 
            insf_spkvad[insf_idx,0], int(insf_st), int(insf_end)]  #save replace point 
    insert_string=" ".join([str(x) for x in insert_info])
    
    #3. concatenate  in point
    #return the insert point in the head silence and tail silence
    head_max_id, tail_max_id, con_speech=concatenate(ori_wav, int(ori_st), int(ori_end), 
                                              [int(ori_overlap_st_len), int(ori_overlap_end_len)], 
                                              spf_seg, [int(insf_overlap_st_len), int(insf_overlap_end_len)])
    

    #4. logging for silence and label
    #4.1 head pause
    pause_head_st = 0 if (head_max_id==0) else up_label[ins_pos-1,0] 
    pause_head_end= head_max_id+insf_overlap_st_len
    #4.2 tail pause
    pause_tail_st = pause_head_end + (insf_spkvad_value[insf_idx,1] -insf_spkvad_value[insf_idx,0]) * ori_sr /1000
    pause_tail_end = tail_max_id + ori_overlap_end_len

    #change value after insert position:
    move=0
    move_pos = len(new_label)
    if (ins_pos < len(up_label)-1): #ont the last segment
        move = pause_tail_end - up_label[ins_pos+1,1]
        if (ins_pos + 2<len(up_label)): 
            #new_label[np.where(new_label[:, 0] == up_label[ins_pos+2,0])[0][0]:,0:2] += move 

            move_pos = bisect(new_label[:,0], pause_tail_end-move) - 1
            #min(np.where(new_label[:,0] == up_label[ins_pos+2,0])[0].tolist() \
            #         + [bisect(new_label[:,1], up_label[ins_pos+2, 0])] )
            new_label[move_pos:,0:2] += move 

            #rm_end = min([bisect(new_label[:-7, 0], pause_tail_end)]\
            #+list(np.where(new_label[:,1] == pause_tail_end-move)[0])) #pause_tail_end == new_label[-1,1]
            #new_label[ins_pos+2:,0:2] += move
            up_label[ins_pos+2:,0:2] += move
        #move = pause_tail_end - up_label[ins_pos,1]
 
    #4.3 insert seg
    #treat silence from spoof as spoof
    #the insert index is fixed. we will iterate ins pos by descending order. 
    # 0 for nonspeech (1000 for normal nonmix, 1001 for nonbona, 1002 for nonspf.)
    # 1 for bonafide
    # >=2 for different spf, 
    st_point = pause_head_st + ori_overlap_st_len
    end_point = pause_tail_end - ori_overlap_end_len
    
    orivad = get_vad_by_uttid(ori_uttid)
    ori_lab = np.max(orivad[:,2]) #get original real label for itself.
    non_lab = np.array(1000) + [0, ori_lab, insf_spkvad_value[insf_idx, 2]]  #numpy support Broadcast() to expand 1000 -> [1000, 1000, 1000]
    #if(args.insert_label_type == "gen2spf"): #To label spoof detection 
    #    non_lab = [1000, 1002, 1001] #[mix, self, other]
    #elif(args.insert_label_type == "spf2gen"): 
    #    non_lab = [1000, 1001, 1002] #[mix, self, other]

    # convert nonspeech to the special nonspeech label. 
    # 20221030
    #(although one times convert is enough, we keep this for simple code writing.)
    new_label[np.where(new_label[:,2]==0),2] = non_lab[1]

    up_label = list(up_label)
    new_label = list(new_label)

    new_label.append([pause_head_st, head_max_id, non_lab[1]])
    new_label.append([head_max_id, st_point, non_lab[0]])
    new_label.append([st_point, pause_head_end, non_lab[2]])
    new_label.append([pause_head_end, pause_tail_st, insf_spkvad_value[insf_idx,2]])
    new_label.append([pause_tail_st, end_point, non_lab[2]]) 
    new_label.append([end_point, pause_tail_st + insf_overlap_end_len, non_lab[0]])
    new_label.append([pause_tail_st+insf_overlap_end_len, pause_tail_end, non_lab[1]])
    #20221030 original label need to save. line 130
    #and write a new script   


    #4.3 insert seg
    #original 
    #treat silence from spoof as spoof
    #the insert index is fixed. we will iterate ins pos by descending order. 
    if(args.insert_label_type == "gen2spf"): #To label spoof detection 
        st_point = pause_head_st + ori_overlap_st_len
        end_point = pause_tail_end - ori_overlap_end_len

        up_label.append([pause_head_st, st_point, 0])
        up_label.append([st_point, end_point, insf_spkvad_value[insf_idx,2]])
        up_label.append([end_point, pause_tail_end, 0])
    elif(args.insert_label_type == "spf2gen"): 
        up_label.append([pause_head_st, head_max_id, 0])
        up_label.append([head_max_id, pause_tail_st + insf_overlap_end_len, insf_spkvad_value[insf_idx,2]])
        up_label.append([pause_tail_st + insf_overlap_end_len, pause_tail_end, 0])
    else:   #common, only this one implement min_sil_len to label diarization. with shortest silence.
        pause_head_len = pause_head_end - pause_head_st
        pause_tail_len = pause_tail_end - pause_tail_st
        if(pause_head_len <= args.min_sil_ms and pause_tail_len <= args.min_sil_ms):
            up_label.append([pause_head_st, pause_tail_end, insf_spkvad_value[insf_idx,2]])
        elif(pause_head_len <= args.min_sil_ms and pause_tail_len > args.min_sil_ms):
            up_label.append([pause_head_st, pause_tail_st, insf_spkvad_value[insf_idx,2]])
            up_label.append([pause_tail_st, pause_tail_end, 0])
        elif(pause_head_len > args.min_sil_ms and pause_tail_len <= args.min_sil_ms):
            up_label.append([pause_head_st, pause_head_end, 0])
            up_label.append([pause_head_end, pause_tail_end, insf_spkvad_value[insf_idx,2]])
        else:  #the common common one
            up_label.append([pause_head_st, pause_head_end, 0])
            up_label.append([pause_head_end, pause_tail_st, insf_spkvad_value[insf_idx,2]])
            up_label.append([pause_tail_st, pause_tail_end, 0])


    #else:   #common, only this one implement min_sil_len to label diarization. with shortest silence.
    #    #TODO check
    #    pause_head_len = pause_head_end - pause_head_st
    #    pause_tail_len = pause_tail_end - pause_tail_st
    #    if(pause_head_len <= args.min_sil_ms and pause_tail_len <= args.min_sil_ms):
    #        new_label.append([pause_head_st, pause_tail_end, insf_spkvad_value[insf_idx,2]])
    #    elif(pause_head_len <= args.min_sil_ms and pause_tail_len > args.min_sil_ms):
    #        new_label.append([pause_head_st, pause_tail_st, insf_spkvad_value[insf_idx,2]])
    #        new_label.append([pause_tail_st, pause_tail_end, 0])
    #    elif(pause_head_len > args.min_sil_ms and pause_tail_len <= args.min_sil_ms):
    #        new_label.append([pause_head_st, pause_head_end, 0])
    #        new_label.append([pause_head_end, pause_tail_end, insf_spkvad_value[insf_idx,2]])
    #    else:  #the common common one
    #        new_label.append([pause_head_st, pause_head_end, 0])
    #        new_label.append([pause_head_end, pause_tail_st, insf_spkvad_value[insf_idx,2]])
    #        new_label.append([pause_tail_st, pause_tail_end, 0])


    #4.4 delete the original seg
    #new_label must be in front of up_label, 
    #since we use start point and end point in the up_label to delete
    new_label = np.array(new_label)
    rm_st = bisect(new_label[:-7, 0], pause_head_st) -1
    # when same num
    #rm_end = bisect(new_label[:-7, 1], pause_tail_end - move) -1 
    #move_pos = bisect(new_label[:,0], pause_tail_end-move) - 1
    rm_end =move_pos 
    #rm_end = min([bisect(new_label[:-7, 0], pause_tail_end)]\
    #        + np.where(new_label[:,1] == pause_tail_end-move)[0].tolist()) #pause_tail_end == new_label[-1,1]

    #rm_st = np.where(new_label[:,0] == pause_head_st)[0][0]
    ##must after append, before sort. min(last_pos, not last position.)
    ## added new part is in the last -1 pos. 
    #rm_end = np.where(new_label[:,1] == new_label[-1,1]-move)[0][0]

    new_label = list(new_label)
    new_label=np.delete(new_label, np.arange(rm_st, rm_end), axis=0)

    #up_label = np.array(up_label)
    #rm_st = np.where(up_label[:,0] == pause_head_st)[0][0]
    #rm_end = np.where(up_label[:,1] == up_label[-1,1]-move)[0][0]
    #up_label = list(up_label)
    up_label=np.delete(up_label, np.arange(max(0, ins_pos-1), min(ins_pos+1, len_label-1)+1), axis=0)

    #if (ins_pos==0): # already delet 0 
    #    up_label=np.delete(up_label, [ins_pos, ins_pos+1], axis=0)
    #    new_label=np.delete(new_label, [ins_pos, ins_pos+1], axis=0)
    #elif (ins_pos >= len_label-1): 
    #    up_label=np.delete(up_label, [ins_pos-1, ins_pos], axis=0)
    #    new_label=np.delete(new_label, [ins_pos-1, ins_pos], axis=0)
    #else:
    #    up_label=np.delete(up_label, [ins_pos-1, ins_pos, ins_pos+1], axis=0)
    #    new_label=np.delete(new_label, [ins_pos-1, ins_pos, ins_pos+1], axis=0)

    ###delete the segments with length=0
    up_label = np.delete(up_label, np.where(up_label[:,0]==up_label[:,1]), axis=0)
    new_label = np.delete(new_label, np.where(new_label[:,0]==new_label[:,1]), axis=0)

    ### Check continue label in vad.
        
    #4.5 sort list
    up_label.astype(float)
    up_label = sorted(up_label,key=(lambda x:x[0])) #array -> list
    new_label.astype(float)
    new_label = sorted(new_label,key=(lambda x:x[0])) #array -> list



    #5. logging for replace. in point    
    con_ID=args.prefix_name +'_{:07d}'.format(COUNTER)
    insert_info=[con_ID, ori_uttid, int(ori_st), int(ori_end),
            insf_spkvad[insf_idx,0], int(insf_st), int(insf_end)] 

    print("finish {}: replace {} [{}: {}], by {} [{}: {}]".format(
        con_ID, ori_uttid, int(ori_st), int(ori_end), 
        insf_spkvad[insf_idx,0], int(insf_st), int(insf_end)))
    insert_string=" ".join([str(x) for x in insert_info])
    logging.info(insert_info)
    insert_logID.write(insert_string+'\n')
    
    #6. set label for used spoof segment to avoid use again.
    if(args.allow_reuse):
        pass
    else:
        insf_spkvad[insf_idx,3] = '0'

    return con_speech, insf_spkvad, up_label, new_label


def mycorrelate(a, b):
    '''correlation or convolution in 1-d array with real numbers'''
    if a is None or b is None:
        return None
### noneed
##     if len(a) > len(b):# Ensure the length of a is no longer than that of b.
##         return mycorrelate_func(b, a, mode)
##
##    # Convert to np.array type
##     a, b = list(map(np.array, [a, b]))
    
    res = []
    lena, lenb = len(a), len(b)
    
    output_length = lena
    tmpa=np.hstack((a, np.zeros(lenb-1)))
    
    range_st = max(lena-lenb, 0)
    for i in range(range_st, output_length):
        val = np.sum(tmpa[i:lenb+i] * b)
        res.append(val)  
      
    return np.argmax(res) + range_st


def conwav_tail_head(x, y, max_id):
    lenx = len(x)
    leny = len(y)
    tolen= max(max_id + leny, lenx)  #max_id + leny: b is longer than the mix part, lenx: b is shorter than the mix part
    
    con_speech=np.zeros(tolen)
    con_speech[:lenx] += x
    con_speech[max_id:max_id+leny] += y
    
    mix_beg = max_id
    mix_end = min(max_id + leny, lenx)
    con_speech[mix_beg : mix_end] = con_speech[mix_beg : mix_end]/2  
    return mix_beg, mix_end, con_speech


def concatenate(ori_sig, st, end, ori_overlap, insf_seg, insf_overlap):

    st= int(st)
    end=int(end)
    #concatenate begin
    if (ori_overlap[0]==0 or insf_overlap[0]==0):
        beg_max_id=st
    else:
        ori_pause_beg = ori_sig[st - ori_overlap[0] : st + 1]
        insf_pause_beg = insf_seg[0:insf_overlap[0]]
        beg_max_id = mycorrelate(ori_pause_beg, insf_pause_beg) + st - ori_overlap[0]    
    head_mix_beg, head_mix_end, be_concat = conwav_tail_head(ori_sig[:st], insf_seg, beg_max_id)
        
    #concatenate end
    if(ori_overlap[1]==0 or insf_overlap[1]==0):
        end_max_id = len(be_concat)
    else:
        ori_pause_end = ori_sig[end : end + ori_overlap[1] ]
        insf_pause_end = insf_seg[-insf_overlap[1]:]
        end_max_id = len(be_concat) - insf_overlap[1] + mycorrelate(insf_pause_end, ori_pause_end ) 
    tail_mix_beg, tail_mix_end, be_end_concate = conwav_tail_head(be_concat, ori_sig[end:], end_max_id)

    return beg_max_id, end_max_id, be_end_concate 



def main():
    global COUNTER
    #for ori_idx, (ori_uttid, vad_dir) in tqdm(enumerate(ori_utt2vad.items())): # iterate ori vad
    for ori_uttid, vad_dir in tqdm(ori_utt2vad.items()): # iterate ori vad
        #if(ori_idx == 233):
        #    print("debug")
        print("start")
        print("original uttid, vad_dir:",ori_uttid, vad_dir)
        
        #1. read vad file for original speech
        orivad = get_vad_by_uttid(ori_uttid)

        orivad=orivad.reshape(-1,3)
        
        #2. select random i segment from spoof segment insert into original speech.
        #ori_spe_idx: original speech index
        #ins_num: how many segment will insert into speech
        #ori_ins_idx: index of original speech to be replaced.

        #2.0 skip those wav only contain silence
        if(len(np.where(orivad[:,2]!=0))==0):
            print('Warn: %s only contain silence, so will skip it.' %(uttid))
            continue

        #2.1 find spoof segment by spkid
        spkid = utt2spk[ori_uttid]

        if(spkid not in insf_spk2vad.keys() ):
            print('Warn: Cannot find speaker %s for %s' %(spkid, ori_uttid))
            continue

        insf_spkvad_file=insf_spk2vad[spkid]
        fileID=open(insf_spkvad_file,'r')
        insf_spkvad=np.loadtxt(fileID, delimiter =' ', dtype=str)
        fileID.close()

        #check wether we have enough segments 
        spf_spe_idx_list = (np.asarray(np.where((insf_spkvad[:,3] != '0.0') & (insf_spkvad[:,3] != '0')))).flatten()
        
        if(len(spf_spe_idx_list)==0 ):
            print('Warn: Cannot find avilible insert_from segments from speaker %s ' %(spkid))
            continue

        #2.2 random i
        ori_spe_idx=(np.asarray(np.where((orivad[:,2]!=0) & (orivad[:,2]!=0.0 )))).flatten()
        if(len(ori_spe_idx) <= 0 ):    
            print("Warn: No segments can be replaced from original %s" %uttid)
            continue

        #random.seed(args.rand_seed)
        random.seed(COUNTER)
        #if(args.insert_label_type == "gen2spf" or args.insert_label_type == "spf2gen" ) : 
        #if(args.insert_label_type == "spf2gen" ) : 

        if(args.must_insert):
            #For spfcon proj, we need to make sure at least one insert for spf2gen. 
            #https://docs.python.org/3/library/random.html#random.randint 
            #Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1).
            ins_num=random.randint(1,min(len(ori_spe_idx), len(spf_spe_idx_list)))
        else:
            ins_num=random.randint(0,min(len(ori_spe_idx), len(spf_spe_idx_list)))

        random.seed(COUNTER)
        ori_ins_idx = random.sample(list(ori_spe_idx), ins_num)
        ori_ins_idx = np.flip(np.sort(ori_ins_idx))
        print("ori_ins_idx:",ori_ins_idx) 


        #2.3 read original original speech
        ori_sr, ori_wav = sciwav.read(utt2wav[ori_uttid])

        #2.4 create the label matrix for speech/non-speech label of original speech, convert ms to point
        new_label = np.hstack((orivad[:,0:2] * ori_sr / 1000.0, orivad[:,2].reshape(-1,1)))  
        print("spkid",spkid)

        #2.5 concatenate
        # up_label is used to insert spoof, also the 1st version of label
        # since we need to use half silence to do replace. 
        up_label = new_label.copy()
        for ins_pos in ori_ins_idx:

            print("one ins_pos:",ins_pos)
            ori_wav, insf_spkvad, up_label, new_label = whole_concatenate(args, ori_uttid, ori_wav, ori_sr, up_label, new_label, ins_pos, 
                                                            insf_spkvad, args.overlap_rate)

        con_ID=args.prefix_name +'_{:07d}'.format(COUNTER)

        # if we don't insert segment, we will use the original one.    
        if(len(ori_ins_idx) == 0):
            #2.5+1. logging for replace. in point    
            insert_info=[con_ID, ori_uttid, 0, len(ori_wav), ori_uttid, 0, len(ori_wav)]

            print("finish {}: using original one {}".format(
                con_ID, ori_uttid))
            insert_string=" ".join([str(x) for x in insert_info])
            logging.info(insert_info)
            insert_logID.write(insert_string+'\n')
            

        #3.1 save wav
        con_name=os.path.join(args.concatenate_wav_dir, con_ID)
        if(args.save_wav):
            sciwav.write(con_name+'.wav', ori_sr, ori_wav.astype(np.int16))

        #3.2 save new vad
        #3.2.1 delete the 0 part, generate in whole_concatenate
        new_label=np.delete(new_label, np.where([x[0]==x[1] for x in new_label]), axis=0)
        new_label = check_con_label(new_label)

        #3.2.2 convert point to s and save
        new_label = np.hstack((new_label[:,0:2] / float(ori_sr) , new_label[:,2].reshape(-1,1))) 
        assert(sum(new_label[1:,0] - new_label[:-1, 1]) == 0)

        new_label_file=con_name + '.vad'
        np.savetxt(new_label_file,new_label,fmt='%s') and print("save new_label for %s" % new_label_file)
        COUNTER=COUNTER+1
            
    insert_logID.close()            
    sys.exit(COUNTER)
    
    


def test_concatenate():
  
    ori_uttid = 'LA_D_1026868'
    vad_dir=ori_utt2vad[ori_uttid]
    
    #1. read vad file for original speech
    fileID=open(vad_dir,'r')
    orivad=np.loadtxt(fileID, delimiter =' ', dtype=np.float)
    fileID.close()
    orivad=orivad.reshape(-1,3)

    if(len(np.where(orivad[:,2]!=0))==0):
        return

    #2.1 random i
    ori_spe_idx=(np.asarray(np.where(orivad[:,2]!=0))).flatten()
    random.seed(18)
    ins_num=random.randint(1,len(ori_spe_idx))

    random.seed(18)
    ori_ins_idx = random.sample(list(ori_spe_idx), ins_num)
    print(ori_ins_idx) 

    #2.2 find spoof segment by spkid
    spkid = utt2spk[ori_uttid]
    if(spkid not in insf_spk2vad.keys() ):
        return
    insf_spkvad_file=insf_spk2vad[spkid]
    fileID=open(insf_spkvad_file,'r')
    insf_spkvad=np.loadtxt(fileID, delimiter =' ', dtype=str)
    fileID.close()

    #2.3 read original original speech
    ori_sr, ori_wav = sciwav.read(utt2wav[ori_uttid])

    #2.4 create the label matrix for speech/non-speech label of original speech, convert ms to point
    new_label = np.hstack((orivad[:,0:2] * ori_sr / 1000, orivad[:,2].reshape(-1,1)))  

    #2.5 concatenate
    for ins_pos in ori_ins_idx:
        ori_wav, insf_spkvad, new_label = whole_concatenate(args, ori_uttid, ori_wav, ori_sr, new_label, ins_pos, 
                                                        insf_spkvad, args.overlap_rate)
    

    #3.1 save wav
    con_name='test.wav'
    sciwav.write(con_name, ori_sr, ori_wav.astype(np.int16))

    #3.2 save new vad
    #3.2.1 delete the 0 part, generate in whole_concatenate
    new_label=np.delete(new_label, np.where([x[0]==x[1] for x in new_label]), axis=0)
    #3.2.2 convert point to s and save
    new_label = np.hstack((new_label[:,0:2] / ori_sr , new_label[:,2].reshape(-1,1))) 
    new_label_file='test.vad'
    np.savetxt(new_label_file,new_label,fmt='%s')
        
        

    
#     ori_sr, ori_wav = sciwav.read(utt2wav[ori_uttid])

#     vad_dir='/home/smg/zhanglin/workspace/00ad/exp/vad_stand_ms/LA_T_1138215.vad'
#     #1. read vad file for original speech
#     fileID=open(vad_dir,'r')
#     orivad=np.loadtxt(fileID, delimiter =' ', dtype=np.float)
#     fileID.close()
#     new_label = np.hstack((orivad[:,0:2] * ori_sr / 1000, orivad[:,2].reshape(-1,1)))  
#     ins_pos=7
#     spkid = utt2spk[ori_uttid]
#     insf_spkvad_file=os.path.join(args.spoof_person_dir,spkid + '.personvad')
#     fileID=open(insf_spkvad_file,'r')
#     insf_spkvad=np.loadtxt(fileID, delimiter =' ', dtype=str)
#     fileID.close()
    
    
    
def debug():
    #ori_uttid='LA_T_1007571'
    #ori_ins_idx=[5]
    #COUNTER=1
    ori_uttid='LA_E_1002903'
    ori_ins_idx=[1]
    COUNTER=0


    vad_dir = ori_utt2vad[ori_uttid] 
    fileID=open(vad_dir,'r')
    orivad=np.loadtxt(fileID, delimiter =' ', dtype=np.float)
    fileID.close()

    ori_sr, ori_wav = sciwav.read(utt2wav[ori_uttid])

    new_label = np.hstack((orivad[:,0:2] * ori_sr / 1000, orivad[:,2].reshape(-1,1)))  

    orivad=orivad.reshape(-1,3)

    spkid = utt2spk[ori_uttid]
    if(spkid not in insf_spk2vad.keys() ):
        return
    insf_spkvad_file=insf_spk2vad[spkid]
    fileID=open(insf_spkvad_file,'r')
    insf_spkvad=np.loadtxt(fileID, delimiter =' ', dtype=str)
    fileID.close()

    for ins_pos in ori_ins_idx:
        whole_concatenate(args, ori_uttid, ori_wav, ori_sr, new_label, ins_pos, 
                          insf_spkvad, overlap_rate = 0.5)
        

if __name__ == '__main__':
    main()
    #debug()
