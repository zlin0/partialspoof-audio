from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
from scipy.fftpack import dct
import PIL.Image as Image
import numpy as np
import cv2
import os

'''MGDCC   tdk'''
'''MGDF    whw'''
def MGDCC(X,Y,gamma,alpha,numcep=20):
    result = []
    for index in range(0,len(X)):
        power = np.abs(X[index])
        mgd = []
        for i in range(0,len(X[index])):
            S = cv2.dct(power[i])[0:30]
            s_power = cv2.idct(S)
            if np.abs(s_power[0] - 0) <= 0.0001:
                mgd.append(0)
            else:
                tame = (X[index][i].real*Y[index][i].real+X[index][i].imag*Y[index][i].imag)/pow(s_power[0],gamma)
                sgn = tame/np.abs(tame)
                mgd.append(sgn*pow(np.abs(tame),alpha))
        result.append(mgd)
    result = np.array(result)
    # dct
    result = dct(result, type=2, axis=1, norm='ortho')[:,:numcep]
    result_size = result.shape
    result = result.reshape(result_size[0], result_size[1])
    return (np.array(result))

def stft_x_nx(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(int(frameSize)))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), int(frameSize)), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    rescaled = []
    for i in frames:
        temp = []
        for j in range(0,len(i)):
            temp.append(i[j]*j)
        rescaled.append(temp)
    rescaled = np.array(rescaled)
    frames *= win
    rescaled *= win
    return (np.fft.rfft(frames),np.fft.rfft(rescaled))    

def mgdf_feat(samples):
    binsize = 2 ** 9
    gamma = 0.9
    alpha = 0.4
    X,Y = (stft_x_nx(samples,binsize))
    mgdcc_data = MGDCC(X,Y,gamma,alpha)
    # norm = []
    # for i in mgdcc_data:
    #     norm.append((i-i.mean())/i.std())
    return np.array(mgdcc_data)


# wavfile = "./100000.wav"
# feats = mgdf_feat(wavfile)
# feats.shape