# coding: utf-8

"""Module for extracting phase features
"""
import argparse
import os
import soundfile as sf  
import numpy as np
from scipy.fftpack import dct
import scipy.io.wavfile as wav
from psf_zl.sigproc import preemphasis, framesig

#from plot import plot_data


NFFT = 1024
PREEMPH = 0.97
HAMMING_WINFUNC = np.hamming
LIFTER = 6
ALPHA = 0.4
GAMMA = 0.9


def get_complex_spec(sig, rate, winstep, winlen, nfft=NFFT, with_time_scaled=False):
    """Return complex spec
    """
#    rate, sig = wav.read(wav_)
#    sig, rate = sf.read(wav_)

    sig = preemphasis(sig, PREEMPH)
    frames = framesig(sig, winlen * rate, winstep * rate, HAMMING_WINFUNC)
    complex_spec = np.fft.rfft(frames, nfft)

    time_scaled_complex_spec = None
    if with_time_scaled:
        time_scaled_frames = np.arange(frames.shape[-1]) * frames
        time_scaled_complex_spec = np.fft.rfft(time_scaled_frames, nfft)

    return complex_spec, time_scaled_complex_spec


def get_mag_spec(complex_spec):
    """Return mag spec
    """
    return np.absolute(complex_spec)


def get_phase_spec(complex_spec):
    """Return phase spec
    """
    return np.angle(complex_spec)


def get_real_spec(complex_spec):
    """Return real spec
    """
    return np.real(complex_spec)


def get_imag_spec(complex_spec):
    """Return imag spec
    """
    return np.imag(complex_spec)


def cepstrally_smoothing(spec):
    """Return cepstrally smoothed spec
    """
    _spec = np.where(spec == 0, np.finfo(float).eps, spec)
    log_spec = np.log(_spec)
    ceps = np.fft.irfft(log_spec, NFFT)
    win = (np.arange(ceps.shape[-1]) < LIFTER).astype(np.float)
    win[LIFTER] = 0.5
    return np.absolute(np.fft.rfft(ceps * win, NFFT))


def get_modgdf(complex_spec, complex_spec_time_scaled):
    """Get Modified Group-Delay Feature
    """
    mag_spec = get_mag_spec(complex_spec)
#    cepstrally_smoothed_mag_spec = cepstrally_smoothing(mag_spec)
#    plot_data(cepstrally_smoothed_mag_spec,
#              "cepstrally_smoothed_mag_spec.png",
#              "cepstrally_smoothed_mag_spec")

    real_spec = get_real_spec(complex_spec)
    imag_spec = get_imag_spec(complex_spec)
    real_spec_time_scaled = get_real_spec(complex_spec_time_scaled)
    imag_spec_time_scaled = get_imag_spec(complex_spec_time_scaled)

    __divided = real_spec * real_spec_time_scaled  + imag_spec * imag_spec_time_scaled
    mag_spec = np.where(mag_spec == 0, np.finfo(float).eps, mag_spec)
    #print(mag_spec**2)
    __tao = __divided / (mag_spec ** 2)
#    __tao = __divided / (cepstrally_smoothed_mag_spec ** (2. * GAMMA))
#    __abs_tao = np.absolute(__tao)
#    __sign = 2. * (__tao == __abs_tao).astype(np.float) - 1.
#    return dct(__sign * (__abs_tao ** ALPHA), type=2, axis=1, norm='ortho')
    return np.transpose(__tao)


def cmvn(spectrogram):
    '''Cepstral Mean and Variance Normalization
    '''
    mu = np.mean(spectrogram, axis=1)
    stdev = np.std(spectrogram, axis=1)
    return (spectrogram - mu.reshape((-1, 1))) / stdev.reshape((-1, 1))


# In[4]:


def main():
    """Main
    """
    pass
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--wav", default='/NASdata/caiwch/asvspoof2019/data/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac')
#    parser.add_argument("--winstep", type=float, default=0.01)
#    parser.add_argument("--winlen", type=float, default=0.025)
#    parser.add_argument("--debug", type=bool, default=False)

#    args = parser.parse_args()
#     winstep = 0.01
#     winlen = 0.025
#     
#     outdir = 'eval_modgdf_la'
#     with open('index/evaldata_la.list') as fa:
#         for line in fa.readlines():
#             line = line.strip()
#             complex_spec, complex_spec_time_scaled = get_complex_spec(
#                 line, winstep,
#                 winlen, with_time_scaled=True)
# 
#             modgdf = get_modgdf(complex_spec, complex_spec_time_scaled)
#             modgdf = cmvn(modgdf)
#             filepath, tempfilename = os.path.split(line)
#             outname = tempfilename.replace('flac','npy')
#            outname = tempfilename.replace('wav','npy')
#            outputname = outdir + '/' + outname
#            print(modgdf.shape, outputname)
#            np.save(outputname, modgdf)
    #print(modgdf)
    #plot_data(modgdf, "modgdf.png", "modgdf")
    #plot_data(np.absolute(modgdf), "abs_modgdf.png", "abs_modgdf")


if __name__ == "__main__":
    main()

