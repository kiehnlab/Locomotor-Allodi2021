import pandas as pd
import numpy as np
import glob
import pyexifinfo as pex
import pdb
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.constants import *

def circular_mean(phi):
    X = np.cos(phi).mean()
    Y = np.sin(phi).mean()
    meanR = np.sqrt(X**2+Y**2)
    meanPhi = np.arctan2(Y,X)
    return meanPhi, meanR

def iqrMean(data):
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)
    IQR = upper_quartile-lower_quartile
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    result = data[np.where((data >= quartileSet[0]) & (data <= quartileSet[1]))]

    return result.mean()

def measureCycles(stride): 
    peaks,_ = find_peaks(stride) 
    thresh = np.diff(peaks).mean()/2 
    peaks,_ = find_peaks(stride,distance=thresh) 
    return len(peaks),peaks 

def processDict(dFiles):
    allData = allData=dict.fromkeys(keys,[])

    for dFname in dFiles:
        data = np.load(dFname,allow_pickle=True).item()
        for key,value in data.items():
            allData[key] = allData[key]+[value]

#    for k in keys:
#        allData[k] = np.hstack(allData[k])
    return allData

def videoMetadata(vid):
    meta = {}
    meta['dur'] = np.float(pex.information(vid)['Composite:Duration'].replace(' s',''))
    meta['fps'] = np.int(pex.information(vid)['RIFF:VideoFrameRate'])
    meta['nFrame'] = nFrame = np.int(pex.information(vid)['RIFF:VideoFrameCount'])
    meta['imW']  = np.int(pex.information(vid)['RIFF:ImageWidth']) 
    meta['imH'] = np.int(pex.information(vid)['RIFF:ImageHeight'])
    meta['xPixW'] = length/meta['imW']

    print('Video of duration %.2f s with total %d frames, at %d fps and image size of %d x %d; each pixel is %.4f mm wide' \
          %(meta['dur'],meta['nFrame'],meta['fps'],meta['imW'],meta['imH'], meta['xPixW']))
    return meta

