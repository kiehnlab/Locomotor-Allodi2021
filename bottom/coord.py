import pandas as pd
import numpy as np
import glob
import pdb
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
from utils.tools import videoMetadata
from utils.constants import *
from scipy.stats import circmean
from matplotlib.gridspec import GridSpec
import matplotlib
from scipy.signal import find_peaks
from utils.constants import bot_model as model
from utils.constants import bot_mrkr as mrkr
from utils.tools import iqrMean,measureCycles,circular_mean

colors = ['salmon','mediumorchid','deepskyblue']

sod_colors = ['salmon','salmon','black']
days = np.array(['baseline','fasted','recovery'])
groups = np.array(['controls','fasted'])

sod_fill = [True,False,True]
cno_colors = ['black','black','deepskyblue','deepskyblue']
cno_fill = [True,False,True,False]
sodpre_colors = ['mediumorchid','mediumorchid','black']
sodpre_fill = [True,False,True]
sodcno_colors = ['darkorange','darkorange','mediumorchid','mediumorchid']
sodcno_fill = [True,False,True,False]

fillstyles = ['none','full']
params = {'font.size': 14,
          'font.sans-serif': 'Arial',
          'font.weight': 'bold',
          'axes.labelsize':14,
          'axes.titlesize':14,
          'axes.labelweight':'bold',
          'axes.titleweight':'bold',
          'legend.fontsize': 12,
         }
matplotlib.rcParams.update(params)

def limbCoord(str_0,str_1,movDur):

        str_0_mean = iqrMean(str_0)
        str_1_mean = iqrMean(str_1)

        sMean = iqrMean(str_0) - iqrMean(str_1)
        relStride = str_0 - str_1
        T = len(relStride)
        xAxis = np.linspace(0,movDur,T)
        phi, R, meanPhi, nSteps = heurCircular(xAxis,relStride,sMean)

        return phi, R, meanPhi*180/np.pi, nSteps

def meanVector(data,files,fig,cScheme,gName,key='h',scatter=False):
    pKey = 'phi_'+key
    rKey = 'R_'+key
    N = len(data[pKey])
    groupPhi = np.zeros(N)
    groupR = np.zeros(N)
    files = np.array([f.split('/')[-1][:20] for f in files])
    uniq = np.unique(files)
    ax = fig.add_subplot(111,polar=True)
    ax.set_rlim(0,1.1)
    ax.spines['polar'].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_theta_offset(np.pi/2)
    ax.grid(linewidth=2)

    lColor = colors[np.where(days==gName.split('_')[0])[0][0]]
    lFill = fillstyles[np.where(groups==gName.split('_')[1])[0][0]]


    for i in range(N): 
        phi = data[pKey][i]
        r = data[rKey][i]
        groupPhi[i],groupR[i] = circular_mean(phi)

    for i in range(len(uniq)):
        animPhi,animR = circular_mean(groupPhi[files==uniq[i]])
        if scatter:
            if lFill == 'full':
                ax.plot((0,animPhi),(0,animR),color=lColor,marker='o',
                        fillstyle=lFill,linestyle='', markeredgecolor='white', markersize=12) 
            else:
                ax.plot((0,animPhi),(0,animR),color=lColor,marker='o',
                        fillstyle=lFill,linestyle='', markersize=12) 

        else:
            ax.annotate("",xytext=(0.0,0.0),xy=(animPhi,animR*1.05),
                   arrowprops=dict(color='dimgray',width=0.2,lw=2))

    meanPhi, meanR = circular_mean(groupPhi)

    ax.annotate("",xytext=(0.0,0.0),xy=(meanPhi,meanR*1.05),
               arrowprops=dict(color=lColor,lw=3,
                               width=0.75,fill=(lFill=='full')))
    return fig, meanPhi, meanR

def heurCircular(xAxis,stride,sMean,idxFlag=False):
    newIdx = measureCycles(stride)[1]
    stride = 2*(stride-stride.min())/ (stride.max()-stride.min()) - 1
    N = len(newIdx)
    phi = np.zeros(N-1)

    for i in range(N-1):

        lIdx = newIdx[i]
        uIdx = newIdx[i+1]
        y = stride[lIdx:uIdx]
        x = np.linspace(0,2*np.pi,len(y))
        phi[i] = ((4-np.trapz(y,x)) * np.pi/4) #% 2*np.pi

    meanPhi,r = circular_mean(phi)
    if idxFlag: 
        return phi, r, meanPhi, N, idx
    else:
        return phi, r, meanPhi, N


def bodyPosCoord(ipFile,speedMean,avgSpeed, meta):

    data = pd.read_hdf(ipFile)
    
    fL = np.asarray(data[model][mrkr[3]]['x'])
    fL = np.convolve(fL, np.ones((smFactor,))/smFactor, mode='valid')
    fR = np.asarray(data[model][mrkr[5]]['x'])
    fR = np.convolve(fR, np.ones((smFactor,))/smFactor, mode='valid')
    hL = np.asarray(data[model][mrkr[4]]['x'])
    hL = np.convolve(hL, np.ones((smFactor,))/smFactor, mode='valid')
    hR = np.asarray(data[model][mrkr[6]]['x'])
    hR = np.convolve(hR, np.ones((smFactor,))/smFactor, mode='valid')
    
    bodyPos = np.asarray([data[model][speedMarkers[i]]['x'] 
                          for i in range(len(speedMarkers))]).mean(0)
    torsoPos = np.asarray([data[model][speedMarkers[i]]['x'] 
                          for i in range(4,len(speedMarkers))]).mean(0)

    bodyLen = (-np.asarray([data[model][speedMarkers[i]]['x'] 
                          for i in range(1,4)]).mean(0) +
              np.asarray(data[model][speedMarkers[0]]['x'])) * meta['xPixW']/10
    ## Sort bodylength in decreasing order and use first 10% of estimates
    bodyLen = np.sort(bodyLen)[::-1][:int(0.1*meta['nFrame'])]
    bodyLen = iqrMean(bodyLen)
    partition = int(meta['imW']/2)
    intervals = np.arange(3) * partition 
    locHist = plt.hist(torsoPos,bins=intervals)[0]/meta['nFrame']
    bodyPos = bodyPos[smFactor-1:]
    # Compute the distance between right and left paws
    hLStride = (bodyPos-hL) * meta['xPixW']
    hLStride = np.convolve(hLStride, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')
    hRStride = (bodyPos-hR) * meta['xPixW']
    hRStride = np.convolve(hRStride, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')
    fLStride = (fL-bodyPos) * meta['xPixW']
    fLStride = np.convolve(fLStride, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')
    fRStride = (fR-bodyPos) * meta['xPixW']
    fRStride = np.convolve(fRStride, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')

    # Exclude strides when animal is not moving speed = 0
    idx = ((speedMean) > 0)
    hLStride = hLStride[:len(idx)][idx]
    hRStride = hRStride[:len(idx)][idx]
    fLStride = fLStride[:len(idx)][idx]
    fRStride = fRStride[:len(idx)][idx]
    
    # Compute distance between hind limbs (width)
    hWidth = np.abs(hL-hR) * meta['xPixW']
    hWidth = np.convolve(hWidth, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')
    hWidth = hWidth[:len(idx)][idx]
    hWidth = hWidth.mean()

    # compute avg speed based on non-dragging portions
    avgSpeed = speedMean[idx][::int(meta['fps']/speedSmFactor)].mean()

    # Duration of movement
    movDur = np.sum(idx == True)/len(speedMean) * meta['dur']    

    # Check frames where animal is not moving

    xAxis = np.linspace(0,movDur,len(hLStride))
    xAxisNew = np.linspace(0,movDur,INTERP*len(hLStride))

    # Interpolate for better zero crossing detections
    hLStride = np.interp(xAxisNew, xAxis, hLStride)
    fLStride = np.interp(xAxisNew, xAxis, fLStride)
    hRStride = np.interp(xAxisNew, xAxis, hRStride)
    fRStride = np.interp(xAxisNew, xAxis, fRStride)

    # Count the number of zero crossings for the entire duration
    # Obtain cadence steps/second 
    hLCadence = (measureCycles(hLStride))[0]/movDur
    fLCadence = (measureCycles(fLStride))[0]/movDur
    hRCadence = (measureCycles(hRStride))[0]/movDur
    fRCadence = (measureCycles(fRStride))[0]/movDur

    # Estimate average stride length using cadence and average speed
    hLStep = avgSpeed/hLCadence
    fLStep = avgSpeed/fLCadence
    hRStep = avgSpeed/hRCadence
    fRStep = avgSpeed/fRCadence

    return (hLCadence, hRCadence, fLCadence, fRCadence),\
            (hLStride, hRStride, fLStride, fRStride),\
            (hLStep, hRStep, fLStep, fRStep), movDur, bodyLen, locHist, hWidth



