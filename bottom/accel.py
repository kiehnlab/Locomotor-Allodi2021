import pandas as pd
import numpy as np
import glob
import pyexifinfo as pex
import pdb
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
from utils.tools import videoMetadata
from utils.constants import *

############# Acceleration analysis #######################

def plotDragReco(xAxis,accMean,drgIdx,recIdx,
                 dragCount,recCount,drgDur,recDur,vid,plotSave=True):

    plt.figure(figsize=(16,10))
    plt.plot(xAxis,accMean,color='lightgray',label='Instantaneous Acceleration')
    plt.plot(xAxis,np.zeros(len(xAxis)),'--k')
    plt.xlabel('Duration in s')
    plt.ylabel('Acceleration in cm/s2')
    for i in range((drgIdx.shape[1])):
        sIdx = drgIdx[0,i]
        lIdx = drgIdx[1,i]
        if i == drgIdx.shape[1]-1:
            plt.plot(xAxis[sIdx:lIdx],accMean[sIdx:lIdx],'b',label='Recovery Events')
        else:
            plt.plot(xAxis[sIdx:lIdx],accMean[sIdx:lIdx],'b')
    for i in range((recIdx.shape[1])):
        sIdx = recIdx[0,i]
        lIdx = recIdx[1,i]
        if i == recIdx.shape[1]-1:
            plt.plot(xAxis[sIdx:lIdx],accMean[sIdx:lIdx],'r',label='Drag Events')
        else:
            plt.plot(xAxis[sIdx:lIdx],accMean[sIdx:lIdx],'r')
    plt.legend(loc='upper left')
    plt.title('Subject '+vid.split('/')[1].split('.')[0]+'; Peak Acceleration: %.2f cm/s2; Threshold: %.2f s \n Total drag events: %d, Recovery Events: %d, Ratio: %.3f \n Total drag dur: %.2f, Recovery dur: %.2f, Ratio: %.3f'
              %(accMean.max(),tThr, dragCount,recCount, (1+dragCount)/(1+recCount+1e-3),drgDur,recDur,(1+drgDur)/(1+recDur+1e-3) ))
    if plotSave:
        plt.savefig(acProfLoc+vid.split('.avi')[0].split('..')[1]+'_accelProfile.pdf')
    return

def countDragReco(nAcc,pAcc,xAxis):

    ### Count drag events
    zIdx = np.where(nAcc==0)[0]
    remIdx = []
    
    for i in range(len(zIdx)-1):
        if zIdx[i+1] == (zIdx[i]+1):
            remIdx.append(i+1)
        if len(remIdx) > 0:
            uniqDragIdx = np.delete(zIdx,np.array(remIdx))

    zIdx = np.where(pAcc==0)[0]
    remIdx = []
    for i in range(len(zIdx)-1):
        if zIdx[i+1] == (zIdx[i]+1):
            remIdx.append(i+1)
        if len(remIdx) > 0:
            uniqRecoIdx = np.delete(zIdx,np.array(remIdx))

    if len(uniqDragIdx) > len(uniqRecoIdx):
        uniqRecoIdx=np.concatenate((uniqRecoIdx,[len(xAxis)]))
    elif len(uniqDragIdx) < len(uniqRecoIdx):
        uniqDragIdx=np.concatenate((uniqDragIdx,[len(xAxis)]))

    if uniqRecoIdx[0] < uniqDragIdx[0]:
        recIdx = np.asarray((uniqRecoIdx,uniqDragIdx))
        drgIdx = np.asarray((uniqDragIdx[:-1],uniqRecoIdx[1:]))
    else:
        drgIdx = np.asarray((uniqDragIdx,uniqRecoIdx))
        recIdx = np.asarray((uniqRecoIdx[:-1],uniqDragIdx[1:]))

    return drgIdx, recIdx

def estimateAccel(speedMean, meta):

    accMean = np.diff(speedMean)
    accMean = np.convolve(accMean, np.ones((accSmFactor,))/accSmFactor, mode='valid')
    xAxis = np.linspace(0,meta['dur'],len(accMean))
    pAcc = accMean * (accMean >= 0)
    nAcc = accMean * (accMean < 0)
    
    drgIdx, recIdx = countDragReco(nAcc,pAcc,xAxis)

    return accMean, drgIdx, recIdx, xAxis

def analyseDragRec(drgIdx,recIdx,fps,tThr):

    ### Drag recovery events
    recos = (recIdx[1]-recIdx[0])/fps
    recIdx = recIdx[:,(recos >= tThr)]
    dragCount = recIdx.shape[1] #np.sum(recos >= tThr)
    drgDur = (recos * (recos >= tThr)).sum()
    
    drags = (drgIdx[1]-drgIdx[0])/fps
    drgIdx = drgIdx[:,(drags >= tThr)]
    recCount = drgIdx.shape[1] #np.sum(drags >= tThr)
    recDur = (drags *(drags >= tThr)).sum()

    return dragCount, recCount, drgDur, recDur, drgIdx, recIdx

def accelProfiler(data_path,saveFlag=False,log=False,plotFlag=False):
    eps = 1e-6
    os.chdir(data_path)
    files = sorted(glob.glob('*.npy'))
    tThrRange = np.arange(0.1,0.55,0.05)
    for t in tThrRange:
        if not os.path.exists(acProfLoc):
            os.mkdir(acProfLoc)
            print('Acceleration profiles will be saved in'+acProfLoc)
        else: 
            print('Using existing location to save acceleration profiles at'+acProfLoc)
        if log:
            fName = 'accelProfile_'+repr(t)[:4].replace('.','_')+'.csv'
            with open(fName,'w') as f:
                print('Name\tPeakAcc.\tNum_drag\tNum_rec\tCount_Ratio\tDur_drag\tDur_rec\tDur_ratio',file=f)
        for k in range(len(files)):
            ipFile = files[k]
            vid = '../'+ipFile.split('.')[0].split('_sp')[0]+'.avi'
            meta = videoMetadata(vid)

            # Load speed data
            speedAll = np.load(ipFile)
            speedMean = speedAll.mean(0)
            speedMean = np.convolve(speedMean, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')


            accMean, drgIdx, recIdx, xAxis = estimateAccel(speedMean,meta)

            dragCount, recCount, drgDur, recDur, drgIdx, recIdx = \
                    analyseDragRec(drgIdx,recIdx,meta['fps'], t)

            if plotFlag:
                plotDragReco(xAxis,accMean,drgIdx,recIdx,\
                             dragCount,recCount,drgDur,recDur,vid)

            print('Sample '+ipFile+', Threshold: %.2f s \nTotal drag events: %d, Recovery Events: %d, Ratio: %.3f \
                  \nTotal drag dur: %.2f, Recovery dur: %.2f, Ratio: %.3f \n'\
                  %(tThr, 1+dragCount,1+recCount, (1+dragCount)/(1+recCount),drgDur,recDur,
                    (1+drgDur+eps)/(1+recDur+eps) ))
            if log:
                with open(fName,'a') as f:
                    print(ipFile.split('_speed')[0]+'\t%.4f\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f'
                          %(accMean.max(),1+dragCount,1+recCount,(1+dragCount)/(1+recCount),
                            drgDur,recDur,(t+drgDur)/(t+recDur)),file=f)
            if saveFlag:
                np.savez(acProfLoc+vid.split('.avi')[0].split('..')[1]+'_acProfile.npz',\
                         accMean=accMean,dragCount=dragCount,recCount=recCount,\
                         drgDur=drgDur,recDur=recDur)

