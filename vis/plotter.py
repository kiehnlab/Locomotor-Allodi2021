import pandas as pd
import numpy as np
import glob
import pdb
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import circmean
from utils.tools import videoMetadata
from utils.constants import *
from bottom.accel import *
from utils.tools import *

def groupPlot(phi,r,fig,gNames):

    ax = fig.add_subplot(111,polar=True)
    ax.set_rlim(0,1.1)
    ax.spines['polar'].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_theta_offset(np.pi/2)
    ax.grid(linewidth=3)
#    pdb.set_trace()
    for i in range(len(phi)):
        lColor = colors[np.where(days==gNames[i].split('_')[0])[0][0]]
        lFill = fillstyles[np.where(groups==gNames[i].split('_')[1])[0][0]]
        ax.plot((10,10),color=lColor,label=gNames[i].replace('_',' '),
                linestyle='none',
               marker='>',fillstyle=lFill,markersize=10)
        ax.annotate("",xytext=(0.0,0.0),xy=(phi[i],r[i]*1.05),
                   arrowprops=dict(color=lColor,width=0.2,lw=3,fill=(lFill=='full')))
    meanPhi = circmean(phi)
    meanR = np.mean(r)
    return fig, meanPhi, meanR

def circularPlot(phi, pairName, fig, gs,row=0,col=0):
    meanPhi,r = circular_mean(phi)
    ax = fig.add_subplot(gs[row,col],polar=True)
    plt.title(pairName)
    ax.set_rlim(0,1.1)
    ax.spines['polar'].set_visible(False)
    ax.set_axisbelow(True)
#    pdb.set_trace()
    ax.set_theta_offset(np.pi/2)
    T = (len(phi) if len(phi) > 1 else 1)

    ax.scatter(phi,np.ones(T),marker='o',s=12,c='tab:grey') #,label='Individual steps')
    ax.annotate("",xytext=(0.0,0.0),xy=(meanPhi,r),
            arrowprops=dict(color='tab:red',lw=3,width=0.5,fill='full'))
    ax.plot((0,meanPhi),(0,r),color='tab:red') #label=legends[vNum],color=colors[vNum])
#    plt.legend() 
    return fig

def cadencePlot(movDur, lStride, rStride, fig, gs,circPlot=True):
    #    plt.box(False)
#    pdb.set_trace()
    lSMean = iqrMean(lStride)
    rSMean = iqrMean(rStride)

    sMean = lSMean - rSMean
    stride = lStride-rStride
    T = len(lStride)
    xAxis = np.linspace(0,movDur,T)

#    phi, R, meanPhi, N = heurCircular(xAxis,stride,sMean)
#    if circPlot:
#       fig = circularPlot(phi,R,fig,gs,1)

    ax = fig.add_subplot(gs[0,:])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(xAxis,stride,label='Relative position of hind limbs')
    plt.xlabel('Duration (s)')
    plt.ylabel('Stride length (mm)')
#    idx = np.where(np.diff(stride > sMean) == True)[0]
    _,idx = measureCycles(stride)
    plt.plot(xAxis[idx],stride[idx],'o',label='Possible full cycle',markersize=4)
    plt.plot(xAxis,np.ones(T)*sMean,'--',color='dimgrey',label='Mean crossing point')
    plt.legend(loc='upper left')
    return fig

############## Speed analysis ##########
    
def plotSpeedProfile(vid, meta, beltSpeed, avgSpeed,
                     speedMean, speedStd, fig, gs):
#    plt.clf()
#    plt.figure(figsize=(16,10))
    ax = fig.add_subplot(gs[0,:])
    newFrame = speedMean.size
    xAxis = np.linspace(0,meta['dur'],newFrame)
    plt.fill_between( xAxis, speedMean-speedStd, speedMean + speedStd,
                                      color='gray', alpha=0.3)
    plt.plot(xAxis,speedMean,label='Avg. Instataneous speed')
    plt.plot(xAxis,beltSpeed/10*np.ones(newFrame),'--',label='Belt Speed')
    plt.plot(xAxis,avgSpeed.mean()*np.ones(newFrame),':',label='Avg. Speed')
    plt.xlabel('Time in s')
    plt.ylabel('Speed in cm/s')
    plt.title('Analysis for '+vid.split('.')[0]+
              '\n Belt Speed: %.2f cm/s \n Avg. Speed: %.2f cm/s'%(beltSpeed/10,avgSpeed))
    plt.legend()
#    if saveFlag:
#   plt.savefig(spProfLoc+vid.split('.avi')[0]+'_speedProfile.pdf')
    return fig

def coordinationPlot(data_path):
    """
    Input: Npz file with all processed data 
    Output: Combined plots for triplicates

    Using the tracks from DeepLabCut estimate the speed of the animal
    and estimate the instantaneous acceleration.
    """
#    pdb.set_trace()
    fig = plt.figure(figsize=(16,10))
    gs = GridSpec(2,2,figure=fig,hspace=0.3)
    nVid = len(dataFiles)

    for j in range(nVid):

        fig = circularPlot(allData['phi_h'][j], allData['R_h'][j], fig, 
                           gs,gsNum=1,vNum=j)
        
    ax = fig.add_subplot(gs[0,:])
    for k in keys:
        allData[k] = np.hstack(allData[k])

    fig = circularPlot([circmean(allData['phi_h'])],np.mean(allData['R_h']), 
                       fig, gs, gsNum=1, vNum=-1)

    fig = cadencePlot(sum(allData['movDur']), allData['lStride'], 
                         allData['rStride'], fig, gs,circPlot=False)
    plt.savefig(f+'.pdf')
    return

def combinedPlot(data_path):
    """
    Input: Npz file with all processed data 
    Output: Combined plots for triplicates

    Using the tracks from DeepLabCut estimate the speed of the animal
    and estimate the instantaneous acceleration.
    """
#    pdb.set_trace()
    os.chdir(data_path)
    files = sorted(glob.glob('*_Profile.npy'))
    uniqF = [f.split('_0deg')[0] for f in files];
    uniqF = list(np.unique(uniqF))

    for f in (uniqF):

        dataFiles = sorted(glob.glob(f+'*_Profile.npy')) 
        allData = processDict(dataFiles)
        fig = plt.figure(figsize=(16,10))
        gs = GridSpec(2,2,figure=fig,hspace=0.3)
        nVid = len(dataFiles)

        for j in range(nVid):

            fig = circularPlot(allData['phi_h'][j], allData['R_h'][j], fig, 
                               gs,gsNum=1,vNum=j)
            
        ax = fig.add_subplot(gs[0,:])
        for k in keys:
            allData[k] = np.hstack(allData[k])

        fig = circularPlot([circmean(allData['phi_h'])],np.mean(allData['R_h']), 
                           fig, gs, gsNum=1, vNum=-1)

        fig = cadencePlot(sum(allData['movDur']), allData['lStride'], 
                             allData['rStride'], fig, gs,circPlot=False)
        plt.savefig(f+'.pdf')
        return

