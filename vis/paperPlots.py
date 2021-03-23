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
from bottom.coord import bodyPosCoord, bodyCoordCircular, circularPlot, cadencePlot
############## Speed analysis ##########
    
def processDict(dFiles):
    allData = allData=dict.fromkeys(keys,[])

    for dFname in dFiles:
        data = np.load(dFname,allow_pickle=True).item()
        for key,value in data.items():
            allData[key] = allData[key]+[value]

#    for k in keys:
#        allData[k] = np.hstack(allData[k])
    return allData

def combinedPlot(data_path,saveFlag=False):
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
        gs = GridSpec(2,2,figure=fig)

        for j in range(len(dataFiles)):

            fig = circularPlot(allData['phi'][j], allData['R'][j], fig, gs,
                               gsNum=0,vNum=j)
            fig = circularPlot(allData['phi_h'][j], allData['R_h'][j], fig, 
                               gs,gsNum=1,vNum=j)
            
        ax = fig.add_subplot(gs[0,:])
#        plt.title('Subject '+f+' .\n Left Cadence: %.2f Hz, Right Cadence: %.2f Hz, nSteps: %d\
#                  \n Using Avg. speed %.2f cm/s, Avg. left stride: %.2f cm, Avg. right stride: %.2f cm'
#           %(np.mean(allData['lCad']),np.mean(allData['rCad']),
#             sum(allData['nSteps']),np.mean(allData['avg']),
#             np.mean(allData['lStLen']),np.mean(allData['rStLen'])))
        for k in keys:
            allData[k] = np.hstack(allData[k])

        fig = circularPlot([circmean(allData['phi'])], 
                           np.mean(allData['R']), fig, gs,gsNum=0,vNum=-1)

        fig = circularPlot([circmean(allData['phi_h'])],
                            np.mean(allData['R_h']), fig, gs, gsNum=1, vNum=-1)

#        pdb.set_trace()
        fig = cadencePlot(sum(allData['movDur']), allData['lStride'], 
                             allData['rStride'], fig, gs,circPlot=False)
#        plt.legend()
        plt.savefig(f+'.pdf')
