import pandas as pd
import numpy as np
import glob
import pdb
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils.tools import videoMetadata
from utils.constants import *
from bottom.accel import *
from bottom.coord import *
from vis.plotter import plotSpeedProfile,circularPlot
from utils.constants import bot_model as model
from utils.constants import df_cols
############## Speed analysis ##########

def estimateSpeed(ipFile,beltSpeed,meta,vid):
#    model = ipFile.split('cms')[1].split('.')[0]
    data = pd.read_hdf(ipFile)
    time = 1/meta['fps'] # interval between successive frames in s
    speedAll = []

    for i in range(len(speedMarkers)):

        m0 = data[model][speedMarkers[i]]
        m0X = np.array(m0['x']) * meta['xPixW']
        m0XSmoothed = np.convolve(m0X, np.ones((smFactor,))/smFactor, mode='valid')
        speed = (-np.diff(m0XSmoothed)/time + beltSpeed)
        
        #Remove outliers
        speed[np.abs(speed) > 3*beltSpeed] = beltSpeed
        speed /= 10 # Convert to cm/s
        speedAll.append(speed)

    speedAll = np.stack(speedAll)

    speedMean = speedAll.mean(0)
    speedMean = np.convolve(speedMean, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')
    avgSpeed = speedMean[::int(meta['fps']/speedSmFactor)].mean()
    print("Avg. Speed: %.2f cm/s"%(avgSpeed))

    speedStd = speedAll.std(0)
    speedStd = np.convolve(speedStd, np.ones((speedSmFactor,))/speedSmFactor, mode='valid')

    return speedAll, speedMean, avgSpeed, speedStd
    
def bottom_profiler(data_path,df):
    """
    Input: Pandas frame with tracks for each marker
    Output: Smoothed speed, acceleration and coordination profiles
    """
    os.chdir(data_path)
    vidFiles = sorted(glob.glob('*.avi'))
    os.chdir(data_path+'labels')
    if not os.path.exists(spProfLoc):
        os.mkdir(spProfLoc)
        print('Speed profiles will be saved in '+spProfLoc)
    else: 
        print('Using existing location to save speed profiles at '+spProfLoc)

    for i in range(len(vidFiles)):

        vid = vidFiles[i]
        ipFile = glob.glob(vid.split('.avi')[0]+'*.h5')[0]
        fName = spProfLoc+vid.split('.avi')[0]
        print("\n Processing tracks for "+vid)

        # Load video metadata
        meta = videoMetadata(data_path+vid)


        # Measure the speed from tracks
        beltSpeed = ipFile.split('cms')[0].split('_')[-1]
        beltSpeed = float(beltSpeed) * 10
        print('Belt speed is : %.2f cm/s'% (beltSpeed/10))

        speedAll,speedMean,avgSpeed,speedStd = estimateSpeed(ipFile,
                beltSpeed, meta, vid)

        accMean, drgIdx, recIdx, xAxis = estimateAccel(speedMean,meta)

        dragCount, recCount, drgDur, recDur, drgIdx, recIdx = \
                analyseDragRec(drgIdx,recIdx,meta['fps'], tThr)
        cadence, stride, stepLen, movDur, \
                bodyLen,locHist,hWidth = bodyPosCoord(ipFile, speedMean, avgSpeed, meta)

        ## Plot speed profile
        fig = plt.figure(figsize=(16,18))
        gs = GridSpec(3,3,figure=fig)
 
        fig = plotSpeedProfile(vid, meta, beltSpeed, avgSpeed, speedMean, speedStd, fig, gs)

        ### Coordination analysis + plots
        ### Coordination of hind limbs (LHRH)
        phi_heur, R_heur, meanPhi_heur, nSteps = limbCoord(stride[0],stride[1],movDur)
        fig = circularPlot(phi_heur,'LHRH',fig,gs,row=1,col=0)

        ### Coordination of homolateral right (RFRH)
        phi_xR, R_xR, meanPhi_xR, nSteps_xR = limbCoord(stride[1],stride[3],movDur)
        fig = circularPlot(phi_xR,'RFRH',fig,gs,row=1,col=1)

        ### Coordination of homolateral left (LFLH)
        phi_xL, R_xL, meanPhi_xL,nSteps_xL = limbCoord(stride[0],stride[2],movDur)
        fig = circularPlot(phi_xL,'LFLH',fig,gs,row=1,col=2)

        ### Diagonal Coordination 1 (LFRH)
        phi_fLhR, R_fLhR, meanPhi_fLhR,nSteps_fLhR = limbCoord(stride[2],stride[1],movDur)
        fig = circularPlot(phi_fLhR,'LFRH',fig,gs,row=2,col=0)

        ### Diagonal Coordination 2 (RFLH) 
        phi_fRhL, R_fRhL, meanPhi_fRhL,nSteps_fRhL = limbCoord(stride[3],stride[0],movDur)
        fig = circularPlot(phi_fRhL,'RFLH',fig,gs,row=2,col=1)

        ### Coordination of fore limbs (LFRF)
        phi_frnt, R_frnt, meanPhi_frnt, nSteps_frnt = limbCoord(stride[2],stride[3],movDur)
        fig = circularPlot(phi_frnt,'LFRF',fig,gs,row=2,col=2)
        plt.savefig(fName+'_bottom.pdf')

        plt.clf()
        fig = plt.figure(figsize=(16,10))
        gs = GridSpec(2,2,figure=fig)
        ax = fig.add_subplot(gs[0,:])
        plt.title('Subject '+vid+' .\n Left Cadence: %.2f Hz, Right Cadence: %.2f Hz, nSteps: %d\
                  \n Using Avg. speed %.2f cm/s, Avg. left stride: %.2f cm, Avg. right stride: %.2f cm'
           %((cadence[0]),(cadence[1]),nSteps,avgSpeed,
             (stepLen[0]),(stepLen[1])))

        df['name'][i]       = vid.split('/')[-1].split('.')[0]
        df['bodyLen'][i]    = bodyLen
        df['duration'][i], df['mov_dur'][i] = meta['dur'], movDur
        df['belt speed'][i]    = beltSpeed/10
        df['avg.speed'][i]  = avgSpeed
        df['peak_acc'][i]   = accMean.max()
        df['loc_front'][i], df['loc_rear'][i] = locHist[0],locHist[1]
        df['num_rec'][i], df['num_drag'][i] = recCount+1, dragCount+1
        df['count_ratio'][i] = (1+dragCount)/(1+recCount)
        df['dur_rec'][i], df['dur_drag'][i] = recDur, drgDur
        df['num_steps'][i] = nSteps
        df['LH_st_len'][i],df['LF_st_len'][i],df['RH_st_len'][i],df['RF_st_len'][i] = \
                stepLen[0], stepLen[2], stepLen[1], stepLen[3]
        df['LH_st_frq'][i],df['LF_st_frq'][i],df['RH_st_frq'][i],df['RF_st_frq'][i] = \
                cadence[0], cadence[2], cadence[1], cadence[3]
        df['LHRH_ang'][i],df['LHLF_ang'][i],df['RHRF_ang'][i],df['LFRH_ang'][i],df['RFLH_ang'][i],df['LFRF_ang'][i] = \
                meanPhi_heur, meanPhi_xL, meanPhi_xR, meanPhi_fLhR, meanPhi_fRhL, meanPhi_frnt
        df['LHRH_rad'][i],df['LHLF_rad'][i],df['RHRF_rad'][i],df['LFRH_rad'][i],df['RFLH_rad'][i],df['LFRF_rad'][i] = \
                R_heur, R_xL, R_xR, R_fLhR, R_fRhL, R_frnt
        df['LHRH_width'][i] = hWidth/10
        data = dict.fromkeys(keys,None)            
        data['speed'] = speedAll
        data['lCad'] = cadence[0]
        data['rCad'] = cadence[1]
        data['flCad'] = cadence[2]
        data['frCad'] = cadence[3]
        data['avg'] = avgSpeed
        data['rStLen'] = stepLen[1]
        data['lStLen'] = stepLen[0]
        data['frStLen'] = stepLen[3]
        data['flStLen'] = stepLen[2]
        data['nSteps']=nSteps
        data['phi_h']=phi_heur
        data['R_h'] = R_heur
        data['phi_xR']=phi_xR
        data['R_xR'] = R_xR
        data['phi_xL']=phi_xL
        data['R_xL'] = R_xL
        data['phi_fLhR']=phi_fLhR
        data['R_fLhR'] = R_fLhR
        data['phi_fRhL']=phi_fRhL
        data['R_fRhL'] = R_fRhL
        data['movDur']=movDur
        data['rStride']=stride[1]
        data['lStride']=stride[0]
        data['fRStride']=stride[3]
        data['fLStride']=stride[2]
        np.save(fName+'_Profile.npy',data)

    return df
