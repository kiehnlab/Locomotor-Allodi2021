import pandas as pd
import glob 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist 
import pdb
from utils.tools import videoMetadata
import argparse
import os
import seaborn as sns
from scipy.stats import norm
from scipy.signal import find_peaks
from bottom.coord import iqrMean
from matplotlib.gridspec import GridSpec
from vis.sticks import makeStickFigure

joints = ['hip','knee','ankle','foot']
colors = ['tab:blue','tab:orange','tab:green','tab:red']

def getSwingIdx(peaks,yval,thresh=0.03):
        
    ### Find out ymin in each cycle and pull it to ground 0
    C = len(peaks)
    for i in range(-1,C-1):
        if i == -1:
            lIdx = 0
        else:
            lIdx = peaks[i]
        uIdx = peaks[i+1]
        step = yval[lIdx:uIdx,:]
        yval[lIdx:uIdx,:] = step - step[:,-1].min()
    swing_idx = np.where((yval[:,-1] >= thresh) & (yval[:,-2] >= thresh))[0]
    return swing_idx, yval

def spatialSync(x,swing_idx):
    delta = 0.1
    x = x * dist.max(1).reshape(-1,1)/args.scale 
    x = x + np.arange(len(x))
    idx = np.where(np.diff(swing_idx) > 1)[0]
    idx = np.concatenate((swing_idx[[0]],idx))
    for i in idx[:-1]:

        tmp = x[swing_idx[i]:swing_idx[i+1]]
        tmp = tmp + np.arange(swing_idx[i+1]) * delta

    stance_idx = np.delete(np.arange(len(swing_idx)),swing_idx)
    x[swing_idx] += x[swing_idx]*2 
    x[stance_idx] += x[stance_idx]/2 

    return x

def cycleAngles(peaks,angle,num_angles=51):
    C = len(peaks)
    cyc_angle = np.zeros((C-1,num_angles))

    for i in range(C-1):
        lIdx = peaks[i]
        uIdx = peaks[i+1]
        xAxis = np.linspace(lIdx,uIdx,num_angles)
        cyc_angle[i,:] = np.interp(xAxis,np.arange(lIdx,uIdx),angle[lIdx:uIdx])

    return cyc_angle

def pDist(arr):
    N = arr.shape[0]
    d = arr.shape[1]
    dist = np.zeros((N,int(d*(d-1)/2)))

    for i in range(N):
        dist[i] = pdist(arr[i].reshape(-1,1))

    return dist

def smooth(x,L=15):
    M = x.shape[1]
    xF = np.zeros((x.shape[0]-L+1,M))
    for i in range(x.shape[1]):
        xF[:,i] = np.convolve(x[:,i],np.ones((L,))/L,mode='valid')
    return xF

def makeVector(yval,xval,i,j):
    vx = (xval[:,i] - xval[:,j]).reshape(-1,1)
    vy = (yval[:,i] - yval[:,j]).reshape(-1,1)
    v = np.concatenate((vx,vy),axis=1)
    v = v/np.linalg.norm(v,axis=1).reshape(-1,1)
    return v

def measureAngles(yval,xval,i,j,k):
    v1 = makeVector(yval,xval,i,j)
    v2 = makeVector(yval,xval,k,j)
    angle = np.arccos((v1*v2).sum(1))
    return 180*angle/np.pi

def lateral_profiler(data_dir,scale,df):
    model = 'DLC_resnet50_lateral_analysisOct26shuffle1_100000'
    bodyparts = ['toe','foot','ankle','knee','hip','crest']
    bodyparts = bodyparts[::-1]

    files = sorted(glob.glob(data_dir+'*.avi'))
    files = [f.split('/')[-1] for f in files]

    data_dir = data_dir.replace('lateral/','')
    dest = data_dir+'all_profiles/'
    if not os.path.exists(dest):
        os.mkdir(dest)

    print("Found %d videos to process"%len(files))
    M = len(bodyparts)
    for fName in files:
        meta = videoMetadata(data_dir+'lateral/'+fName)
        data_file = glob.glob(data_dir+'labels/'+fName.replace('.avi','*.h5'))[0]
        data = pd.read_hdf(data_file)
        N = data.shape[0]
        xval = np.zeros((N,M))
        yval = np.zeros((N,M))
        for m in range(M):
            xval[:,m] = data[model][bodyparts[m]]['x'].values
            yval[:,m] = data[model][bodyparts[m]]['y'].values
        # Smooth values
        xval = smooth(xval,L=10)
        yval = smooth(yval,L=10)

        # Rescale x,y to be [0,1]; y per time point, x across
        yval = (1-(yval-yval.min()) / (yval.max()-yval.min()))
        xval = (xval - xval.min(1).reshape(-1,1)) / (xval.max(1)-xval.min(1)).reshape(-1,1)
        t = np.linspace(0,meta['dur'],len(xval))[::-1]
        dist = pDist(xval)
        dist = dist/dist.max()
        xAng = xval * dist.max(1).reshape(-1,1)/scale + t.reshape(-1,1)
        xAng = xAng - (xAng[:,[-1]] - t.reshape(-1,1))

        ### Measure cycles based on foot angle
        peaks, _ = find_peaks(yval[:,-1])
        meanStep = iqrMean(np.diff(peaks))
        ## Remove outlier peaks
        peaks, _ = find_peaks(yval[:,-1],distance=meanStep)
        swing_idx,yval = getSwingIdx(peaks,yval) 
        ### Measure angles
        angles = [measureAngles(yval,xAng,i,i+1,i+2) for i in range(4)]
        cyc_angles = [cycleAngles(peaks,angle) for angle in angles]
        cyc_angles = np.array(cyc_angles)
        ### Check ankle angle for differences
        minAng = cyc_angles.min(2)
        maxAng = cyc_angles.max(2)
        dAng = maxAng-minAng
        dAng[dAng > 120] = 120
        dAng = dAng.mean(1)
        for i in range(len(joints)):
#           df[joints[i]+'_ang'][df.Name in fName] = dAng[i]
            df.at[df.name == fName[9:-4],joints[i]+'_ang'] = dAng[i]
        ### Save cycle_angles
        np.save(dest+fName.replace('.avi','.npy'),cyc_angles)
        makeStickFigure(xval,yval,dist,angles,\
                meta['dur'],fName,cyc_angles,peaks,swing_idx,scale,dest)

    return df
