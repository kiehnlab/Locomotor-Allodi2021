from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import pdb
import pickle
import os
from coord import coordProfiler, meanVector, groupPlot
import numpy as np
from combinedPlot import processDict
import glob
import matplotlib.pyplot as plt
from constants import locKeys
import itertools
from pycircstat import watson_williams
#from scipy.stats import circmean
from tools import circular_mean as circmean
import pandas as pd

PI = np.pi
np.set_printoptions(precision=4,suppress=True)

def sampleSteps(phi,N,discrete=False,disc_ang=30):
    """
    Sample steps matching the distribution between the upper and lower
    halves of the circle
    """
#    pdb.set_trace()

    if discrete:
        phi = phi/PI*180
        phi = phi + (disc_ang - phi % disc_ang)
        phi = phi/180*PI
    phi = phi[np.random.permutation(len(phi))]
    lIdx = (phi >= PI/2) & (phi <= 3*PI/2)
    lPhi = phi[lIdx]
    uPhi = phi[~lIdx]
    if len(lPhi) >= len(uPhi):
        ratio = int(np.ceil(N*(len(uPhi)+1)/(len(lPhi)+1)-1))
        phi = np.concatenate((uPhi[:ratio],lPhi[:(N-ratio)]))
    else:
#        pdb.set_trace()
        ratio = int(np.ceil(N*(len(lPhi)+1)/(len(uPhi)+1)-1))
        phi = np.concatenate((lPhi[:ratio],uPhi[:(N-ratio)]))

    return phi

def groupSteps(data,files,nSteps,key='h',T=1,bins=2):
    files = np.array([f.split('/')[-1][:4] for f in files])
    uniq = np.unique(files)
    N = len(uniq)
    groupData = np.zeros((T,N,nSteps))

    for t in range(T):
        for i in range(len(uniq)):
            animSteps = np.zeros(1)
            idx = list(np.argwhere(files==uniq[i]).reshape(-1))
            for anim in idx:
                animSteps = np.concatenate((animSteps,data['phi_'+key][anim]))
            animSteps = animSteps[1:]
            nA = len(animSteps)
            if nA <= nSteps: #or nA <= bins:
                if nA <= nSteps:
                    nSteps = nA
                    print("Using %d steps"%nSteps)
                groupData[t,i,:nSteps] = animSteps[:nSteps]

            else:
                groupData[t,i,:nSteps] = sampleSteps(animSteps,nSteps)

    return groupData

def con_style(ax,idx=0,sig='n.s',xOf=0):
    if xOf == 0: 
        consty = "bar,fraction=0.1"
    else:
        consty = "bar,fraction=0.2"
    ax.annotate("", xytext=(550-xOf,yLoc[idx][0]),xy=(550-xOf, yLoc[idx][1]), xycoords=("figure pixels"),arrowprops=dict(arrowstyle="-",connectionstyle=consty))
    ax.annotate(sig, xy=(555-xOf,yLoc[idx][2]), xycoords=("figure pixels"),rotation=-90,fontsize=12)

    return ax


## MAIN PROGRAM STARTS HERE ## 

sig = ['*','**','n.s']
yLoc = [[615,638,620],[615,665,635],[642,670,652],[595,620,605]]

dirs = ['SOD1_study/',
       'CNO_study/',
       'SOD1-CNO_study/Pre-symptomatic_vs_onset/',
       'SOD1-CNO_study/SOD1-CNO_vs_before_after_CNO/']
colors = ['sod','cno','sodpre','both']

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/home/raghav/erda/Roser_project/Tracking/coordinationStudy/',
                    help='Path to coordination profiles')
parser.add_argument('--colors', type=str, default='',
                    help='Color schemes: sod,cno,both')
parser.add_argument('--steps', type=int, default='15',
                    help='Number of steps to use') 
parser.add_argument('--seed', type=int, default='0',
                    help='Random seed') 

args = parser.parse_args()
print("Analyzing coordination from "+args.data)

### Process DLC tracks to obtain the speed profiles
groups = list(set(glob.glob(args.data+'*')) - set(glob.glob(args.data+'*.*')))
groups = sorted(groups)
groupR = np.zeros(len(groups))
groupPhi = np.zeros(len(groups))
gNames = [g.split('/')[-1] for g in groups]

saveLoc= args.data.replace(args.data.split('/')[-2],'')[:-1]
#'/home/raghav/erda/Roser_project/Tracking/coordinationStudy/'
limbPair = ['h','xL','xR','fLhR','fRhL']
seeds = [0,9,6,401,101]
df = pd.DataFrame(columns=['tmp'],index=np.arange(10))
df.to_excel('allCircularData.xlsx')

#for kIdx in range(1,len(limbPair)):
for kIdx in range(5):

    limbs = limbPair[kIdx]
    print("\n#### Statistics for "+locKeys[kIdx]+" ####")

    cIdx = 0
    for data_dir in dirs[cIdx:]:
        print("\nCONDITION#",cIdx+1)
        np.random.seed(seeds[kIdx])
#        if 'Pre-sym' in data_dir: # or '_CNO' in args.data:
#            np.random.seed(400)
        print("#########################\nAnalyzing coordination for "\
              +data_dir+"\nUsing %d steps"%args.steps+\
              "\n#########################")
        data_dir = args.data+data_dir
        ### Process DLC tracks to obtain the speed profiles
#        pdb.set_trace()
        groups = list(set(glob.glob(data_dir+'*')) - set(glob.glob(data_dir+'*.pdf')))
        groups = sorted(groups)
        gNames = [g.split('/')[-1] for g in groups]
        groupR = np.zeros(len(gNames))
        groupPhi = np.zeros(len(gNames))
        gCombs = [[*x] for x in itertools.combinations(gNames,2)]
        if 'SOD1_study' in data_dir or 'Pre-sym' in data_dir:
            gCombs = gCombs[:1]
        if len(gCombs) == 6:
            gCombs = [gCombs[0],gCombs[-1]]
        if 0:
            for gIdx in range(len(groups)):
                g = groups[gIdx]
                files = sorted(glob.glob(g+'/*.npy'))
                data = processDict(files)
                fig = plt.figure(figsize=(16,10))
                fig,groupPhi[gIdx],groupR[gIdx] = meanVector(data,files,fig,colors[cIdx],gIdx,key=limbs)
                plt.title(gNames[gIdx].replace('_',' '))
                plt.savefig(g+'.pdf')

        pVal = np.zeros(len(gCombs))
        pIdx = 0
        for groups in gCombs:
            studyData = []
            for gIdx in range(len(groups)):
                g = groups[gIdx]
                files = sorted(glob.glob(data_dir+g+'/*.npy'))
                data = processDict(files)
                groupData = groupSteps(data,files,args.steps,key=limbs)
                studyData.append(groupData)
#            pVal[pIdx], table = watson_williams(circmean(studyData[0],axis=2),circmean(studyData[1],axis=2))
            pVal[pIdx], table = watson_williams(studyData[0],studyData[1])

            print("\nSignificance test for "+groups[0]+' and '+groups[1]+' with (p=0.05)')
            print("p=%.4f "%pVal[pIdx],[pVal[pIdx]<0.05])
#            print(pVal[pIdx],pVal[pIdx]<0.05)
            pIdx += 1
        fig = plt.figure(figsize=(16,10))
        df = pd.DataFrame(index=np.arange(50))
        popNp = np.empty((len(gNames),2,50))
        popNp[:] = np.nan
        for gIdx in range(len(gNames)):

            g = gNames[gIdx]
            files = sorted(glob.glob(data_dir+g+'/*.npy'))
            data = processDict(files)
            groupData = groupSteps(data,files,args.steps,key=limbs)

#            pdb.set_trace()
            popDataPhi = [(circmean(groupData[0][i])[0]*180/np.pi % 360) for i in range(groupData.shape[1])]
            popDataR = [circmean(groupData[0][i])[1] for i in range(groupData.shape[1])]

            popNp[gIdx,0,:len(popDataPhi)] = popDataPhi
            popNp[gIdx,1,:len(popDataPhi)] = popDataR

            df[g+' ang.'] = popNp[gIdx,0]
            df[g+' rad.'] = popNp[gIdx,1]

            fig,groupPhi[gIdx],groupR[gIdx] = meanVector(data,files,fig,
                                                     colors[cIdx],gIdx,key=limbs,scatter=True)

#        pdb.set_trace()
        with pd.ExcelWriter('allCircularData.xlsx',engine='openpyxl',mode='a') as writer:
                df.to_excel(writer,sheet_name=data_dir.split('/')[-2]+'_'+locKeys[kIdx],index=False,float_format="%.4f")

        fig,meanPhi,meanR = groupPlot(groupPhi,groupR,fig,gNames,cScheme=colors[cIdx])
        plt.title(args.data.split('/')[-1]) 
        plt.legend(loc='upper left',bbox_to_anchor=(-0.05, 1.08),frameon=False)
        ax = fig.add_subplot(111,polar=False)
        ax = plt.gca()
#        pdb.set_trace()
        for i in range(pIdx):
            if pVal[i] < 0.005:
                sig = '**'
            elif pVal[i] < 0.05:
                sig = '* '
            else:
                sig = 'n.s'
            if i == 1: 
                xOf = 0
            else:
                xOf = 20 

            if cIdx == 1 or cIdx == 3:
                xOf = 50
                ax = con_style(ax,idx=i+2,sig=sig,xOf=xOf)
            elif len(gCombs) == 1:
                ax = con_style(ax,idx=i+2,sig=sig,xOf=xOf)
            else:
                ax = con_style(ax,idx=i,sig=sig,xOf=xOf)
        ### Annotate significance lines
        plt.savefig(args.data+data_dir.split('/')[-2]+'_'+locKeys[kIdx]+'.pdf')
#        pdb.set_trace()
        cIdx += 1
