from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import pdb
import pickle
import os
#from coord import coordProfiler, meanVector, groupPlot
from constants import locKeys
import numpy as np
from combinedPlot import processDict
import glob
import matplotlib.pyplot as plt
from pycircstat import watson_williams 
from scipy.stats import circmean
import itertools
from statsmodels.stats.multitest import multipletests
np.set_printoptions(precision=4,suppress=True)
from numpy import pi as PI
import pandas as pd
dirs = ['SOD1_study/',
       'CNO_study/',
       'SOD1-CNO_study/Pre-symptomatic_vs_onset/',
       'SOD1-CNO_study/SOD1-CNO_vs_before_after_CNO/']

def densitySample(phi,N,bins):
    """
    Sample steps matching the actual distribution
    """
    phi = phi[np.random.permutation(len(phi))]
    pdf, levels, _ = plt.hist(phi,bins=bins)
    levels = np.concatenate((levels.reshape(-1),np.array([levels.sum()])))

#    pdb.set_trace()
    pdf = pdf/pdf.sum()
    stepDist = (np.round(pdf*N)).astype(int)
    if stepDist.sum() != N:
        stepDist[np.argmax(stepDist)] -= (stepDist.sum()-N)
    phiSample = np.zeros(1)

    for l in range(bins):
        if stepDist[l] > 0:
            samples = phi[(phi > levels[l]) & (phi <= levels[l+1])]
            samples = samples[:stepDist[l]]
            phiSample = np.concatenate((phiSample,samples.reshape(-1)))
    
    return phiSample[-N:]

def sampleSteps(phi,N,discrete=False):
    """
    Sample steps matching the distribution between the upper and lower
    halves of the circle
    """
#    pdb.set_trace()
    if discrete:
        phi = phi/PI*180
        phi = phi + (30 - phi % 30)
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
    #        assert nA >= nSteps, 'Only %d steps found! Consider reducing nSteps to use.'%nA
            if nA <= nSteps: #or nA <= bins:
#                pdb.set_trace()
                if nA <= nSteps:
                    nSteps = nA
                    print("Using %d steps"%nSteps)
                groupData[t,i,:nSteps] = animSteps[:nSteps]

#            animIdx = np.random.permutation(np.arange(sIdx,nA-sIdx))
#            groupData[t,i,:nSteps] = animSteps[animIdx[:nSteps]] 
#            print("Using %d steps"%nSteps)

            else:
    #            groupData[t,i,:nSteps] = densitySample(animSteps,nSteps,bins) 
                groupData[t,i,:nSteps] = sampleSteps(animSteps,nSteps) 

    return groupData


## MAIN PROGRAM STARTS HERE ## 

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/home/raghav/erda/Roser_project/Tracking/coordinationStudy/',
                    help='Path to coordination profiles')
parser.add_argument('--colors', type=str, default='',
                    help='Color schemes: sod,cno,both')
parser.add_argument('--steps', type=int, default='15',
                    help='Number of steps to use')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--trial', type=int, default=1,
                    help='No. of random trials')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Significance threshold')
 
args = parser.parse_args()
df = pd.DataFrame(columns=['tmp'],index=np.arange(10))
df.to_excel('allData.xlsx')

for data_dir in dirs:
    np.random.seed(args.seed)
    if 'Pre-sym' in data_dir: # or '_CNO' in args.data:
        np.random.seed(400)
    print("\n#########################\nAnalyzing coordination for "\
          +data_dir+"\nUsing %d steps"%args.steps+\
          "\n#########################")
    data_dir = args.data+data_dir
    ### Process DLC tracks to obtain the speed profiles

#    pdb.set_trace()
    groups = list(set(glob.glob(data_dir+'*')) - set(glob.glob(data_dir+'*.pdf')))
    groups = sorted(groups)
    groupR = np.zeros(len(groups))
    groupPhi = np.zeros(len(groups))
    gNames = [g.split('/')[-1] for g in groups]
    gCombs = [[*x] for x in itertools.combinations(gNames,2)]
    if len(gCombs) == 6:
        gCombs = [gCombs[0],gCombs[-1]]
    saveLoc='/home/raghav/erda/Roser_project/Tracking/coordinationStudy/'
    keys =  ['h','xL','xR','fLhR','fRhL']
    for kIdx in range(len(keys)):
        key = keys[kIdx]
        print("\n#### Statistics for "+locKeys[kIdx]+" ####")
        for groups in gCombs:

            studyData = []
            for gIdx in range(len(groups)):
                g = groups[gIdx]
                files = sorted(glob.glob(data_dir+g+'/*.npy'))
                data = processDict(files)

                groupData = groupSteps(data,files,args.steps,key=key,T=args.trial)
                studyData.append(groupData)


            pVal = np.zeros(args.trial)
            for t in range(args.trial):
                if len(studyData) == 3:
                    pVal[t], table = watson_williams(studyData[0][t],studyData[1][t],studyData[2][t])
                elif len(studyData) == 4:
                    pVal[t], table = watson_williams(studyData[0][t],studyData[1][t],studyData[2][t],studyData[3][t])
                elif len(studyData) == 2:
                    pVal[t], table = watson_williams(studyData[0][t],studyData[1][t])
                    df = pd.DataFrame(columns=[groups[0],groups[1]],index=np.arange(50))
                    popNp = np.empty((2,50))
                    popNp[:] = np.nan

#                    pdb.set_trace()
                    for j in range(2):
                        popData = [circmean(studyData[j][t][i]) for i in range(studyData[j][t].shape[0])]
                        popNp[j,:len(popData)] = popData
                        df[groups[j]] = popNp[j]

    #            print('Trial %d: p-value for groups '%(t)+groups[0]+' and '+groups[1]+': %.4f'%(pVal[t]))

    #        pdb.set_trace()
            levels, pAdj,_, p = multipletests(pVal,alpha=args.alpha)
            print("\nSignificance test for "+groups[0]+' and '+groups[1]+' with (p=%.4f)'%args.alpha)
            with pd.ExcelWriter('allData.xlsx',engine='openpyxl',mode='a') as writer:
                df.to_excel(writer,sheet_name=groups[0]+' vs '+groups[1],index=False)
    #        print("Before Bonferroni correction (p=%.4f)"%args.alpha)
            print(pVal,levels)
    #        if args.trial > 1:
    #            print("After Bonferroni correction:")# (p=%.4f)"%p)
    #            print(pAdj)
    #            print("%d/%d trials are significant"%(len(levels[levels==True]),args.trial))
#            print(levels)
