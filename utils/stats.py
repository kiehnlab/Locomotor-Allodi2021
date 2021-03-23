from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import pdb
import pickle
import os
from utils.constants import locKeys
import numpy as np
from vis.plotter import processDict
import glob
import matplotlib.pyplot as plt
from pycircstat import watson_williams 
import itertools
from statsmodels.stats.multitest import multipletests
np.set_printoptions(precision=4,suppress=True)
from numpy import pi as PI

def densitySample(phi,N,bins):
    """
    Sample steps matching the actual distribution
    """
    phi = phi[np.random.permutation(len(phi))]
    pdf, levels = np.histogram(phi,bins=bins)
    levels = np.concatenate((levels.reshape(-1),np.array([levels.sum()])))

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

def medianSamples(phi,N):
    """
    Sample steps matching the distribution on either side of median angle
    Use median step angle and sample N/2 on either side
    """
    phi = np.sort(phi)
    medAng = np.median(phi)
    lIdx = (phi >= medAng) 
    lPhi = phi[lIdx]#[::-1]
    uPhi = phi[~lIdx][::-1]
    phi = np.concatenate((uPhi[:N//2],lPhi[:N//2+1]))
    phiRet = np.zeros(N)
    phiRet[:len(phi)] = phi
    return phiRet


def sampleSteps(phi,N,discrete=False):
    """
    Sample steps matching the distribution between the upper and lower
    halves of the circle
    Use median step angle and sample N/2 on either side
    """
    if discrete:
        phi = phi/PI*180
        phi = phi + (30 - phi % 30)
        phi = phi/180*PI
    pdb.set_trace()
    phi = phi[np.random.permutation(len(phi))]
    lIdx = (phi >= PI/2) & (phi <= 3*PI/2)
    lPhi = phi[lIdx]
    uPhi = phi[~lIdx]
    if len(lPhi) >= len(uPhi):
        ratio = int(np.ceil(N*(len(uPhi)+1)/(len(lPhi)+1)-1))
        phi = np.concatenate((uPhi[:ratio],lPhi[:(N-uPhi[:ratio].shape[0])]))
    else:
        ratio = int(np.ceil(N*(len(lPhi)+1)/(len(uPhi)+1)-1))
        phi = np.concatenate((lPhi[:ratio],uPhi[:(N-lPhi[:ratio].shape[0])]))
    phiRet = np.zeros(N)
    phiRet[:len(phi)] = phi
    return phiRet

def groupSteps(data,files,nSteps,key='h',T=1,bins=2):
    files = np.array([f.split('/')[-1][:20] for f in files])
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
                 groupData[t,i,:nSteps] = np.sort(animSteps)[:nSteps]

    return groupData
