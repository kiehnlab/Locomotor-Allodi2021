import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import time
import os
from combinedPlot import processDict
from tools import videoMetadata
from constants import *
import pdb 
from scipy.stats import circmean
from coord import iqrMean, heurCircular
from matplotlib import gridspec
from accel import estimateAccel, analyseDragRec

## MAIN PROGRAM STARTS HERE ## 

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/home/raghav/erda/Roser_project/Tracking/coordinationStudy/',
                    help='Path to labeled videos')
parser.add_argument('--colors', type=str, default='',
                    help='Color schemes: sod,cno,both')
parser.add_argument('--start', type=float, default=0,
                    help='Video starting time')
parser.add_argument('--end', type=float, default=-1,
                    help='Video ending time')


args = parser.parse_args()
print("Analyzing coordination from "+args.data)

os.chdir(args.data)
files = sorted(glob.glob('*.avi'))
files = [f.replace('.avi','') for f in files]
videoName = args.data.split('/')[-2].replace('_',' ')
for ipFile in files:
        meta = videoMetadata(ipFile+'.avi')
        sFrame = 0
        eFrame = meta['nFrame']

        if not os.path.exists('overlays'):
                os.mkdir('overlays')
        filename = sorted(glob.glob(ipFile+'*_labeled.mp4'))[0] 

        dataFiles = sorted(glob.glob(ipFile+'*_Profile.npy')) 
        allData = processDict(dataFiles)

        ### Obtain speed parameters
        speedAll = allData['speed'][0]
        speedMean = speedAll.mean(0)
        speedMean = np.convolve(speedMean, 
                                np.ones((speedSmFactor,))/speedSmFactor, mode='valid')
        speedStd = speedAll.std(0)
        speedStd = np.convolve(speedStd, 
                               np.ones((speedSmFactor,))/speedSmFactor, mode='valid')


        newFrame = speedMean.shape[0]
        yMax = (speedMean + speedStd).max() # + speedStd.max()
        yMin = (speedMean - speedStd).min()
        avgSpeed = speedMean[::int(meta['fps']/3)].mean()

        
        ### Obtain cadence parameters
        lStride = allData['lStride'][0]
        rStride = allData['rStride'][0]
        lSMean = iqrMean(lStride)
        rSMean = iqrMean(rStride)
        sMean = lSMean - rSMean
        stride = lStride-rStride
        sMin = stride.min()
        sMax = stride.max()
        T = len(lStride)

#        pdb.set_trace()
        if args.start > 0:
            idx = int(args.start/meta['dur']*newFrame)
            speedMean = speedMean[idx:]
            speedStd = speedStd[idx:]
            
            idx = int(args.start/meta['dur']*T)
            stride = stride[idx:]
            sFrame = int(args.start/meta['dur']*meta['nFrame'])

        if args.end > -1:
            idx = newFrame - int(args.end/meta['dur']*newFrame)
            speedMean = speedMean[:-idx]
            speedStd = speedStd[:-idx]

 
            idx = T-int(args.end/meta['dur']*T)
            stride = stride[:-idx]
            eFrame = int(args.end/meta['dur']*meta['nFrame'])
        
        if args.start > 0 or args.end > -1:

            newFrame = speedMean.shape[0]
            T = len(stride)
            dur = args.end - args.start
            phi,R,_,_,idx = heurCircular(np.arange(T),stride,sMean,True)
            xAxis = np.linspace(0,dur,newFrame)
            cAxis = np.linspace(0,dur,T)

        else:
            ### Obtain circular plot parameters
            phi = allData['phi_h'][0]
            R = allData['R_h'][0]
            idx = np.where(np.diff(stride > sMean)==True)[0]

            xAxis = np.linspace(0,meta['dur'],newFrame)
            cAxis = np.linspace(0,meta['dur'],T)

        if len(idx) % 2 != 0: 
                idx = idx[:-1]
        idx = idx.reshape(-1,2)[:,0]


#        pdb.set_trace()
        ### Perform acceleration analysis
        accMean, drgIdx, recIdx, _ = estimateAccel(speedMean,meta)
        dragCount, recCount, drgDur, recDur, drgIdx, recIdx = \
        analyseDragRec(drgIdx,recIdx,meta['fps'], tThr)
        aAxis = np.linspace(0,dur,accMean.shape[0])

        print("Drag events: %d, Rec. events : %d"%(dragCount,recCount))

        ### Initialize output frame
        fig = plt.figure(figsize=(35,20))
        gs = gridspec.GridSpec(3, 4)

        ### DLC Video frame in output frame
        ax0 = plt.subplot(gs[1,1:-1])
        ### Capture a frame
        cap = cv2.VideoCapture(filename)
        plt.ion()
        plt.show()
        skipFrame = int(meta['fps']/frameRate)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Measure the speed from tracks
        beltSpeed = ipFile.split('cms')[0].split('_')[-1]
        beltSpeed = float(beltSpeed)
        print('Belt speed is : %.2f cm/s'% (beltSpeed))

        ### Speed profile in output frame
        ax1 = plt.subplot(gs[1,:2])#    plt.subplot(4,1,2)
        plt.title('Speed Profile',fontdict={'size':18})# for '+videoName)
        plt.plot(xAxis,beltSpeed*np.ones(newFrame),
                 '--',label='Belt Speed = '+repr(beltSpeed)+'cm/s')
        plt.plot(xAxis,avgSpeed.mean()*np.ones(newFrame),':',label='Avg. Speed')
        plt.ylabel('Speed (cm/s)')
        plt.xlabel('Time (s)')
        plt.legend(loc='upper right',prop={'size': 16})

        ### Cadence profile in output frame
        ax1 = plt.subplot(gs[2,:2]) #plt.subplot(4,1,3)
        plt.title('Cadence Profile',fontdict={'size':18})# for '+videoName)
        plt.plot(cAxis,np.ones(T)*sMean,'--',
                 color='dimgrey',label='Mean crossing point')
        plt.xlabel('Time (s)')
        plt.ylabel('Stride length (mm)')
        plt.legend(loc='upper right',prop={'size': 16})
        textstr = '{:.2f}'.format(avgSpeed)+' cm/s'
        textvar = plt.text(0, yMax , textstr, fontsize=14,bbox=props)


        ### Acceleration in output frame
        ax1 = plt.subplot(gs[1,2:])
        plt.title('Acceleration Profile',fontdict={'size':18})
        plt.plot(xAxis,np.zeros(len(xAxis)),'--k',label='Zero acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration in cm/s2')

        ### LR Coordination in output frame
        ax = plt.subplot(gs[2,2:],polar=True)
        ax.set_rlim(0,1.1)
        ax.spines['polar'].set_visible(False)
        ax.scatter(0,0,marker='o',s=250,c='royalblue')
        ax.set_axisbelow(True)
        ax.set_theta_offset(np.pi/2)
        plt.title('Left-Right Alternation Profile',fontdict={'size':18})# for '+videoName)

        N = (len(phi) if len(phi) > 1 else 1)

        k = -1
        dIdx = -1
        rIdx = -1

        cadIdx = np.array(np.arange(0,T,skipFrame*T/newFrame),dtype=int)
        for i,j,fNum,m in zip(np.arange(0,newFrame,skipFrame),
                             cadIdx,np.arange(sFrame,eFrame,skipFrame),
                             np.arange(0,accMean.shape[0],skipFrame)):
#                m += accSmFactor

                if (j >= idx[k+1]) and (k < len(idx)):
                        k += 1

                ### Plot video frame
                ax0 = plt.subplot(gs[0,1:-1]) #plt.subplot(4,1,1)
                cap.set(1,fNum+accSmFactor+speedSmFactor-2)
                flag, frame = cap.read()
                plt.imshow(frame)
                plt.title('Analysis for '+videoName,fontdict={'size':24})

                ### Plot speed profile
                ax1 = plt.subplot(gs[1,:2]) #plt.subplot(4,1,2)
                plt.fill_between( xAxis[:i], 
                                 speedMean[:i]-speedStd[:i], 
                                 speedMean[:i] + speedStd[:i], 
                                 color='lightgray')
                plt.plot(xAxis[:i],speedMean[:i],color='tab:blue',
                         label='Avg. Instataneous speed',linewidth=3)

                plt.ylim([yMin-5,yMax+5])
                plt.xlim([0,5.5])

                textvar.remove()
                textstr = '{:.2f}'.format(speedMean[i])+' cm/s'
                textvar = plt.text(0, yMax , textstr, fontsize=14,bbox=props)
                textvar.set_text(textstr) 
               
                ### plot acceleration profile
                ax1 = plt.subplot(gs[1,2:])
                ax1.clear()
                plt.title('Acceleration Profile',fontdict={'size':18})
                plt.plot(aAxis,np.zeros(len(aAxis)),
                         '--k',label='Zero acceleration')
                plt.xlabel('Time (s)')
                plt.ylabel('Acceleration in cm/s2')


                plt.plot(aAxis[:m],accMean[:m],color='lightgray',
                        label='Instantaneous acceleration',linewidth=4)
                if m > 0:
#                    if m == 168:
#                        pdb.set_trace()
                    accMax = np.argmax(accMean[:m])+1
                    plt.plot(aAxis[accMax],accMean[accMax],marker='^',
                             markersize=12,linestyle='',
                        label='Current Peak Acceleration',c='crimson')

                if drgIdx.size > 0 and dIdx < (drgIdx.shape[1]-1): 
                    if m >= drgIdx[1,dIdx+1]:
                        dIdx += 1
                for d in range(dIdx+1):
#                    pdb.set_trace()
                   
                    sIdx = drgIdx[0,d] #+ accSmFactor
                    lIdx = drgIdx[1,d] #+ accSmFactor
                    if d == 0:
                        plt.plot(aAxis[sIdx:lIdx],accMean[sIdx:lIdx],
                             linestyle='-',linewidth=3,
                             c='tab:blue',label='Recovery event')
                    else:
                        plt.plot(aAxis[sIdx:lIdx],accMean[sIdx:lIdx],
                             linestyle='-',linewidth=3,
                             c='tab:blue',label='_nolegend_')


                if recIdx.size > 0 and rIdx < (recIdx.shape[1]-1): 
                    if m >= recIdx[1,rIdx+1]:
                        rIdx += 1
                for r in range(rIdx+1):
                    sIdx = recIdx[0,r] #+ accSmFactor
                    lIdx = recIdx[1,r] #+ accSmFactor
                    if r == 0:
                        plt.plot(aAxis[sIdx:lIdx],accMean[sIdx:lIdx],
                             linestyle='-',linewidth=3,
                             c='tab:orange',label='Drag event')
                    else:
                        plt.plot(aAxis[sIdx:lIdx],accMean[sIdx:lIdx],
                             linestyle='-',linewidth=3,
                             c='tab:orange',label='_nolegend_')
 

                plt.legend(loc='lower right',prop={'size': 16})
                plt.ylim([-2.5,2])
                plt.xlim([0,5.5])


                ### plot cadence profile
                ax1 = plt.subplot(gs[2,:2]) #plt.subplot(4,1,3)
                ax1.clear()
                plt.title('Cadence Profile',fontdict={'size':18})# for '+videoName)
                plt.plot(cAxis,np.ones(T)*sMean,'--',
                         color='dimgrey',label='Mean crossing point')
                plt.xlabel('Time (s)')
                plt.ylabel('Stride length (mm)')

                plt.ylim([0.8*sMin,1.2*sMax])
                plt.plot(cAxis[:j],stride[:j],
                         color='tab:blue',linewidth=3,
                         label='Relative position of hind limbs')
                plt.xlim([0,5.5])

                if k > -1:
#                        pdb.set_trace()
                        plt.plot(cAxis[idx[:k+1]],stride[idx[:k+1]],
                                'o',label='Possible full cycle',
                                markersize=12,c='crimson')

                        plt.legend(loc='upper right',prop={'size': 16})
                        ax = plt.subplot(gs[2,2:],polar=True)
#                       pdb.set_trace()
                        ax.clear()
                        ax = plt.subplot(gs[2,2:],polar=True)

#                       ax.cla()
                        
#                       ax = plt.subplot(gs[1:,2:],polar=True)
                        ax.scatter(phi[:k+1],np.ones(k+1),marker='o',
                                   s=250,c='royalblue') 
                        ax.scatter(phi[k],np.ones(1),marker='o',
                                  s=250,c='crimson')
                        ax.annotate("",xytext=(0.0,0.0),xy=(circmean(phi[:k+1]),R),
                                   arrowprops=dict(facecolor=colors[-1],width=0.5))
                        ax.plot((0,circmean(phi[:k+1])),(0,R),color=colors[-1])

                        plt.title('Left-Right Alternation Profile',fontdict={'size':18})# for '+videoName)
#               else:
#                       ax = plt.subplot(gs[1:,2:],polar=True)
#                       ax.cla()
#                       ax.set_theta_offset(np.pi/2)                    
#                       ax.scatter(0,0,marker='o',s=100,c='crimson')
                ax.set_rlabel_position(25)
                ax.set_rlim(0,1.1)
                ax.set_theta_offset(np.pi/2)
                plt.tight_layout()
                plt.savefig('overlays/fig'+'{0:05d}'.format(i)+'.jpg');


