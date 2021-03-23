from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import pdb
import pickle
import os
os.environ['DLClight'] = 'True'
import shutil
import glob
import pandas as pd
from bottom.profiler import bottom_profiler
from lateral.lateral import lateral_profiler
from vis.plotter import combinedPlot
from utils.constants import *

def process_view(data,config,analyze_video,label_video):

    print("Using config file:"+config)
    if 'lateral' in data:
        dest = data.replace('lateral/','')+'labels/'
    else:
        dest = data+'labels/'
    if os.path.exists(dest):
        if analyze_video:
            print("Found directory with labels... Overwriting them....")
    else:
         os.mkdir(dest)

    ### Step:1 
    ### Use trained DeepLabCut model to obtain tracks for markers
    if analyze_video:
        deeplabcut.analyze_videos(config,[data],videotype='.avi',destfolder=data)
    if label_video:
        if 'lateral' in data:
            labvid_dir = data.replace('lateral/','')+'labeled_videos'
        else:
            labvid_dir=data+'labeled_videos'
        if os.path.exists(labvid_dir):
            print("Found directory with labelled videos... Overwriting if labels exist....")
        else:
            os.mkdir(labvid_dir)
            
        deeplabcut.create_labeled_video(config,[data],videotype='.avi')
        labels=glob.glob(data+'*labeled.mp4')
        [shutil.copy(f,labvid_dir) for f in labels]
        [os.remove(f) for f in labels]

       
    if analyze_video:
        labels=glob.glob(data+'*.h5')
        [shutil.copy(f,dest) for f in labels]           # Overwrite if file exists
        [os.remove(f) for f in labels]              # Clean up

        labels=glob.glob(data+'*.pickle')
        [shutil.copy(f,dest) for f in labels]
        [os.remove(f) for f in labels]

    return

## MAIN PROGRAM STARTS HERE ## 

parser = argparse.ArgumentParser()
parser.add_argument('--lateral', action='store_true', default=False,
                    help='Track lateral view videos')
parser.add_argument('--lateral_cfg', type=str, 
                    default=lat_cfg,help='Path to lateral config file')
parser.add_argument('--scale', type=int, 
                    default=2, help='Scaling to control distance bw sticks')
parser.add_argument('--bottom', action='store_true', default=False,
                    help='Track bottom view videos')
parser.add_argument('--bottom_cfg', type=str, 
                    default=bot_cfg, help='Path to bottom config file')
parser.add_argument('--data', type=str, default='/home/raghav/erda/Roser_project/Tracking/',
                    help='Path to video files.')
parser.add_argument('--analyze_video', action='store_true', default=False,
                    help='Analyze videos.')
parser.add_argument('--label_video', action='store_true', default=False,
                    help='Make labelled videos.')
parser.add_argument('--compute_stats', action='store_true', default=False,
                    help='Compute statistics from labels')


args = parser.parse_args()
if args.data[-1] != '/':
    args.data = args.data+'/'

print("Using videos in "+args.data)
if args.analyze_video or args.label_video:      #Check if videos should be retracked in DLC
    import deeplabcut

if args.bottom:                                 # Obtain labels from bottom view 
    print("Processing bottom views...")
    process_view(args.data,args.bottom_cfg,args.analyze_video,args.label_video)


lateral_dir = args.data+'lateral/'              # Assumed that lateral videos are in this loc.
if args.lateral:                                # Obtain labels from lateral view
    print("Processing lateral views...")
    process_view(lateral_dir,args.lateral_cfg,args.analyze_video,args.label_video)

if args.compute_stats:                          # Now for all the analysis
    N = len(glob.glob(args.data+'*.avi'))
    df = pd.DataFrame(columns=df_cols,index=range(N)) 
    df[df_cols[1:]] = df[df_cols[1:]].apply(pd.to_numeric)
    if args.bottom:
        ### Step 2:
        ### Process DLC tracks to obtain all measurements
        df = bottom_profiler(data_path=args.data,df=df)   # Speed, acceleration, coord.

    ### Step 4:
    ### Lateral analysis
    if args.lateral:
        df = lateral_profiler(lateral_dir,args.scale,df)    # Measure joint angles, make stick figures
    df.to_csv(args.data+'all_profiles/statistics.csv',index=False,float_format='%.4f',na_rep='0')
