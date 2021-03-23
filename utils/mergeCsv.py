import pandas as pd
import numpy as np
import glob
import pdb
import copy

index = np.loadtxt('/home/raghav/erda/kiehn_lab/raghav/source/nathalie/allFiles.txt',delimiter='/',dtype=str)
speedIdx = [(index[i,-1].split('cms')[0][-2:]) for i in range(len(index))]
vidIdx = [(index[i,-1].split('UP')[-1][-1]) for i in range(len(index))]
index = [index[i].tolist()[:-1]+[speedIdx[i]]+[vidIdx[i]] for i in range(len(index))]
indexArr = np.array(index)[:,-1]

columns=['Name','bodyLen','Duration','locFrnt','locMid','locRear',
        'Belt Speed','Avg.Speed','PeakAcc.','Num_drag','Num_rec','Count_Ratio',
        'Dur_drag','Dur_rec','Dur_ratio','MovDur','Num_steps', 'Phi_heur',
        'R_heur', 'hLCad.', 'hRCad.', 'fLCad.', 'fRCad', 'hLStride', 'hRStride',
        'fLStride', 'fRStride']

#tuples = list((index))
mIndex = pd.MultiIndex.from_tuples(index, names=['Study', 'Group','Animal', 'Speed','Video']) 
hierDf = pd.DataFrame(index=mIndex,columns=columns)

fileNames = '/home/raghav/erda/kiehn_lab/raghav/source/nathalie/hierStructure.txt'
fileNames = open(fileNames, 'r')
files = fileNames.readlines()

N = len(files)

for fIdx in range(N):
    f = files[fIdx]
    print("Processing %3d/%d ..."%(fIdx,N))
    fLoc = f.strip().split('/')
    f = f[:len(fLoc[0])]+'/data'+f[len(fLoc[0]):]
    fName = '/home/raghav/erda/kiehn_lab/nathalie/'+f.strip()+'/speedProfile.csv'
    
    newDf = pd.DataFrame(columns=columns)
    df = pd.read_csv(fName,delimiter='\t')

#    pdb.set_trace()
    if df.shape[0] > 0:
        hierDf.loc[tuple(fLoc),:] = df.values
flatDf = hierDf.reset_index(level=[0,1,2,3,4])
sumDf = flatDf.groupby(['Study','Group','Animal','Speed']).sum()
sumDf = sumDf.reset_index(level=[0,1,2,3])
sumDf['Num_vid'] = 0
for i in range(sumDf.shape[0]):
    sumDf.iloc[i,-(len(columns)):] = sumDf.iloc[i,-(len(columns)):] / float(len(sumDf['Video'][i]))
    sumDf['Num_vid'][i] = float(len(sumDf['Video'][i]))
hierDf.reset_index(level='Video').drop(columns='Video').set_index(['Name'],append=True).to_excel('analysis.xls')
sumDf = sumDf.drop(columns=['Name'])
sumDf = sumDf.groupby(['Study','Group','Animal','Speed']).sum()
sumDf.to_excel('mean_analysis.xls')

