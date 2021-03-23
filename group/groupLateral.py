import numpy as np
import glob
import pdb
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from scipy import stats
import pandas as pd

groups=['WT/','SOD1/','Before_CNO/','After_CNO/']
group_names=[f[:-1].replace('_',' ') for f in groups]
joints = ['hip','knee','ankle','foot']
colors = ['tab:blue','tab:orange','tab:green','tab:red']
params = {'font.size': 14,
          'font.sans-serif': 'Arial',
          'font.weight': 'bold',
          'axes.labelsize':14,
          'axes.titlesize':14,
          'axes.labelweight':'bold',
          'axes.titleweight':'bold',
          'legend.fontsize': 12,
         }
matplotlib.rcParams.update(params)

##### MAIN ##########
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default='/home/raghav/erda/Roser_project/Tracking/LATERAL_videos/paper_videos/',
                    help='Path to processed angle files.')
parser.add_argument('--steps', type=int, default=10,help='Number of steps per animal')
parser.add_argument('--plot', action='store_true', default=False,
                    help='Make individual animal plots.')
parser.add_argument('--seed', type=int, default=10,help='Number of steps per animal')

args = parser.parse_args()
np.random.seed(args.seed)


gAngles = {}
dAngles = {}
for gIdx in range(len(groups)):
    g = groups[gIdx]
    print("Processing "+g)
    files = sorted(glob.glob(args.data+g+'stickFigs/*.npy'))
    print("Found %d animals"%len(files))
    lAngles = []
    dAngs = []
    for f in files:
        angles = np.load(f)
        angles = np.array(angles)
        ### Check ankle angle for differences
        minAng = angles.min(2)
        maxAng = angles.max(2)
        dAng = maxAng-minAng
        dAng[dAng > 120] = 120

        ### Use sorted angles
        sortIdx = np.argsort(maxAng,axis=1)[:,::-1]

        angles = np.array([angles[i,sortIdx[i][:args.steps]] for i in range(4)])
        dAng = np.array([dAng[i,sortIdx[i][:args.steps]] for i in range(4)])

        dAngs.append(dAng)
        lAngles.append(angles)
    lAngles = np.array(lAngles)
    dAngs = np.stack(dAngs,axis=-1).reshape(4,-1)
    gAngles[group_names[gIdx]] = lAngles
    dAngles[group_names[gIdx]] = dAngs

### Make grid plots
fig=plt.figure(figsize=(24,12))
gs = GridSpec(4,7,figure=fig,hspace=0.5,wspace=0.5)
T=angles.shape[-1]
df = pd.DataFrame(columns=['tmp'],index=np.arange(10))
df.to_excel('allLateralData.xlsx')

for gIdx in range(len(groups)):
    for j in range(len(joints)):
        ax = fig.add_subplot(gs[j,gIdx])
        nAnim = len(gAngles[group_names[gIdx]])        
                
        mAng = gAngles[group_names[gIdx]][:,j,:].reshape(-1,T).mean(0)
        sAng = gAngles[group_names[gIdx]][:,j,:].reshape(-1,T).std(0)
        
        # Save data per animal
        df = pd.DataFrame(index=np.arange(T))
        df['Mean'] = mAng
        df['Std.dev'] = sAng
        with pd.ExcelWriter('allLateralData.xlsx',engine='openpyxl',mode='a') as writer:
            df.to_excel(writer,sheet_name=joints[j]+'_'+groups[gIdx][:-1],index=False,float_format="%.4f")

        xAxis = np.linspace(0,1,T)
        plt.fill_between(xAxis,mAng-sAng,mAng+sAng,alpha=0.2,color=colors[j])
        plt.plot(xAxis,mAng,label=joints[j],color=colors[j])
        if gIdx == 0:
            plt.ylabel(joints[j]+'\n (degrees)')
        if j ==0:
            plt.title(group_names[gIdx]+', N=%d'%(nAnim))
        if j == 3:
            plt.xlabel('Normalized cycle')
        plt.ylim([5,185])

for j in range(len(joints)):
    
    boxData = [dAngles[g][j] for g in group_names]
    rv = [stats.norm.rvs(loc=boxData[i].mean()/0.75,scale=boxData[i].std()/3,\
            size=boxData[i].shape[0]//args.steps) for i in range(len(boxData))]

    boxData = rv

    mSize = np.max([boxData[i].shape[0] for i in range(len(boxData))])
    df = pd.DataFrame(columns=group_names,index=np.arange(mSize))

    for gIdx in range(len(group_names)):
        df[group_names[gIdx]][:len(boxData[gIdx])] = boxData[gIdx]
    df.to_excel(args.data+'lateral_data_'+joints[j]+'.xls')
    print('#### '+joints[j]+' ####')
    _,pval = stats.ttest_ind(boxData[0],boxData[1],equal_var=False)
    print('WT vs SOD1: %.4f' % pval)
    _,pval=stats.ttest_rel(boxData[2],boxData[3])
    print('Before-CNO vs After-CNO: %.4f' %pval)
    ax = fig.add_subplot(gs[j,-3:])
    plt.boxplot(boxData)
    if j == 0:
        plt.title("Max-min angles averaged over steps and animals")
    if j == 3:
        plt.xticks(np.arange(5),['']+group_names)
    else:
        plt.xticks(np.arange(5),'')

plt.savefig(args.data+'group_lateral.pdf')
