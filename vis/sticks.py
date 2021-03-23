import numpy as np
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
from matplotlib.gridspec import GridSpec

joints = ['hip','knee','ankle','foot']
colors = ['tab:blue','tab:orange','tab:green','tab:red']

def makeStickFigure(x,y,dist,dur,fName,cyc_angles,peaks,\
        swing_idx,time_sync,sticks_only,num_steps=10,skip=1):
    plt.clf()
    stance_col = 'tab:grey'
    swing_col = 'tab:red'
    t = np.linspace(0,dur,len(x))[::-1]
    if not sticks_only:
        fig = plt.figure(figsize=(20,10))
        gs = GridSpec(3,6,figure=fig,hspace=0.5,wspace=0.5)
    ### Plot instantaneous angles
        ax = fig.add_subplot(gs[1,:3])
        for i in range(len(joints)):    
            sns.lineplot(t,angles[i],label=joints[i])
        plt.ylim([-5,185])
        plt.xlim([-0.1,dur+1])
        plt.xlabel("Time")
        plt.legend()
        
        ### Plot angle densities
        ax = fig.add_subplot(gs[1,3:])
        norm=None
        B = 36
        for i in range(len(joints)):    
            sns.distplot(angles[i],bins=B,label=joints[i],fit=norm)
        plt.ylim([-0.01,0.11])
        plt.xticks(np.arange(0,190,30))
        ax.legend()
        plt.xlabel("Joint Angle in degrees")

        ### Plot cycle angles per joint
        idx = np.random.permutation(num_steps)
        cyc_angles = [cyc_angles[i][idx] for i in range(4)]
        for i in range(4):
            ax = fig.add_subplot(gs[2,i])
            mAng = cyc_angles[i].mean(0)
            sAng = cyc_angles[i].std(0)
            xAxis = np.linspace(0,1,len(mAng))
            plt.fill_between(xAxis,mAng-sAng,mAng+sAng,alpha=0.2,color=colors[i])
            plt.plot(xAxis,mAng,label=joints[i],color=colors[i])
            plt.ylim([-5,185])
            plt.title(joints[i])
        cyc_angles = np.array(cyc_angles)
        minAng = cyc_angles.min(2)
        maxAng = cyc_angles.max(2)
        diffAng = maxAng - minAng
        ax = fig.add_subplot(gs[2,i+1:])
        plt.boxplot(diffAng.T,showfliers=True)
        plt.xticks(np.arange(5),['']+joints)
        plt.title("Max-Min angle per cycle")
        plt.ylim([-10,130])
        ax = fig.add_subplot(gs[0,:])
    else:
        fig = plt.figure(figsize=(20,2))

    if time_sync:
         ### Plot sticks
        x = x * dist.max(1).reshape(-1,1)/args.scale + t.reshape(-1,1)
        x = x - (x[:,[-1]] - t.reshape(-1,1))
        plt.plot(x.T,y.T+0.1,'tab:grey',linewidth=0.25)
        plt.plot(x[swing_idx,:].T,y[swing_idx,:].T+0.1,'tab:red',linewidth=0.25)
        plt.stem(t,-0.05*np.ones(len(t)),'tab:grey',markerfmt=" ",basefmt=" ")
        plt.stem(t[swing_idx],-0.05*np.ones(len(swing_idx)),'tab:red',markerfmt=" ",basefmt=" ")
        plt.axis('off')
        plt.ylim([-1,1.1])
    else:
        # Pull tip to floor
        stance_idx = np.delete(np.arange(len(x)),swing_idx)
        y[stance_idx] = y[stance_idx] - y[stance_idx][:,[-1]]

        x = x * dist.max(1).reshape(-1,1)/args.scale
        step = 0
        delta=0.05
        for i in range(len(x)):
            if (swing_idx == i).sum():
                step += delta
                plt.plot(x[i]+step,y[i]+0.1,'tab:red',linewidth=0.25)
            else:
                step += 0.1*delta
                plt.plot(x[i]+step,y[i]+0.1,'tab:grey',linewidth=0.25)
        plt.axis('off')
        plt.ylim([-1,1.1])
#    t = np.linspace(0,dur,len(angles[0]))[::-1]
    plt.savefig(dest+fName.replace('.avi','.pdf'))

    return
