import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import random
import time
import types
from FasterPG_robustRL_utils import *

num_states=50
num_actions=3
dp=10


#Garnet problem as in Wang, Yue, and Shaofeng Zou. 
# "Policy gradient method for robust reinforcement learning." 
# International Conference on Machine Learning. PMLR, 2022.

cost=np.ones((num_states,num_actions,num_states))
cost[0,0]=0
cost[range(1,num_states),1]=0
rho=np.ones(num_states)/num_states

set_seed(1)
Psi=np.random.uniform(size=(dp,num_actions,num_states),low=1,high=2)
Psi/=Psi.sum(axis=2,keepdims=True)
Psi_sum=Psi.sum(axis=2)
xi0=np.ones((num_states,dp))/dp
robustMDP_big=env_setup\
    (seed_init=1,state_space=range(num_states),action_space=range(num_actions),rho=rho,\
     Psi=Psi,xi0=xi0,xi_radius=4e-3,cost=cost,gamma=0.95)  

tau=0.1 
Tpi=20
Tp=10
eta=(1-robustMDP_big['gamma'])/tau
eta=1.0 #eta=0.1 is good, see if eta=1 can be better. 
xi0=robustMDP_big['xi0'].copy()
beta=0.001
alpha_TD=0.001
N_TD=1
iters_TD=100000
H_grad=500
N_grad=10000
num_Viter=1000
progress_txt='./progress20240210.txt'


FastPG_BigSpace_results={}
d_range=[150,149,130,100,50,20,5]

for d in d_range:
    print("Begin d="+str(d))
    set_seed(1)
    Phi=np.random.uniform(size=(num_states,num_actions,d),low=0,high=1)
    set_seed(1)
    FastPG_BigSpace_results[d]=FastPG_stoc(robustMDP_big,tau,Tp,Tpi,eta,beta,Phi=Phi,xi0=xi0,w0=None,alpha_TD=alpha_TD,N_TD=N_TD,\
                    iters_TD=iters_TD,H_grad=H_grad,N_grad=N_grad,num_Viter=num_Viter,p_norm_cutoff=1e-8,is_save_p=True,\
                        is_save_pi=True,is_print=True,progress_txt=progress_txt)
    print("\n\n")

save_folder='./BigSpaceResults/'
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
plt.figure()
colors=['red','black','blue','green','cyan','grey','magenta']
k=0
for d in d_range:
    trunc=np.inf
    x=FastPG_BigSpace_results[d]['pi_iters']+FastPG_BigSpace_results[d]['p_iters']
    index=np.where(x<=trunc)[0]
    plt.plot(x[index],FastPG_BigSpace_results[d]['J_max'][index],color=colors[k],label="d="+str(d))
    k+=1
    save_folder1=save_folder+'d'+str(d)+'/'
    if not os.path.isdir(save_folder1):
        os.makedirs(save_folder1)
    for key1 in ['J_max','J','Jtau_max','Jtau']:
        np.save(save_folder1+key1+'.npy',FastPG_BigSpace_results[d][key1])
plt.legend(loc=4)
plt.xlabel('Iteration complexity')
plt.ylabel(r'$\Phi_{\rho}(\pi_t):=\max_{p\in\mathcal{P}}~J_{\rho}(\pi_t,p)$')
plt.savefig(save_folder+'BigSpaceResult.png',dpi=200, bbox_inches='tight')

d=d_range[0]
hyp_txt=open(save_folder+'hyperparameters.txt','w')
hyp_txt.write('Hyperparameters--\n')
hyp_txt.write('tau='+str(tau)+'\n')
hyp_txt.write('Outer updates of transition kernel p: Tp='\
              +str(FastPG_BigSpace_results[d]['Tp'])+' iterations, stepsize beta='+str(FastPG_BigSpace_results[d]['beta'](0))+'\n')    
hyp_txt.write('Inner updates of policy pi: Tpi='\
              +str(FastPG_BigSpace_results[d]['Tpi'])+' iterations, stepsize eta='+str(FastPG_BigSpace_results[d]['eta'](0))+"\n")
hyp_txt.write('TD: stepsize alpha='+str(alpha_TD)+', batchsize N='+str(N_TD)+' with '+str(iters_TD)+' iterations\n')
hyp_txt.write('stochastic gradient: truncation level H='+str(H_grad)+', batchsize N_grad='+str(N_grad)+'\n')
hyp_txt.write(str(num_Viter)+' value iterations.\n')
hyp_txt.write("Range of d is "+str(d_range)+"\n")
hyp_txt.close()



