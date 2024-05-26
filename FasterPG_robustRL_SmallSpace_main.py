import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import random
import time
import types
from FasterPG_robustRL_utils import *

num_states=5
num_actions=3

#Garnet problem as in Wang, Yue, and Shaofeng Zou. 
# "Policy gradient method for robust reinforcement learning." 
# International Conference on Machine Learning. PMLR, 2022.

cost=np.ones((num_states,num_actions,num_states))
cost[0,0]=0
cost[range(1,num_states),1]=np.array(range(1,num_states)).reshape(-1,1)/num_states 
xi0=np.ones((num_states,num_actions,num_states))/num_states
rho=np.ones(num_states)/num_states

robustMDP_small=env_setup\
    (seed_init=1,state_space=range(num_states),action_space=range(num_actions),rho=rho,\
     Psi=None,xi0=xi0,xi_radius=0.03,cost=cost,gamma=0.95)

set_seed(1)
Tp=5
Tpi=1
p_update=["PGA", "Frank-Wolfe"][0]
is_normalized_g=False
tau=1e-3   
beta=1e-3
eta=(1-robustMDP_small['gamma'])/tau
FastPG_exact_results=FastPG_exact\
    (robustMDP_small,tau,Tp,Tpi,eta,beta,p_update=p_update,\
     is_normalized_g=is_normalized_g,is_save_p=True,is_save_pi=True,is_print=True)

Tpi=5
Tp=1
eta=10
beta=1e-3
DRPG_results=DRPG(robustMDP_small,Tp,Tpi,eta,beta,is_save_p=True,is_save_pi=True,is_print=True)

Tpi=5
Tp=1
eta=500
alpha_m=1
AC_results=AC(robustMDP_small,Tp,Tpi,eta,alpha_m,pi0=None,p0=None,num_Viter=1000,\
              p_norm_cutoff=1e-8,is_save_p=False,is_save_pi=False,is_print=True)

save_folder='./SmallSpaceResults/'
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
trunc=40
x=FastPG_exact_results['pi_iters']+FastPG_exact_results['p_iters']
index=np.where(x<=trunc)[0]
plt.plot(x[index],FastPG_exact_results['J_max'][index],color="red",marker="P",markevery=1,label="Algorithm 1")
x=DRPG_results['pi_iters']+DRPG_results['p_iters']
index=np.where(x<=trunc)[0]
plt.plot(x[index],DRPG_results['J_max'][index],color="black",linestyle="dashed",marker="v",markevery=1,label="DRPG")
x=AC_results['pi_iters']+AC_results['p_iters']
index=np.where(x<=trunc)[0]
plt.plot(x[index],AC_results['J_max'][index],color="green",linestyle="dotted",marker="*",markevery=2,label="Actor-Critic")
plt.legend()
plt.xlabel('Iteration complexity')
plt.ylabel(r'$\Phi_{\rho}(\pi_t):=\max_{p\in\mathcal{P}}~J_{\rho}(\pi_t,p)$')
plt.savefig(save_folder+'SmallSpaceResult.png',dpi=200, bbox_inches='tight')

hyp_txt=open(save_folder+'/hyperparameters_SmallSpace.txt','w')
hyp_txt.write('Hyperparameters for DRPG--\n')
hyp_txt.write('Outer updates of policy pi: Tpi='\
              +str(DRPG_results['Tpi'])+' iterations, stepsize eta='+str(DRPG_results['eta'])+'\n')

if type(DRPG_results['Tp']) in [int,float]:
    Tp_tmp=DRPG_results['Tp']
else:
    Tp_tmp=DRPG_results['Tp'](0)
hyp_txt.write('Inner updates of transition kernel p: Tp='\
              +str(Tp_tmp)+' iterations, stepsize beta='+str(DRPG_results['beta'])+'\n'+'\n')

    
if type(AC_results['Tp']) in [int,float]:
    Tp_tmp=AC_results['Tp']
else:
    Tp_tmp=AC_results['Tp'](0)
    
if type(AC_results['alpha_m']) in [int,float]:
    alpha_m_tmp=AC_results['alpha_m']
else:
    alpha_m_tmp=AC_results['alpha_m'](0)
    
hyp_txt.write('Hyperparameters for Actor-Critic--\n')
hyp_txt.write('Outer updates of policy pi: Tpi='\
              +str(AC_results['Tpi'])+' iterations, stepsize eta='+str(AC_results['eta'])+'\n')
hyp_txt.write('Inner updates of transition kernel p: Tp='\
              +str(Tp_tmp)+' iterations, stepsize alpha='+str(alpha_m_tmp)+'\n'+'\n')

hyp_txt.write('Hyperparameters for our Algorithm 1--\n')
p_update=FastPG_exact_results['p_update']
if not (p_update=="Frank-Wolfe"):
    p_update="Projected Gradient Ascent"
hyp_txt.write('Outer updates of transition kernel p: Tp='\
              +str(FastPG_exact_results['Tp'])+' iterations, stepsize beta='+str(FastPG_exact_results['beta'])+', update rule='+str(p_update)+'\n')    
hyp_txt.write('Inner updates of policy pi: Tpi='\
              +str(FastPG_exact_results['Tpi'])+' iterations, stepsize eta='+str(FastPG_exact_results['eta']))

hyp_txt.close()

