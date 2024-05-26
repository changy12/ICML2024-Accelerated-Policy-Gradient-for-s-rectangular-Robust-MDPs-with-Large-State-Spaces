import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import random
import time
import types

#Default values:
num_states=5
num_actions=3

def set_hyps(a,a_default):
    if a is None:
        return a_default
    else:
        return a

def set_seed(seed=1):
    if seed is not None:
        np.random.seed(seed)    
        random.seed(seed)

def env_setup(seed_init=1,state_space=None,action_space=None,rho=None,Psi=None,xi0=None,xi_radius=0.01,cost=None,gamma=0.95):
    #The L2 ambiguity set contains all P(s'|s,a)=<Psi[:,a,s'], xi[s,:]> such that 
    #||xi[s,:]-xi0[s,:]||_2<=xi_radius[s] for all s
    #sum_{s'} <Psi[:,a,s'], xi0[s,:]>  =  sum_{s'} <Psi[:,a,s'], xi[s,:]>  =1
    
    #If Psi=None, then this is tabular case where xi[s,:] parameterizes P(s'|s,a) by reshaping a |A| X |S| matrix.
    # Equivalently, Psi[:,a,s'] can be seen as one-hot with (a,s')-th entry being

    env_dict={}
    set_seed(seed_init)
    env_dict['seed_init']=seed_init
    
    env_dict['state_space']=set_hyps(a=state_space,a_default=range(num_states))
    env_dict['action_space']=set_hyps(a=action_space,a_default=range(num_actions))  
    env_dict['num_states']=len(env_dict['state_space'])
    env_dict['num_actions']=len(env_dict['action_space'])
    
    if rho is None:
        rho=10+np.random.normal(size=(env_dict['num_states']))
    else:
        if isinstance(rho, list):
            rho=np.array(rho)
        assert rho.size==env_dict['num_states'], "rho should have "+str(env_dict['num_states'])+" entries."
    rho=np.abs(rho).reshape(env_dict['num_states'])
    env_dict['rho']=rho/rho.sum()
    
    #The L2 ambiguity set contains all P(s'|s,a)=<Psi[:,a,s'], xi[s,:]> such that 
    #||xi[s,:]-xi0[s,:]||_2<=xi_radius for all s
    #sum_{s'} <Psi[:,a,s'], xi0[s,:]>  =  sum_{s'} <Psi[:,a,s'], xi[s,:]>  =1
    
    #If Psi=None, then this is tabular case where xi[s,a,s'] parameterizes P(s'|s,a)
    # Equivalently, Psi[:,a,s'] can be seen as one-hot with (a,s')-th entry being
    warn="xi_radius should be a float, an int or an np.array with shape (num_states)"
    if type(xi_radius) is np.ndarray:
        xi_radius=xi_radius.reshape(-1)
        assert xi_radius.shape == (num_states,), warn
        env_dict['xi_radius']=xi_radius.copy()
    elif type(xi_radius) in [float,int]:
        env_dict['xi_radius']=xi_radius*np.ones(env_dict['num_states'])
    else:
        assert False, warn

    if Psi is None:
        xi_shape=(env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'])
        if xi0 is None:
            env_dict['xi0']=np.abs(10+np.random.normal(size=xi_shape))   #xi0[s,a,s']
            env_dict['xi0']=env_dict['xi0']/np.sum(env_dict['xi0'],axis=2,keepdims=True)
        else:
            assert type(xi0) is np.ndarray, "xi0 should be an np.ndarray with shape (num_states,num_actions,num_states)" 
            assert xi0.shape == xi_shape, \
                "xi0 should have shape: (num_states,num_actions,num_states)"
            xi0=np.abs(xi0)
            env_dict['xi0']=xi0/np.sum(xi0,axis=2,keepdims=True)
    else: 
        warn="Psi should be None or an np.ndarray with shape (dp,num_actions,num_states)"
        assert type(Psi) is np.ndarray, warn
        assert len(Psi.shape)==3, warn
        env_dict['dp']=Psi.shape[0]
        assert Psi.shape == (env_dict['dp'],env_dict['num_actions'],env_dict['num_states']), warn
        Psi_sums=Psi.sum(axis=2)
        assert np.linalg.matrix_rank(Psi_sums)<env_dict['dp'], "The dp*|A| matrix sum_{s'} Psi(:,:,s') should have rank<dp."
        env_dict['Psi']=Psi.copy()
        env_dict['Psi_proj']=np.linalg.lstsq(Psi_sums, np.identity(env_dict['dp']), rcond=None)[0]
        env_dict['Psi_proj']=Psi_sums.dot(env_dict['Psi_proj'])

        warn="xi0 should be None or an np.ndarray with shape: (num_states,dp)"
        xi_shape=(env_dict['num_states'],env_dict['dp'])
        if xi0 is None: 
            xi0=np.abs(10+np.random.normal(size=xi_shape))
            env_dict['xi0']=xi0/np.sum(xi0,axis=2,keepdims=True)
        else: 
            assert type(xi0) is np.ndarray, warn
            assert xi0.shape == xi_shape, warn
        
        tmp=(Psi_sums.reshape(1,env_dict['dp'],env_dict['num_actions'])\
            *(xi0.reshape(env_dict['num_states'],env_dict['dp'],1))).sum(axis=1)-1 
        assert xi0.min()>=0, "All entries of xi0 must be non-negative."
        assert np.abs(tmp).max()<1e-14, "sum_j Psi[j,a,s']*xi0[s,j] should be 1."  
        env_dict['xi0']=xi0.copy()
    
    env_dict['gamma']=gamma
    
    cost_shape=(env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'])   
    if cost is None:
        env_dict['cost'] = np.random.uniform(size=cost_shape,low=0,high=1)
    else:
        warn="cost should be either None or an np.array with shape (num_states,num_actions,num_states)"
        assert type(cost) is np.ndarray, warn 
        assert cost.shape == cost_shape, warn
        env_dict['cost']=cost.copy()
    return env_dict

def get_s(state_space,xi,s,a,num=1,Psi=None):
    if Psi is None: 
        s_next=np.random.choice(state_space, size=num, p=xi[s,a,:], replace=True)
    else:
        s_next=np.random.choice(state_space, size=num, p=xi[s].dot(Psi[:,a,:]), replace=True)
    if num==1:
        return s_next[0]
    return s_next

def V_robust_iter(pi,xi0,cost,xi_radius,gamma,num_Viter=1000,xi_norm_cutoff=1e-8,Psi=None,Psi_proj=None,V0=None,is_print=False):   #Compute max_p V(pi,p) of certain policy pi via value iteration
    num_states,num_actions=pi.shape
    if V0 is None:
        V=np.zeros(num_states)
    else:
        V=V0.copy()
    for t in range(num_Viter):
        V_past=V.copy()
        if Psi is None:
            vec=pi.reshape(num_states,num_actions,1)*(cost+gamma*V.reshape(1,1,-1))  #vec[s,a,s']=pi(a|s)*[c(s,a,s')+gamma*V(s')]
            xi=vec-(vec.mean(axis=2,keepdims=True))
            xi_norm=np.sqrt((xi*xi).sum(axis=(1,2)))
            coeff=xi_radius/xi_norm
            coeff[xi_norm<=xi_norm_cutoff]=0
            xi=xi0+(xi*coeff.reshape(num_states,1,1))
            V=(xi*vec).sum(axis=(1,2))
        else:
            dp=Psi.shape[0]
            vec=((cost+gamma*V.reshape(1,1,-1)).reshape(1,num_states,num_actions,num_states)*Psi.reshape(dp,1,num_actions,num_states)).sum(axis=3)  
                #vec[j,s,a] for j-th feature dimension
            
            vec=(pi.reshape(1,num_states,num_actions)*vec).sum(axis=2)  #vec[:,s]=sum_{a,s'} pi(a|s)*Psi(:,a,s')*[c(s,a,s')+gamma*V(s')]
            
            #Will project vec[:,s] onto {Psi_sums[:,a]: all actions a} for every s
            if Psi_proj is None:   
                Psi_sums=Psi.sum(axis=2)  #Psi_sums[j,a] for j-th feature dimension 
                xi=np.linalg.lstsq(Psi_sums, vec, rcond=None)[0]
                xi=Psi_sums.dot(xi)
            else: 
                xi=Psi_proj.dot(vec)
            xi=(vec-xi).T
            xi_norm=np.sqrt((xi*xi).sum(axis=1))
            coeff=xi_radius/xi_norm
            coeff[xi_norm<=xi_norm_cutoff]=0
            xi=(xi0+(xi*coeff.reshape(num_states,1)))
            V=(vec.T*xi).sum(axis=1)
        if is_print:
            print('Value iteration '+str(t)+': ||V_{t+1}-V_t||_{infty}='+str(np.abs(V_past-V).max()))        
    return V, xi

def Vp_minpi(p,cost,gamma,num_Viter=1000,V0=None,is_print=False):   
    #Compute max_pi V(pi,p_xi) of certain transition kernel parameter xi via value iteration
    #if Psi is None, then use direct parameterization p_xi=xi. 
    num_states=p.shape[0]
    num_actions=p.shape[1]
    if V0 is None:
        V=np.zeros(num_states)
    else:
        V=V0.copy()
    for t in range(num_Viter):
        V_past=V.copy()
        Q=(p*(cost+gamma*V.reshape(1,1,-1))).sum(axis=2) 
        V=Q.min(axis=1)
        if is_print:
            print('Value iteration '+str(t)+': ||V_{t+1}-V_t||_{infty}='+str(np.abs(V_past-V).max()))        
    return V

def get_p(xi,Psi=None):
    if Psi is None:
        return xi
    dp, num_actions, num_states=Psi.shape
    return (Psi.reshape(1,dp,num_actions,num_states)*xi.reshape(num_states,dp,1,1)).sum(axis=1)

def get_transP_s2s(pi,xi,Psi=None): #Obtain state transition distribution transP_s2s(s,s')=sum_a p(s,a,s')*pi(s,a)
    num_states,num_actions=pi.shape
    if Psi is None:
        return (xi*(pi.reshape(num_states,num_actions,1))).sum(axis=1)
    p=get_p(xi,Psi)
    return (p*(pi.reshape(num_states,num_actions,1))).sum(axis=1)

def stationary_dist(transP_s2s):  #Stationary state distribution corresponding to transP_s2s(s,s')
    evals, evecs = np.linalg.eig(transP_s2s.T)  #P.T*evecs=evecs*np.diag(evals)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = np.abs(evec1[:, 0])
    stationary = evec1 / evec1.sum()
    return stationary.real

def occupation(pi,xi,rho,gamma,Psi=None): #Exact occupation measure
    p=get_p(xi,Psi)
    p2=gamma*p+(1-gamma)*rho.reshape((1,1,-1))
    p_s2s=get_transP_s2s(pi,p2,Psi=None)
    return stationary_dist(p_s2s)

def Jrho_func(pi,xi,rho,cost,gamma,Psi=None):
    num_states, num_actions=pi.shape
    p=get_p(xi,Psi)
    occup=occupation(pi,p,rho,gamma,Psi=None)
    return ((((p*cost).sum(axis=2))*pi).sum(axis=1)*occup).sum()/(1-gamma)

def V_func(pi,xi,cost,gamma,Psi=None):
    p=get_p(xi,Psi)
    p_s2s=get_transP_s2s(pi,p,Psi=None)
    Ecost=((p*cost).sum(axis=2)*pi).sum(axis=1)
    A=np.identity(p_s2s.shape[0])-gamma*p_s2s
    return np.linalg.solve(A,Ecost)

def Q_func(pi,xi,cost,gamma,Psi=None,V=None):
    p=get_p(xi,Psi)
    if V is None:
        V=V_func(pi,p,cost,gamma,Psi=None)
    return (p*(cost+gamma*V.reshape(1,1,-1))).sum(axis=2)   
    #Return matrix Q where Q[s,a] is the Q function value at (s,a)

def proj_L2_xi(xi,xi0,xi_radius,Psi=None,Psi_proj=None):
    if Psi is None:   
        u=xi-xi0
        #For each s, project xi[s,:,:] onto {p: p[s,a,.] is probability vector AND ||xi[s,:,:]-xi0[s,:,:]||_2<=xi_radius[s]}
        u-=u.mean(axis=2,keepdims=True)
        r=np.sqrt((u*u).sum(axis=(1,2)))
        index=(r>xi_radius)
        u[index]*=(xi_radius[index]/r[index]).reshape(-1,1,1)
    else:
        vec=xi-xi0
        if Psi_proj is None:   
            Psi_sums=Psi.sum(axis=2)  #Psi_sums[j,a] for j-th feature dimension
            u=np.linalg.lstsq(Psi_sums, vec.T, rcond=None)[0]
            u=Psi_sums.dot(u)
        else: 
            u=Psi_proj.dot(vec.T)
        u=vec-u.T
        r=np.sqrt((u*u).sum(axis=1))
        index=(r>xi_radius)
        u[index]*=(xi_radius[index]/r[index]).reshape(-1,1)
    return xi0+u
    
def proj_Pr(y):  #Project vector x into probability space
    u=np.flip(np.sort(y))
    D=u.shape[0]
    y=y.reshape(-1)
    a=1-np.sum(u)
    sum_remain=1
    for j in range(D):
        sum_remain-=u[j]
        lambda_now=sum_remain/(j+1)
        if lambda_now+u[j]>0:
            lambda_save=lambda_now
    x=y+lambda_save
    x[x<0]=0
    return x

def proj_L2_pi(pi):
    num_states, num_actions=pi.shape
    pi2=pi.copy()
    for s in range(num_states):
        pi2[s]=proj_Pr(pi[s])
    return pi2

def findP_CPI(pi,Pprime,p0,p_radius,cost,rho,gamma,p_norm_cutoff=1e-8):
    #Algorithm 3.2 of
    # Li, M., Sutter, T., and Kuhn, D. (2023b). Policy gradient algorithms for 
    # robust mdps with non-rectangular uncertainty sets. ArXiv:2305.19004. 
    num_states,num_actions=pi.shape
    ds=occupation(pi,Pprime,rho,gamma,Psi=None).reshape(num_states,1,1)
    V=V_func(pi,Pprime,cost,gamma,Psi=None)
    g=(ds/(1-gamma))*pi.reshape(num_states,num_actions,1)*(cost+gamma*V.reshape(1,1,-1))
    p=g-(g.mean(axis=2,keepdims=True))
    p_norm=np.sqrt((p*p).sum(axis=(1,2)))
    coeff=p_radius/p_norm
    coeff[p_norm<=p_norm_cutoff]=0
    return p0+(p*coeff.reshape(num_states,1,1))

def TD(state_space,action_space,cost,tau,gamma,u,xi,rho,alpha=0.01,N=1,num_iters=1000,\
       Psi=None,Phi=None,w0=None,discard_num=None,is_print=False):
    #For small state space, Psi=Phi=None, and Log-policy u and transition kernel xi are used.
    #If discard_num=None, sample initial state s0 from distribution rho;
    #If discard_num is an integer, sample a trajectory with length discard_num and 
    #   initial state distriution rho, and then obtain the final state as s0. 

    #If N is a function, use batchsize N(t) at iteration t. If N is a positive integer, use batchsize N at each iteration.
    num_states=len(state_space)
    num_actions=len(action_space)
    if type(alpha) in [int,float]:
        alpha1=np.abs(alpha)
        def alpha(n):
            return alpha1
    if type(N) in [int,float]:
        N1=int(np.abs(N))
        if N1==0:
            N1=1
        def N(n):
            return N1
    if Psi is None:
        assert (Phi is None), "Psi and Phi should both be None, or both be np arrays \
            with shapes (dp,num_actions,num_states) and (num_states,num_actions,d) respectively"
        warn="Transition kernel xi should be an np array with shape (num_states,num_actions,num_states)"
        assert type(xi) is np.ndarray, warn
        assert xi.shape==(num_states,num_actions,num_states), warn
        warn="Policy u should be an np array with shape (num_states,num_actions)"
        assert type(u) is np.ndarray, warn
        assert u.shape==(num_states,num_actions), warn
        pi=np.exp(u-u.max(axis=1,keepdims=True))
        pi/=pi.sum(axis=1,keepdims=True)
        if discard_num is None:
            transP_s2s=get_transP_s2s(pi,xi,Psi=None)
            mu=stationary_dist(transP_s2s)
        else:
            warn="discard_num should be None or a positive integer"
            assert isinstance(discard_num,int), warn
            assert discard_num>=1, warn
        if w0 is None:
            w=np.zeros((num_states,num_actions))
        else:
            warn="w0 should be an np array with shape (num_states,num_actions)"
            assert type(w0) is np.ndarray, warn
            assert w0.shape==(num_states,num_actions), warn
            w=w0.copy()
        w_avg=w.copy()
        for t in range(num_iters):
            Nt=N(t)
            alpha_t=alpha(t)/Nt
            w_past=w.copy()
            if is_print:
                w_avg_past=w_avg.copy()   
            for i in range(Nt):
                if discard_num is None:
                    s=np.random.choice(state_space, size=1, p=mu)[0]
                else:
                    s=np.random.choice(state_space, size=1, p=rho)[0]
                    for j in range(discard_num):
                        a=np.random.choice(action_space, size=1, p=u[s])[0]
                        s=get_s(state_space,xi,s,a,num=1,Psi=None)
                a=np.random.choice(action_space, size=1, p=u[s])[0]
                s2=get_s(state_space,xi,s,a,num=1,Psi=None)
                a2=np.random.choice(action_space, size=1, p=u[s2])[0] 
                w[s,a]+=alpha_t*(cost[s,a,s2]+tau*u[s,a]+gamma*w_past[s2,a2]-w_past[s,a])
            weight=1/(t+2)
            w_avg=w_avg*(1-weight)+w*weight
            if is_print:
                print(str(t)+"th step: ||w_{t+1}-w_t||_{infty}="+str(np.abs(w-w_past).max())\
                      +" ||w_avg_{t+1}-w_avg_t||_{infty}="+str(np.abs(w_avg-w_avg_past).max()))
    else:
        warn="u should be an np array with shape (d)"
        assert type(u) is np.ndarray, warn
        u=u.reshape(-1)
        d=u.shape[0]
        warn="xi should be an np array with shape (num_states,dp)"
        assert type(xi) is np.ndarray, warn
        dp=xi.shape[1]
        assert xi.shape==(num_states,dp),warn
        warn="Psi should be an np array with shape (dp,num_actions,num_states)"
        assert type(Psi) is np.ndarray, warn
        assert Psi.shape==(dp,num_actions,num_states), warn
        warn="Phi should be an np array with shape (num_states,num_actions,d)"
        assert type(Phi) is np.ndarray, warn
        assert Phi.shape==(num_states,num_actions,d), warn
        if w0 is None:
            w=np.zeros(d)
        else:
            warn="w0 should be an np array with shape (d)"
            assert type(w0) is np.ndarray, warn
            w=w0.reshape(-1)
            assert w.shape==(d), warn
        w_avg=w.copy()
        if discard_num is None:        
            log_pi=(Phi*(-u/(1-gamma)).reshape((1,1,d))).sum(axis=2)
            log_pi-=log_pi.max(axis=1,keepdims=True)
            pi=np.exp(log_pi)
            pi_sum=pi.sum(axis=1).reshape((-1,1))
            pi/=pi_sum
            log_pi-=np.log(pi_sum) 
            transP_s2s=get_transP_s2s(pi,xi,Psi=Psi)
            mu=stationary_dist(transP_s2s)
            for t in range(num_iters):
                Nt=N(t)
                alpha_t=alpha(t)/Nt
                w_past=w.copy()
                if is_print:
                    w_avg_past=w_avg.copy()           
                for i in range(Nt):                
                    s=np.random.choice(state_space, size=1, p=mu)[0]
                    a=np.random.choice(action_space, size=1, p=pi[s])[0]
                    s2=get_s(state_space,xi,s,a,num=1,Psi=Psi)
                    a2=np.random.choice(action_space, size=1, p=pi[s2])[0]
                    w+=alpha_t*(cost[s,a,s2]+tau*log_pi[s,a]+w_past.dot(gamma*Phi[s2,a2]-Phi[s,a]))*Phi[s,a]
                weight=1/(t+2)
                w_avg=w_avg*(1-weight)+w*weight
                if is_print:
                    print(str(t)+"th step: ||w_{t+1}-w_t||_{infty}="+str(np.abs(w-w_past).max())\
                          +" ||w_avg_{t+1}-w_avg_t||_{infty}="+str(np.abs(w_avg-w_avg_past).max()))
        else:
            for t in range(num_iters):
                Nt=N(t)
                alpha_t=alpha(t)/Nt
                w_past=w.copy()
                if is_print:
                    w_avg_past=w_avg.copy()
                for i in range(Nt):
                    s=np.random.choice(state_space, size=1, p=rho)[0]
                    for j in range(discard_num):
                        log_pis=(Phi[s]*(-u/(1-gamma)).reshape((1,d))).sum(axis=1)
                        log_pis-=log_pis.max()
                        pis=np.exp(log_pis)
                        pis_sum=pis.sum()
                        pis/=pis_sum
                        a=np.random.choice(action_space, size=1, p=pis)[0]
                        s=get_s(state_space,xi,s,a,num=1,Psi=None)
                    log_pis=(Phi[s]*(-u/(1-gamma)).reshape((1,d))).sum(axis=1)
                    log_pis-=log_pis.max()
                    pis=np.exp(log_pis)
                    pis_sum=pis.sum()
                    pis/=pis_sum
                    log_pisa=log_pis[a]-np.log(pis_sum)
                    a=np.random.choice(action_space, size=1, p=pis)[0]
                    s2=get_s(state_space,xi,s,a,num=1,Psi=Psi)
    
                    log_pis2=(Phi[s2]*(-u/(1-gamma)).reshape((1,d))).sum(axis=1)
                    log_pis2-=log_pis2.max()
                    pis2=np.exp(log_pis2)
                    pis2_sum=pis2.sum()
                    pis2/=pis2_sum
                    a2=np.random.choice(action_space, size=1, p=pis2)[0]            
                    w+=alpha_t*(cost[s,a,s2]+tau*log_pisa+w_past.dot(gamma*Phi[s2,a2]-Phi[s,a]))
                weight=1/(t+2)
                w_avg=w_avg*(1-weight)+w*weight
                if is_print:
                    print(str(t)+"th step: ||w_{t+1}-w_t||_{infty}="+str(np.abs(w-w_past).max())\
                          +" ||w_avg_{t+1}-w_avg_t||_{infty}="+str(np.abs(w_avg-w_avg_past).max()))
    return w_avg,w

def stoc_grad(state_space,action_space,cost,tau,gamma,u,xi,w,rho,H=50,N=100,Psi=None,Phi=None,is_print=False):
    #For small state space, Psi=None,Psi=None. log-policy u, transition kernel xi and Q function w are used. 
    num_states=len(state_space)
    num_actions=len(action_space)
    pHi=gamma**np.array(range(H))
    pHi/=pHi.sum()
    if Psi is None:
        assert (Phi is None), "Psi and Phi should both be None, or both be np arrays \
            with shapes (dp,num_actions,num_states) and (num_states,num_actions,d) respectively"
        warn="Transition kernel xi should be an np array with shape (num_states,num_actions,num_states)"
        assert type(xi) is np.ndarray, warn
        assert xi.shape==(num_states,num_actions,num_states), warn
        warn="Policy u should be an np array with shape (num_states,num_actions)"
        assert type(u) is np.ndarray, warn
        assert u.shape==(num_states,num_actions), warn
        warn="w should be an np array with shape (num_states,num_actions)"
        assert type(w) is np.ndarray, warn
        assert w.shape==(num_states,num_actions), warn
        pi=np.exp(u-u.max(axis=1,keepdims=True))
        pi_sum=pi.sum(axis=1).reshape((-1,1))
        pi/=pi_sum
        coeff=1/(N*(1-gamma))
        weights=np.zeros((num_states,1,1))
        for i in range(N):
            if is_print:
                print("Now computing the "+str(i)+"-th sample among the "+str(N)+" samples.")
            Hi=np.random.choice(range(H), size=1, p=pHi, replace=True)[0]
            s=np.random.choice(state_space, size=1, p=rho)[0]
            for n in range(Hi):
                a=np.random.choice(action_space, size=1, p=pi[s])[0]
                s=get_s(state_space,xi,s,a,num=1,Psi=None)
            weights[s]+=coeff
        g=weights*(cost+(tau*u).reshape((num_states,num_actions,1))+\
                   gamma*(pi*w).sum(axis=1).reshape(1,1,-1))*pi.reshape((num_states,num_actions,1))
    else:
        warn="u should be an np array with shape (d)"
        assert type(u) is np.ndarray, warn
        u=u.reshape(-1)
        d=u.shape[0]
        warn="xi should be an np array with shape (num_states,dp)"
        assert type(u) is np.ndarray, warn
        dp=xi.shape[1]
        assert xi.shape==(num_states,dp),warn
        warn="Psi should be an np array with shape (dp,num_actions,num_states)"
        assert type(Psi) is np.ndarray, warn
        assert Psi.shape==(dp,num_actions,num_states), warn
        warn="Phi should be an np array with shape (num_states,num_actions,d)"
        assert type(Phi) is np.ndarray, warn
        assert Phi.shape==(num_states,num_actions,d), warn
        warn="w should be an np array with shape (d)"
        assert type(w) is np.ndarray, warn
        assert w.shape==(d,), warn
        g=np.zeros((num_states,dp))
        coeff=N*(1-gamma)
        for i in range(N):
            if is_print:
                print("Now computing the "+str(i)+"-th sample among the "+str(N)+" samples.")
            Hi=np.random.choice(range(H), size=1, p=pHi, replace=True)[0]
            s=np.random.choice(state_space, size=1, p=rho)[0]
            for n in range(Hi):
                log_pis=(Phi[s]*(-u/(1-gamma)).reshape((1,d))).sum(axis=1)
                log_pis-=log_pis.max()
                pis=np.exp(log_pis)
                pis_sum=pis.sum()
                pis/=pis_sum
                a=np.random.choice(action_space, size=1, p=pis)[0]
                s=get_s(state_space,xi,s,a,num=1,Psi=Psi)
                
            log_pis=(Phi[s]*(-u/(1-gamma)).reshape((1,d))).sum(axis=1)
            log_pis-=log_pis.max()
            pis=np.exp(log_pis)
            pis_sum=pis.sum()
            pis/=pis_sum
            a=np.random.choice(action_space, size=1, p=pis)[0]
            log_pisa=log_pis[a]-np.log(pis_sum)

            s2=get_s(state_space,xi,s,a,num=1,Psi=Psi)            
            log_pis2=(Phi[s2]*(-u/(1-gamma)).reshape((1,d))).sum(axis=1)
            log_pis2-=log_pis2.max()
            pis2=np.exp(log_pis2)
            pis2_sum=pis2.sum()
            pis2/=pis2_sum
            a2=np.random.choice(action_space, size=1, p=pis)[0]
            pxi=(Psi[:,a,s2]*xi[s]).sum()
            sum3=cost[s,a,s2]+tau*log_pisa+gamma*(Phi[s2,a2]*w).sum()
            g[s]+=(sum3/(pxi*coeff))*Psi[:,a,s2]
    return g

def FastPG_exact(env_dict,tau,Tp,Tpi,eta,beta,p0=None,p_update=["PGA", "Frank-Wolfe"][0],num_Viter=1000,p_norm_cutoff=1e-8,is_normalized_g=False,is_save_p=False,is_save_pi=False,is_print=True):   #Our Algorithm 1 under deterministic case (i.e., eps1=eps2=0)
    #p_update="PGA" or "wolfe" means to update transition kernel p using projected gradient ascent (PGA) or Frank-Wolfe respectively
    if type(tau) in [int,float]:
        cc=tau
        def tau(t):
            return cc
    if type(eta) in [int,float]:
        cc1=eta
        def eta(t):
            return cc1
    if type(beta) in [int,float]:
        cc2=beta
        def beta(t):
            return cc2
    
    shape=Tp+1
    results={'J_max':np.zeros(shape),'J':np.zeros(shape),'Jtau_max':np.zeros(shape),'Jtau':np.zeros(shape),\
             'Fp':np.zeros(shape),'p_iters':np.zeros(shape),'pi_iters':np.zeros(shape),'p_norm_cutoff':p_norm_cutoff,\
             'tau':tau,'Tp':Tp,'Tpi':Tpi,'eta':eta,'beta':beta,'num_Viter':num_Viter,'p_update':p_update}
    if p0 is None:
        p=env_dict['xi0'].copy()
    else:
        assert p0.shape==(env_dict['num_states'],env_dict['num_actions'],env_dict['num_states']), \
            "p0 should be None or an np array with shape (env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'])"
        p=p0.copy()
    
    negLog_A=-np.log(env_dict['num_actions'])
    pi=np.ones((env_dict['num_states'],env_dict['num_actions']))/env_dict['num_actions']
    log_pi=negLog_A*np.ones((env_dict['num_states'],env_dict['num_actions'])) 
    tau_now=tau(0)
    reg_cost=env_dict['cost']+tau_now*negLog_A

    results['J'][0]=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None).dot(env_dict['rho'])
    results['J_max'][0]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],\
                                num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
    results['Jtau'][0]=V_func(pi,p,reg_cost,env_dict['gamma'],Psi=None).dot(env_dict['rho'])
    results['Jtau_max'][0]=V_robust_iter(pi,env_dict['xi0'],reg_cost,env_dict['xi_radius'],env_dict['gamma'],\
                                num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
    results['Fp'][0]=Vp_minpi(p,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
    results['p_iters'][0]=0
    results['pi_iters'][0]=0
    if is_save_pi:
        results['pi']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions']))
        results['pi'][0]=pi.copy()
        results['log_pi']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions']))
        results['log_pi'][0]=log_pi.copy()
    if is_save_p:
        results['p']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'],))
        results['p'][0]=p.copy()
        
    for t in range(Tp):
        pi=np.ones((env_dict['num_states'],env_dict['num_actions']))/env_dict['num_actions']
        log_pi=np.log(pi[0,0])*np.ones((env_dict['num_states'],env_dict['num_actions']))
        tau_now=tau(t)
        for k in range(Tpi):
            eta2=eta(k)/(1-env_dict['gamma'])
            reg_cost=env_dict['cost']+tau_now*log_pi.reshape(env_dict['num_states'],env_dict['num_actions'],1)
            Q=Q_func(pi,p,reg_cost,env_dict['gamma'],Psi=None,V=None) 
            log_pi-=eta2*Q
            log_pi-=log_pi.max(axis=1,keepdims=True)
            pi=np.exp(log_pi)
            sum1=pi.sum(axis=1).reshape(-1,1)
            log_pi-=np.log(sum1)
            pi/=sum1
            if np.max(np.abs(np.exp(log_pi)-pi))>1e-14:
                assert False, "Value Error: exp(log_pi) should almost be equal to pi"

        ds=occupation(pi,p,env_dict['rho'],env_dict['gamma'],Psi=None).reshape(env_dict['num_states'],1,1)
        reg_cost=env_dict['cost']+tau_now*log_pi.reshape(env_dict['num_states'],env_dict['num_actions'],1)
        V=V_func(pi,p,reg_cost,env_dict['gamma'],Psi=None)
        
        results['J'][t+1]=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None).dot(env_dict['rho'])
        results['J_max'][t+1]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],\
                                    num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
        results['Jtau'][t+1]=V_func(pi,p,reg_cost,env_dict['gamma'],Psi=None).dot(env_dict['rho'])
        results['Jtau_max'][t+1]=V_robust_iter(pi,env_dict['xi0'],reg_cost,env_dict['xi_radius'],env_dict['gamma'],\
                                    num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
        results['Fp'][t+1]=Vp_minpi(p,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
        results['p_iters'][t+1]=t
        results['pi_iters'][t+1]=(t+1)*Tpi
        if is_save_pi:
            results['pi'][t+1]=pi.copy()
            results['log_pi'][t+1]=log_pi.copy()
        if is_save_p:
            results['p'][t+1]=p.copy()
        if is_print:
            print("Iteration "+str(t)+": J_max="+str(results['J_max'][t])+", J="+str(results['J'][t])\
                  +", Jtau_max="+str(results['Jtau_max'][t])+", Jtau="+str(results['Jtau'][t])+", Fp="+str(results['Fp'][t]))
        
        if p_update=="Frank-Wolfe":
            p_eps=findP_CPI(pi,p,env_dict['xi0'],env_dict['xi_radius'],reg_cost,\
                            env_dict['rho'],env_dict['gamma'],p_norm_cutoff)
            beta_t=beta(t)
            p=(1-beta_t)*p+beta_t*p_eps
        else:
            if is_normalized_g:
                g=ds*pi.reshape(env_dict['num_states'],env_dict['num_actions'],1)\
                    *(reg_cost+env_dict['gamma']*V.reshape(1,1,-1))
                beta2=beta(t)
                g*=beta2/np.sqrt(np.sum(g*g))
            else:
                beta2=beta(t)/(1-env_dict['gamma'])
                g=(beta2*ds)*pi.reshape(env_dict['num_states'],env_dict['num_actions'],1)\
                    *(reg_cost+env_dict['gamma']*V.reshape(1,1,-1))
            p=proj_L2_xi(p+g,env_dict['xi0'],env_dict['xi_radius'],Psi=None,Psi_proj=None)
    return results

def FastPG_stoc(env_dict,tau,Tp,Tpi,eta,beta,Phi=None,xi0=None,w0=None,\
                alpha_TD=1e-3,N_TD=1,iters_TD=100000,H_grad=2000,N_grad=10000,num_Viter=1000,p_norm_cutoff=1e-8,\
                   is_save_p=False,is_save_pi=False,is_print=True,progress_txt='./progress.txt'):   
    #If Psi is None, use transition kernel xi. 
    #If Psi=Phi=None, implement our Algorithm 2; Otherwise, implement Algorithm 3
    if type(tau) in [int,float]:
        cc=tau
        def tau(t):
            return cc
    if type(eta) in [int,float]:
        cc1=eta
        def eta(t):
            return cc1
    if type(beta) in [int,float]:
        cc2=beta
        def beta(t):
            return cc2
    
    shape=Tp+1
    results={'J_max':np.zeros(shape),'J':np.zeros(shape),'Jtau_max':np.zeros(shape),'Jtau':np.zeros(shape),'Fp':np.zeros(shape),\
             'p_iters':np.zeros(shape),'pi_iters':np.zeros(shape),'p_norm_cutoff':p_norm_cutoff,'tau':tau,'Tp':Tp,'Tpi':Tpi,'eta':eta,\
             'beta':beta,'num_Viter':num_Viter,'Phi':Phi,'alpha_TD':alpha_TD,"N_TD":N_TD,"iters_TD":iters_TD,"H_grad":H_grad,"N_grad":N_grad}

    if env_dict['Psi'] is None:
        Phi=None
        xi_shape=(env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'])
        if xi0 is None:
            xi=env_dict['xi0'].copy()
        else:
            assert xi0.shape==xi_shape, \
                "if env_dict['Psi'] is None, xi0 should be None or an np array with shape (env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'])"
            xi=xi0.copy()
    else:
        warn="Phi should an np array with shape is env_dict['Psi'] is an np array."
        assert type(Phi) is np.ndarray, warn
        assert len(Phi.shape)==3, warn
        d=Phi.shape[2]
        assert Phi.shape[0]==env_dict['num_states'] and Phi.shape[1]==env_dict['num_actions'],warn
        
        dp=env_dict['Psi'].shape[0]
        xi_shape=(env_dict['num_states'],dp)
        if xi0 is None:
            xi=env_dict['xi0'].copy()
        else:
            assert xi0.shape==xi_shape, \
                "if env_dict['Psi'] is np array, xi0 should be None or an np array with shape (env_dict['num_states'],dp)"
            xi=xi0.copy()
            
    tau_now=tau(0)
    negLog_A=-np.log(env_dict['num_actions'])
    reg_cost=env_dict['cost']+tau_now*negLog_A
    pi=np.ones((env_dict['num_states'],env_dict['num_actions']))/env_dict['num_actions']
    log_pi=np.ones((env_dict['num_states'],env_dict['num_actions']))*negLog_A
    p=get_p(xi,env_dict['Psi'])
    results['J'][0]=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None).dot(env_dict['rho']) 
    results['J_max'][0]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],num_Viter=num_Viter,\
                                      Psi=env_dict['Psi'] ,Psi_proj=env_dict['Psi_proj'],V0=None,is_print=False)[0].dot(env_dict['rho'])
    results['Jtau'][0]=V_func(pi,p,reg_cost,env_dict['gamma'],Psi=None).dot(env_dict['rho'])
    results['Jtau_max'][0]=V_robust_iter(pi,env_dict['xi0'],reg_cost,env_dict['xi_radius'],env_dict['gamma'],num_Viter=num_Viter,\
                                         Psi=env_dict['Psi'],Psi_proj=env_dict['Psi_proj'],V0=None,is_print=False)[0].dot(env_dict['rho'])
    results['Fp'][0]=Vp_minpi(p,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
    results['p_iters'][0]=0
    results['pi_iters'][0]=0
    if is_save_pi:
        results['pi']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions']))
        results['pi'][0]=pi.copy()
        results['log_pi']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions']))
        results['log_pi'][0]=log_pi.copy()
    if is_save_p:
        results['p']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions'],env_dict['num_states']))
        results['p'][0]=p.copy()

    if Phi is None:            
        assert False, "The stochastic algorithm under Phi=None is under construction"
    else:
        for t in range(Tp):
            tau_now=tau(t)
            u=np.zeros(d)
            for k in range(Tpi):
                hyp_txt=open(progress_txt,'a')
                hyp_txt.write('Inner t='+str(t)+', Outer k='+str(k)+'\n')
                hyp_txt.close()
                eta2=eta(k)
                # reg_cost=env_dict['cost']+tau_now*log_pi.reshape(env_dict['num_states'],env_dict['num_actions'],1) 
                
                w_avg,w=TD(env_dict['state_space'],env_dict['action_space'],env_dict['cost'],tau_now,env_dict['gamma'],\
                            u=u,xi=xi,rho=env_dict['rho'],alpha=alpha_TD,N=N_TD,num_iters=iters_TD,\
                            Psi=env_dict['Psi'],Phi=Phi,w0=None,discard_num=None,is_print=False)
            
                u+=eta2*w_avg
            log_pi=(Phi*(-u/(1-env_dict['gamma'])).reshape(1,1,d)).sum(axis=2)
            log_pi-=log_pi.max(axis=1,keepdims=True)
            pi=np.exp(log_pi)
            pi_sum=pi.sum(axis=1).reshape((-1,1))
            pi/=pi_sum
            log_pi-=np.log(pi_sum) 
            p=get_p(xi,env_dict['Psi'])
            ds=occupation(pi,p,env_dict['rho'],env_dict['gamma'],Psi=None).reshape(env_dict['num_states'],1,1)
            reg_cost=env_dict['cost']+tau_now*log_pi.reshape(env_dict['num_states'],env_dict['num_actions'],1)
            V=V_func(pi,p,reg_cost,env_dict['gamma'],Psi=None)
            results['J'][t+1]=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None).dot(env_dict['rho'])
            results['J_max'][t+1]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],num_Viter=num_Viter,\
                                              Psi=env_dict['Psi'] ,Psi_proj=env_dict['Psi_proj'],V0=None,is_print=False)[0].dot(env_dict['rho'])
            results['Jtau'][t+1]=V_func(pi,p,reg_cost,env_dict['gamma'],Psi=None).dot(env_dict['rho'])
            results['Jtau_max'][t+1]=V_robust_iter(pi,env_dict['xi0'],reg_cost,env_dict['xi_radius'],env_dict['gamma'],num_Viter=num_Viter,\
                                                 Psi=env_dict['Psi'],Psi_proj=env_dict['Psi_proj'],V0=None,is_print=False)[0].dot(env_dict['rho'])
            results['Fp'][t+1]=Vp_minpi(p,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
            results['p_iters'][t+1]=t
            results['pi_iters'][t+1]=(t+1)*Tpi
            if is_save_pi:
                results['pi'][t+1]=pi.copy()
                results['log_pi'][t+1]=log_pi.copy()
            if is_save_p:
                results['p'][t+1]=p.copy()
            if is_print:
                print("Iteration "+str(t)+": J_max="+str(results['J_max'][t])+", J="+str(results['J'][t])\
                      +", Jtau_max="+str(results['Jtau_max'][t])+", Jtau="+str(results['Jtau'][t])+", Fp="+str(results['Fp'][t]))
            
            w_avg,w=TD(env_dict['state_space'],env_dict['action_space'],env_dict['cost'],tau_now,env_dict['gamma'],\
                       u=u,xi=xi,rho=env_dict['rho'],alpha=alpha_TD,N=N_TD,num_iters=iters_TD,\
                       Psi=env_dict['Psi'],Phi=Phi,w0=None,discard_num=None,is_print=False)        
            g=stoc_grad(env_dict['state_space'],env_dict['action_space'],env_dict['cost'],tau_now,env_dict['gamma'],\
                        u,xi,w_avg,env_dict['rho'],H=H_grad,N=N_grad,Psi=env_dict['Psi'],Phi=Phi,is_print=False)     
            xi=proj_L2_xi(xi+beta(t)*g,env_dict['xi0'],env_dict['xi_radius'],Psi=env_dict['Psi'],Psi_proj=env_dict['Psi_proj'])
    return results

def DRPG(env_dict,Tp,Tpi,eta,beta,pi0=None,p0=None,num_Viter=1000,is_save_p=False,is_save_pi=False,is_print=True):   
    #DRPG algorithm in Wang, Qiuhao, Chin Pang Ho, and Marek Petrik. 
    # "Policy gradient in robust mdps with global convergence guarantee." 
    # International Conference on Machine Learning. PMLR, 2023.
    
    shape=Tpi+1
    results={'J_max':np.zeros(shape),'J':np.zeros(shape),'p_iters':np.zeros(shape),'pi_iters':np.zeros(shape),\
             'Fp':np.zeros(shape),'Tp':Tp,'Tpi':Tpi,'eta':eta,'beta':beta,'num_Viter':num_Viter}
    
    if type(Tp) in [int,float]:
        cc=Tp
        def Tp(t):
            return cc
        
    if p0 is None:
        p0=env_dict['xi0'].copy()
    else:
        assert p0.shape==(env_dict['num_states'],env_dict['num_actions'],env_dict['num_states']), \
            "p0 should be None or an np array with shape (env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'])"

    if pi0 is None:
        pi=np.ones((env_dict['num_states'],env_dict['num_actions']))/env_dict['num_actions']
    else:
        assert pi0.shape==(env_dict['num_states'],env_dict['num_actions']), \
            "pi0 should be None or an np array with shape (env_dict['num_states'],env_dict['num_actions'])"
        pi=pi0.copy()
    results['J'][0]=V_func(pi,p0,env_dict['cost'],env_dict['gamma'],Psi=None).dot(env_dict['rho'])
    results['J_max'][0]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],\
                                num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
    results['Fp'][0]=Vp_minpi(p0,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
    results['p_iters'][0]=0
    results['pi_iters'][0]=0
    if is_save_pi:
        results['pi']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions']))
        results['pi'][0]=pi.copy()
    if is_save_p:
        results['p']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'],))
        results['p'][0]=p0.copy()
    eta2=eta/(1-env_dict['gamma'])
    beta2=beta/(1-env_dict['gamma'])
    for t in range(Tpi):
        Tp_now=Tp(t)
        pi2=pi.reshape(env_dict['num_states'],env_dict['num_actions'],1)
        p=p0.copy()
        # Jmax_p=-np.Inf
        for k in range(Tp_now):
            ds=occupation(pi,p,env_dict['rho'],env_dict['gamma'],Psi=None).reshape(env_dict['num_states'],1,1)   
            V=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None)  
            Jnow=V.dot(env_dict['rho'])  
            p=proj_L2_xi(p+beta2*ds*pi2*(env_dict['cost']+env_dict['gamma']*V.reshape(1,1,-1)),\
                         env_dict['xi0'],env_dict['xi_radius'],Psi=None,Psi_proj=None)  
                
        V=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None)
        if is_print:
            print("Iteration "+str(t)+": J_max="+str(results['J_max'][t])+", J="+str(results['J'][t])\
                  +", Fp="+str(results['Fp'][t]))
        ds=occupation(pi,p,env_dict['rho'],env_dict['gamma'],Psi=None).reshape(env_dict['num_states'],1)
        Q=Q_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None,V=V)
        pi=proj_L2_pi(pi-eta2*ds*Q)    
        
        V=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None)
        Jnow=V.dot(env_dict['rho'])
        results['J'][t+1]=Jnow
        results['J_max'][t+1]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],\
                                    num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
        results['p_iters'][t+1]=results['p_iters'][t]+Tp_now
        results['pi_iters'][t+1]=t+1
        results['Fp'][t+1]=Vp_minpi(p,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
        if is_save_pi:
            results['pi'][t+1]=pi.copy()
        if is_save_p:
            results['p'][t+1]=p.copy()
    return results


def AC(env_dict,Tp,Tpi,eta,alpha_m,pi0=None,p0=None,num_Viter=1000,p_norm_cutoff=1e-8,is_save_p=False,is_save_pi=False,is_print=True):   
    #Actor-Critic (AC) algorithm is Algorithm 4.1 in 
    # Li, M., Sutter, T., and Kuhn, D. (2023b). Policy gradient algorithms for 
    # robust mdps with non-rectangular uncertainty sets. ArXiv:2305.19004. 
    shape=Tpi+1
    results={'J_max':np.zeros(shape),'J':np.zeros(shape),'p_iters':np.zeros(shape),'pi_iters':np.zeros(shape),\
             'Fp':np.zeros(shape),'Tp':Tp,'Tpi':Tpi,'eta':eta,'alpha_m':alpha_m,'num_Viter':num_Viter}
        
    if type(Tp) in [int,float]:
        cc1=Tp
        def Tp(t):
            return cc1
    
    if type(alpha_m) in [int,float]:
        cc2=alpha_m
        def alpha_m(t):
            return cc2
    
    if p0 is None:
        p0=env_dict['xi0'].copy()
    else:
        assert p0.shape==(env_dict['num_states'],env_dict['num_actions'],env_dict['num_states']), \
            "p0 should be None or an np array with shape (env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'])"

    if pi0 is None:
        pi=np.ones((env_dict['num_states'],env_dict['num_actions']))/env_dict['num_actions']
    else:
        assert pi0.shape==(env_dict['num_states'],env_dict['num_actions']), \
            "pi0 should be None or an np array with shape (env_dict['num_states'],env_dict['num_actions'])"
        pi=pi0.copy()
    results['J'][0]=V_func(pi,p0,env_dict['cost'],env_dict['gamma'],Psi=None).dot(env_dict['rho'])
    results['J_max'][0]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],\
                                num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
    results['Fp'][0]=Vp_minpi(p0,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
    results['p_iters'][0]=0
    results['pi_iters'][0]=0
    if is_save_pi:
        results['pi']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions']))
        results['pi'][0]=pi.copy()
    if is_save_p:
        results['p']=np.zeros((shape,env_dict['num_states'],env_dict['num_actions'],env_dict['num_states'],))
        results['p'][0]=p0.copy()
    eta2=eta/(1-env_dict['gamma'])
    for t in range(Tpi):
        Tp_now=Tp(t)
        p=p0.copy()        
        for k in range(Tp_now):    
            p_eps=findP_CPI(pi,p,env_dict['xi0'],env_dict['xi_radius'],env_dict['cost'],\
                            env_dict['rho'],env_dict['gamma'],p_norm_cutoff)
            alpha_m_now=alpha_m(k)
            p=(1-alpha_m_now)*p+alpha_m_now*p_eps
        V=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None)
        if is_print:
            print("Iteration "+str(t)+": J_max="+str(results['J_max'][t])+", J="+str(results['J'][t])\
                  +", Fp="+str(results['Fp'][t]))
        ds=occupation(pi,p,env_dict['rho'],env_dict['gamma'],Psi=None).reshape(env_dict['num_states'],1)
        Q=Q_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None,V=V)
        pi=proj_L2_pi(pi-eta2*ds*Q)
        
        V=V_func(pi,p,env_dict['cost'],env_dict['gamma'],Psi=None)
        Jnow=V.dot(env_dict['rho'])
        results['J'][t+1]=Jnow
        results['J_max'][t+1]=V_robust_iter(pi,env_dict['xi0'],env_dict['cost'],env_dict['xi_radius'],env_dict['gamma'],\
                                    num_Viter=num_Viter,Psi=None,Psi_proj=None,V0=None,is_print=False)[0].dot(env_dict['rho'])
        results['Fp'][t+1]=Vp_minpi(p,env_dict['cost'],env_dict['gamma'],num_Viter=num_Viter,V0=None,is_print=False).dot(env_dict['rho'])
        results['p_iters'][t+1]=results['p_iters'][t]+Tp_now
        results['pi_iters'][t+1]=t+1
        if is_save_pi:
            results['pi'][t+1]=pi.copy()
        if is_save_p:
            results['p'][t+1]=p.copy()
    return results

