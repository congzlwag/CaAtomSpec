# -*- coding: utf-8 -*-
import numpy as np
from numpy import random as rdm
from specPreProcess import cm_1_to_eV
from sys import path
path.append("./Numerov")
from numerov import eigensolve
import time

# def eigEnergy(l,j,e_approx,params,dx,rmax):
# 	if len(params) == 5:
# 		a1,a2,a3,a4,rc = params
# 	elif len(params) == 4:
# 		a1,a2,a3,rc = params
# 		a4 = 0
# 	else:
# 		raise ValueError("len(params) should be 4 or 5")
# 	return eigE(l,j if j is not None else -1, e_approx,a1,a2,a3,a4,rc,dx,rmax)

# From M_Aymar_1991_J._Phys._B 24
paramss = np.load("modelPotUparams/4param/M_Aymar_1991_J_Phys_B_24.npy")
params5s = np.hstack((paramss[:,:3],np.zeros((4,1)),paramss[:,3:]))
# params5s = np.load('modelPotUparams/5param/Jan23.npy')

# datae = np.load('specs/CaII.npz')
data_ebar = np.load('specs/CaII_ebar.npz')
data_ebar_ex = np.load('specs/CaII_ebar_ex.npz')

def Diff(l,n_,j_,e_,params,dx=5e-4):
	N = n_.size
	assert N==e_.size
	if j_ is None:
		j_ = [-1]*N
	resid = []
	for k in range(N):
		# print(k,e_[k],end='\t',flush=True)
		resid.append(e_[k]-eigensolve(l,j_[k],e_[k], params, dx, n_[k]*2*(n_[k]+15), False))
		# print(resid[-1],flush=True)
	return np.array(resid)

def Loss(l,n_,j_,e_,params,dx=1e-3):
	# if isinstance(l,int):
	resid = Diff(l,n_,j_,e_,params,dx)
	resid = (resid/e_)**2
	w_ = 1/n_
	w_ /= w_.sum()
	resid *= w_
	return (resid.sum())**0.5
	# else:
	# 	ni = np.asarray([n__.size for n__ in n_])
	# 	losses = np.asarray([Loss(l[i],n_[i],j_[i],e_[i],params,dx) for i in range(len(l))])
	# 	return (((losses**2)*ni).sum() / ni.sum())**0.5

def SimAnnealParams(l,data,SOC=True,n_param=5,T=4e-5,Tdecay=0.97,h=0.02,n_iter=50,sd=1):
	if isinstance(h,float):
		h_params = np.ones(n_param)*h
	else:
		h_params = np.asanyarray(h)
	rdm.seed(sd)
	fp = open("AnnealLog/SOC%s/%dparam/%d/%s.log"\
		 %('' if SOC else "free",n_param,l,time.strftime("%m_%d_%Y-%H%M", time.localtime(int(time.time())))), 'w')
	print("seed\th",file=fp)
	print("%d\t"%sd+"\t".join(["%.5f"%x for x in h_params]),file=fp)
	if n_param ==5:
		params = params5s[l].copy()
	elif n_param==4:
		params = paramss[l].copy()
	if SOC:
		# data = datae[str(l)]
		j_ = data['j']
	else:
		# data = data_ebar[str(l)]
		j_ = None
	los = Loss(l, data['n'],j_,data['energy/au'], params)
	kkk = 0
	print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T)
	print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T,file=fp)
	kkk += 1
	while True:
		kkk_ = kkk+n_iter
		while kkk < kkk_:
			steps= (2*rdm.rand(params.size)-1)*h_params
			los_ = Loss(l, data['n'],j_,data['energy/au'], params+steps)
			if los_<los or (los_>los and rdm.rand() < np.exp((los-los_)/T)):
				params += steps
				los = los_
			print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T)
			print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T,file=fp)
			T *= Tdecay
			kkk += 1
		instruct = input("Continue for another %d iteration? ([y]/n)"%n_iter)
		if (instruct=='n' or instruct=='N' or instruct=='q'):
			break
	fp.close()

if __name__ == '__main__':
	# params5s[3] = np.array([2.38524,3.37088,11.14158,0.13348,0.80602])
	# paramss[0] = np.array([4.64179,1.84771,11.42727,2.40568])
	# SimAnnealParams(3,data_ebar_ex['3'][:15],False,5,T=1e-3,Tdecay=0.99,h=np.array([0.02,0.01,0.05,0.01,0.005])*2,sd=1126,n_iter=100)
	#												np.array([0.01,0.01,0.05,0.001])*0.5
	# print("This is a test on Diff")
	for l in range(4):
		print('L =',l)
		data = data_ebar[str(l)]
		# print(data['energy/au'])
		print(Loss(l,data['n'],None,data['energy/au'],paramss[l],1e-3))
	l = 4
	print("L =",l)
	data = data_ebar[str(l)]
	print(Loss(l,data['n'],None,data['energy/au'],paramss[l-1],1e-3))
	# l=1
	# data = data_ebar[str(l)]
	# print(Diff(l,data['n'],None,data['energy/au'],params5s[l],dx=1e-3))
	# k=2
	# data = data_ebar[str(l)]
	# _n = data['n'][k]
	# print(_n, data['energy/au'][k])
	# e = eigensolve(l, -1, data['energy/au'][k], params5s[2], 1e-3, _n*2*(_n+15), False)


	# e = eigE(0, 0.5, datae['0']['energy/eV'][5],paramss[0,0],paramss[0,1],paramss[0,2],paramss[0,3], 1e-3, 200)
	# print(datae['0']['energy/eV'][5], e)
	# SimAnnealParams(0,5,.95,n_iter=100)
	# e_est = datae['0']
	# print(eigE(0,0.5,,))