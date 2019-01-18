# -*- coding: utf-8 -*-
import numpy as np
from sys import path
from numerov import eigensolve, integrate
# from scipy.constants import h,c,e, physical_constants
import time

# eV_in_au = e/physical_constants['atomic unit of energy'][0]

# def eigensolvenergy(l,j,e_approx,params,dx,rmax):
# 	return eigensolve(l,j if j is not None else -1, e_approx,params,dx,rmax)

# From M_Aymar_1991_J._Phys._B 24
paramss = np.load("../modelPot/4param/M_Aymar_1991_J_Phys_B_24.npy")
# paramss = np.array([[  4.0099,   2.1315,  13.023 ,   1.6352], 
# #3.02649,3.24448,12.84206,2.05081 ~ 0.452%
# #3.77590,3.05022,12.97352,1.92428 ~ 0.577%
# 		    		[  4.2056,   2.0186,  12.658 ,   1.5177], 
# #4.23575,1.41471,12.72250,1.44112 ~ 0.114%
# #4.37144,1.33350,12.62975,1.44869 ~ 0.027%
# 		    		[  3.5058,   2.2648,  12.399 ,   1.6187],
# # hardly better ~ 1.166%
# 		    		[  3.7741,   3.1848,  13.232 ,   0.715 ],
# # 3.98287,3.33161,13.30770,0.64801 ~ 2.062%
# 		    		[  3.6   ,   2.3   ,  12.8   ,   0.8   ]])
# #3.53060,0.98481,12.59119,0.36529 ~ 0.574%
# params5s = np.hstack((paramss[:,:3],np.zeros((5,1)),paramss[:,3:]))

# # datae = np.load('../specs/CaII.npz')
# data_ebar = np.load('../specs/CaII_ebar.npz')

# def Diff(l,n_,j_,e_,params,dx=5e-4):
# 	N = n_.size
# 	assert N==e_.size
# 	if j_ is None:
# 		j_ = [None]*N
# 	resid = []
# 	for k in range(N):
# 		# print(k,e_[k],end='\t',flush=True)
# 		resid.append(e_[k]-eigensolvenergy(l,j_[k],e_[k], params, dx, n_[k]*2*(n_[k]+15))[0])
# 		# print(resid[-1],flush=True)
# 	return np.array(resid)

# def Loss(l,n_,j_,e_,params,dx=1e-3):
# 	resid = Diff(l,n_,j_,e_,params,dx)
# 	resid = (resid/e_)**2
# 	return (resid.sum())**0.5

# def SimAnnealParams(l,SOC=True,n_param=5,T=4e-5,Tdecay=0.97,h=0.02,n_iter=50,sd=1):
# 	h_params = np.ones(n_param)*h
# 	rdm.seed(sd)
# 	fp = open("AnnealLog/SOC%s/%dparam/%d/%s.log"\
# 		 %('' if SOC else "free",n_param,l,time.strftime("%m_%d_%Y-%H%M", time.localtime(int(time.time())))), 'w')
# 	print("seed\th",file=fp)
# 	print("%d\t"%sd+"\t".join(["%.5f"%x for x in h_params]),file=fp)
# 	if n_param ==5:
# 		params = params5s[l].copy()
# 	elif n_param==4:
# 		params = paramss[l].copy()
# 	if SOC:
# 		data = datae[str(l)]
# 		j_ = data['j']
# 	else:
# 		data = data_ebar[str(l)]
# 		j_ = None
# 	los = Loss(l, data['n'],j_,data['energy/eV'], params)
# 	kkk = 0
# 	print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T)
# 	print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T,file=fp)
# 	kkk += 1
# 	while True:
# 		kkk_ = kkk+n_iter
# 		while kkk < kkk_:
# 			steps= (2*rdm.rand(params.size)-1)*h_params
# 			los_ = Loss(l, data['n'],j_,data['energy/eV'], params+steps)
# 			if los_<los or (los_>los and rdm.rand() < np.exp((los-los_)/T)):
# 				params += steps
# 				los = los_
# 			print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T)
# 			print("#%d loss=%.5f"%(kkk,los), "("+",".join(["%.5f"%x for x in params])+")", "T=%.2g"%T,file=fp)
# 			T *= Tdecay
# 			kkk += 1
# 		instruct = input("Continue for another %d iteration? (y/n)"%n_iter)
# 		if not (instruct=='y' or instruct=='Y'):
# 			break
# 	fp.close()

def IntegTest(n,l):
	# Test on Z=2 Hydrogen-like ion
	E = -2**2/(2*(n**2))
	return n,l,E,paramss[l]
	# ui,ri = integrate(l,-1,E,paramss[l], 1e-3, n*2*(n+15), 5e-4,1e-7, 0)

if __name__ == '__main__':
	from matplotlib import pyplot as plt
	n,l,E,params = IntegTest(2,1)
	ui,ri = integrate(l,-1,E,paramss[l], 1e-3, n*2*(n+15), 5e-4,1e-7, 0)
	# print("This is a test on Diff")
	# for l in range(2):
	# 	print('L =',l)
	# 	data = data_ebar[str(l)]
	# 	print(data['energy/eV'])
	# 	print(Diff(l,data['n'],None,data['energy/eV'],paramss[l],5e-4))
	
	# l=0
	# k=0
	# data = data_ebar[str(l)]
	# n = data['n'][k]
	# print("On L=%d, n=%d"%(l,n),data['energy/au'][k-1:k+2])
	# # e, u, r = eigensolve(l,-1,data['energy/au'][k],paramss[l],5e-4,n*2*(n+15))

	# ui,ri = integrate(l,-1,data['energy/au'][k], paramss[l],1e-3,n*2*(n+15),5e-4,1e-7,0)
	# from matplotlib import pyplot as plt
	# plt.plot(r,u)
	# plt.show()
	# e = eigensolve(0, 0.5, datae['0']['energy/eV'][5],paramss[0,0],paramss[0,1],paramss[0,2],paramss[0,3], 1e-3, 200)
	# print(datae['0']['energy/eV'][5], e)
	# SimAnnealParams(0,5,.95,n_iter=100)
	# e_est = datae['0']
	# print(eigensolve(0,0.5,,))