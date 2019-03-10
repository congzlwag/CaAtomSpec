# -*- coding: utf-8 -*-
import numpy as np
from numerov import integrate, eigensolve
import matplotlib.pyplot as plt
import pickle

data_ebar = np.load('../specs/CaII_ebar.npz')
with open("../specs/ebar_quant_defect.pickle",'rb') as f:
    ebar_qd = pickle.load(f)
paramss = np.load('../modelPotUparams/5param/Feb2.npy')
# paramss = np.load('../modelPotUparams/4param/M_Aymar_1991_J_Phys_B_24.npy')
dx = 1e-2

def request_energy(n,l, dt_ebar=data_ebar, qdefect=ebar_qd):
	if str(l) in dt_ebar.keys():
		data = dt_ebar[str(l)]
		n_ = data['n']
		if n in n_:
			return data['energy/au'][n-n_[0]]
		return -2 / ((qdefect[str(l)](n))**2)
	else:
		assert l > 4
		return -2 / ((qdefect['4'](n))**2)

single_ur_database = {}
def request_ur(n,l,dtbs=single_ur_database,Uparams=paramss,dt_ebar=data_ebar, qdefect=ebar_qd):
	if (n,l) in dtbs.keys():
		return dtbs[(n,l)]
	energ = request_energy(n,l,dt_ebar,qdefect)
	u_, r_ = integrate(l, -1, energ, paramss[min(l,3)], 1e-4, n*2*(n+15), dx, 1e-8, 0)
	dtbs[(n,l)] = (u_,r_)
	return (u_,r_)

if __name__ == '__main__':
	for idx, nl in enumerate([(4,0),(4,1),(4,2),(3,2)]):
		n,l = nl
		ax = plt.subplot(2,2,idx+1)
		ui, ri = request_ur(n,l)
		ax.plot(ri**0.5,ui,'--',c='C%d'%(idx+1), label='(%d,%d) Integrate'%nl)
		e, ue, re = eigensolve(l,-1,request_energy(n,l),paramss[l], dx, n*2*(n+15),True)
		ax.plot(re**0.5,ue,':',c='C%d'%(idx+1), label='(%d,%d) Eigensolve'%nl)
		ax.legend()
		ax.set_xlim(0,6)
	# plt.legend()
	# plt.xlim(0,6)
	plt.show()
	# pass