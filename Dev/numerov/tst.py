# -*- coding: utf-8 -*-
import numpy as np
from sys import path, argv
typ = argv[1]
if typ == "numerov":
	from numerov import integrate
elif typ == "numeroveig":
	from numeroveig import integrate, eigensolve
else:
	from numerov import debug
	from numeroveig import debug as debugeig
import matplotlib.pyplot as plt

paramss = np.load("../modelPotUparams/4param/M_Aymar_1991_J_Phys_B_24.npy")

datae = np.load('../specs/CaII.npz')

if __name__ == '__main__':
	l = 0
	data = datae[str(l)]
	_k= 3
	j = data['j'][_k]
	energy = data['energy/au'][_k]
	n = data['n'][_k]
	params = paramss[l]
	a1,a2,a3,rc = params
	a4 = 0
	rmax = 2*n*(n+15)
	dx = 1e-3
	rmin = 1e-6
	print("n =",n,"j =",j,"energy/au =",energy)
	init1,init2 = (1e-8,0)
	
	if typ == "debug":
		w1_,r_ = debug(rmin, rmax, dx, init1, init2, l, 0.5, j,\
			  energy, 3.26, 20, 2, a1,a2,a3,a4,rc,0.9999862724756698)
		params = np.array([a1,a2,a3,a4,rc])
		w2_,r_ = debugeig(l, j, energy, params, rmin, rmax, dx, init1, init2)
		diffw = w1_ - w2_
		print(abs(diffw).max())
		plt.plot(r_**0.5,diffw)
		plt.show()
	else:
		if typ == "numerov":
			sol = integrate(rmin, rmax, dx, init1, init2, l, 0.5, j,\
				  energy, 3.26, 20, 2, a1,a2,a3,a4,rc,0.9999862724756698)
			r_ = sol[1]
			u_ = sol[0]
			u_ /= np.trapz(u_**2,r_)**0.5
		elif typ == "numeroveig":
			params = np.array([a1,a2,a3,a4,rc])
			# u_, r_ = integrate(l, j, energy, params, rmin, rmax, dx, init1, init2)
			eexact, u_, r_ = eigensolve(l, j, energy, params, dx, rmax)
			print(eexact)
			u_ /= np.trapz(u_**2,r_)**0.5
		plt.plot(r_**0.5,u_)
		plt.show()
		# plt.savefig("correct.pdf")