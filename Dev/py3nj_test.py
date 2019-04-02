import numpy as np
from sys import path
path.append("./Numerov")
import pickle
from numerov import integrate, eigensolve, socXi, uInnerProd_1d
from sympy.physics.wigner import gaunt
from sympy.physics.wigner import clebsch_gordan as CG_sym
from sympy import N
from py3nj import wigner3j
from py3nj import clebsch_gordan as CG_for

def angulaV_sym(l,l1_,l2_,L_,mL_,l1,l2,L,mL):
	if mL != mL_ or L!=L_:
		return 0
	if l > max(l1_+l1,l2_+l2) or l < min(abs(l1-l1_),abs(l2-l2_)):
		return 0
	res = 0
	# print(range(max(-l1,mL-l2),min(l1,mL+l2)+1))
	for m1 in range(max(-l1,mL-l2),min(l1,mL+l2)+1):
		# print("m1 =",m1,range(max(-l1_,mL_-l2_,m1-l),min(l1,mL_+l2_,m1+l)+1))
		for m1_ in range(max(-l1_,mL_-l2_,m1-l),min(l1,mL_+l2_,m1+l)+1):
			m = m1_-m1
			res += N(CG_sym(l1_,l2_,L_, m1_,mL_-m1_,mL_)*CG_sym(l1, l2, L,  m1, mL -m1, mL )*gaunt(l1_,l,l1,-m1_,m,m1)*gaunt(l2_,l,l2,-(mL_-m1_),-m,mL-m1))\
				 * (-1 if (mL+m)%2 == 1 else 1)
			# print("\t\t", res)
		# print("")
	return res

def gaunt_for(two_l1, two_l2, two_l3, two_m1, two_m2, two_m3):
	return ((two_l1+1)*(two_l2+1)*(two_l3+1)/(4*np.pi))**0.5*wigner3j(two_l1,two_l2,two_l3,two_m1,two_m2,two_m3)*wigner3j(two_l1,two_l2,two_l3,0,0,0)

def angulaV_for(l,l1_,l2_,L_,mL_,l1,l2,L,mL):
	if mL != mL_ or L!=L_:
		return 0
	if l > max(l1_+l1,l2_+l2) or l < min(abs(l1-l1_),abs(l2-l2_)):
		return 0
	tw_l,tw_l1_,tw_l2_,tw_l1,tw_l2,tw_L,tw_mL = 2*l,2*l1_,2*l2_,2*l1,2*l2,2*L,2*mL
	res = 0
	# print(range(max(-l1,mL-l2),min(l1,mL+l2)+1))
	for tw_m1 in range(2*max(-l1,mL-l2),2*min(l1,mL+l2)+1,2):
		# print("m1 =",m1,range(max(-l1_,mL_-l2_,m1-l),min(l1,mL_+l2_,m1+l)+1))
		for tw_m1_ in range(max(-tw_l1_,tw_mL-tw_l2_,tw_m1-tw_l),min(tw_l1,tw_mL+tw_l2_,tw_m1+tw_l)+1):
			tw_m = tw_m1_-tw_m1
			res += CG_for(tw_l1_,tw_l2_,tw_L, tw_m1_,tw_mL-tw_m1_,tw_mL)\
				  *CG_sym(tw_l1, tw_l2, tw_L, tw_m1, tw_mL-tw_m1, tw_mL)\
				  *gaunt_for(tw_l1_,tw_l,tw_l1,-tw_m1_,tw_m,tw_m1)\
				  *gaunt_for(tw_l2_,tw_l,tw_l2,-(tw_mL-tw_m1_),-tw_m,tw_mL-tw_m1)\
				 * (-1 if (tw_mL+tw_m)%4 == 2 else 1)
			# print("\t\t", res)
		# print("")
	return res

if __name__ == '__main__':
	tests = [(0,0,0,0,0,0,0,0,0),(0,1,0,1,0,1,0,1,0),(0,1,0,1,1,1,0,1,1)]
	for t in tests:
		print(angulaV_for(*t),angulaV_sym(*t))
# time efficiency
# %timeit [angulaV_for(*t) for t in tests]                                
# 2.8 ms +- 47.2 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
# %timeit [angulaV_sym(*t) for t in tests]                                
# 1.66 ms +- 14.5 us per loop (mean +- std. dev. of 7 runs, 1000 loops each)


