# -*- coding: utf-8 -*-
import numpy as np
from sys import path
path.append("./Numerov")
import pickle
from numerov import integrate, eigensolve, socXi, uInnerProd_1d
from sympy.physics.wigner import gaunt, clebsch_gordan
from sympy import N
from scipy.linalg import eigh
from tqdm import tqdm

data_ebar = np.load('specs/CaII_ebar.npz')
with open("specs/ebar_quant_defect.pickle",'rb') as f:
    ebar_qd = pickle.load(f)
# paramss = np.load('modelPotUparams/5param/Feb2.npy')
paramss = np.load('modelPotUparams/4param/M_Aymar_1991_J_Phys_B_24.npy')

n_lowest_valence = 4
dx = 5e-3

# class SingleElectronBase:
# 	def __init__(self, n,l,ml,ms):
# 		self.n = n
# 		self.l = l
# 		self.ml = ml
# 		self.ms = ms
# 		self.get_energy()
# 		self.get_ur()

# 	def get_energy(self):
# 		if not hasattr(self,'_energy'):
# 			data = data_ebar[str(self.l)]
# 			n_ = data['n']
# 			if self.n in n_:
# 				self._energy = data['energy/au'][self.n-n_[0]]
# 			else:
# 				self._energy = -2 / (ebar_qd[str(self.l)](self.n)**2)
# 		return self._energy

# 	def get_ur(self):
# 		if not hasattr(self,'_u_'):
# 			self._u_, self._r_ = \
# 				integrate(self.l,-1,self._energy, paramss[self.l], 1e-3, self.n*2*(self.n+15), 5e-3, 1e-8,0)
# 		return self._u_, self._r_

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
def request_ur(n,l,dtbs=single_ur_database,Uparams=paramss,dt_ebar=data_ebar, qdefect=ebar_qd, eig=False):
	if (n,l) in dtbs.keys():
		return dtbs[(n,l)]
	energ = request_energy(n,l,dt_ebar,qdefect)
	if not eig:
		u_, r_ = integrate(l, -1, energ, paramss[min(l,3)], 1e-3, n*2*(n+15), dx, 1e-8, 0)
	else:
		e_, u_, r_ = eigensolve(l,-1, energ, paramss[min(l,3)],dx, n*2*(n+15),True)
	dtbs[(n,l)] = (u_,r_)
	return (u_,r_)

class LS2eState:
	def __init__(self, e_config, L,mL, S,mS):
		n1,l1,n2,l2 = e_config
		assert n1+0.7*l1 >= n_lowest_valence and n2+0.7*l2 >= n_lowest_valence
		assert n1>l1 and n2>l2
		if n1+0.7*l1 > n2+0.7*l2:
			self.e_config = (n2,l2,n1,l1)
		else:
			self.e_config = e_config
		self.L = L
		self.mL = mL
		self.S = S 
		self.mS = mS
		self._Energy_unperturb = request_energy(*(self.e_config[:2]))+request_energy(*(self.e_config[2:]))
	
	def defineBasis_LSrestricted(self, l_range, n1_range, n2_range):
		n1, l1, n2, l2 = self.e_config
		if not (n1 in range(*n1_range) and n2 in range(*n2_range)):
			raise ValueError("n ranges do not cover this state (n1,n2)=(%d,%d)"%(n1,n2))
		if not (l1 in range(*l_range) and l2 in range(*l_range)):
			raise ValueError("l range does not cover l1=%d or l2=%d"%(l1,l2))
		self.relevant_basis = []
		for nn1 in range(*n1_range):
			# n_lowest_valence <= nn1 + 0.7 ll1
			for ll1 in range(max(int(np.ceil(10*(n_lowest_valence-nn1)/7)),l_range[0]),min(l_range[1],nn1)):
				# nn1 + 0.7 ll1 <= nn2 + 0.7 ll2 <= nn2 + 0.7 (nn2 - 1) = 1.7 nn2 -0.7
				for nn2 in range(max(int(np.ceil((10*nn1+7*ll1+7)/17)),n2_range[0]), n2_range[1]):
					# nn1 + 0.7 ll1 <= nn2 + 0.7 ll2
					for ll2 in range(max(int(np.ceil((10*nn1+7*ll1-10*nn2)/7)),l_range[0],abs(ll1-self.L)),min(l_range[1],nn2,ll1+self.L+1)):
						if (nn1,ll1,nn2,ll2) != self.e_config:
							b = LS2eState((nn1,ll1,nn2,ll2),self.L,self.mL,self.S,self.mS)
							if b.normalization_factor() > 0:
								self.relevant_basis.append(LS2eState((nn1,ll1,nn2,ll2),self.L,self.mL,self.S,self.mS))
						else:
							self.relevant_basis.append(self)
							self.idx_in_relev_basis = len(self.relevant_basis) -1
	def displayBasis(self):
		print("%d states in the subspace of (L,mL)=(%d,%d), (S,mS)=(%d,%d) are relevant"%(len(self.relevant_basis), self.L,self.mL,self.S,self.mS))
		print("e config.",[s.e_config for s in self.relevant_basis])

	def V_mat_construct(self):
		if not hasattr(self,'relevant_basis'):
			raise AttributeError("Please define relevant_basis first")
		self.V_mat = np.empty((len(self.relevant_basis),len(self.relevant_basis)),'d')
		for i,bv in enumerate(self.relevant_basis):
			for j_i,bu in tqdm(enumerate(self.relevant_basis[i:]), desc='Row %d'%i, unit='mat.entry'):
				j = i+j_i
				self.V_mat[i,j] = V_mat_element_LScoupledStates(bv,bu)
				if j!= i:
					self.V_mat[j,i] = self.V_mat[i,j].conj()
		return self.V_mat

	def diagonalize(self):
		if not hasattr(self,"V_mat"):
			self.V_mat_construct()
		H = self.V_mat.copy()
		for i, b in enumerate(self.relevant_basis):
			H[i,i] += b._Energy_unperturb
		S = np.identity(len(self.relevant_basis))
		for i, bi in enumerate(self.relevant_basis):
			for j_i_1, bj in enumerate(self.relevant_basis[i+1:]):
				j = i+j_i_1+1
				S[i,j] = inner_prod(bi,bj)
				S[j,i] = S[i,j]
		print("Unperturbed energy = %.4f, 1st order perturbed = %.4f"%(self._Energy_unperturb, H[self.idx_in_relev_basis,self.idx_in_relev_basis]))
		print("Max amplitude of residue S due to radial functions = %.2g"%(abs(S-np.identity(S.shape[0])).max()))
		w, v = eigh(H, b=S)
		idx_in_mat_v = np.argmax(abs(v[self.idx_in_relev_basis]))
		print("Col. Index in transition mat v is", idx_in_mat_v)
		self._Energy = w[idx_in_mat_v]
		return w, v

	def normalization_factor(self):
		if self.e_config[:2]==self.e_config[2:]:
			if (self.S+self.L)%2==1:
				return 0
			else:
				return 2
		else:
			return 2**0.5

def V_mat_element_LScoupledStates(LS2e_,LS2e):
	"""<LS2e_|\frac{1}{r_{12}}|LS2e>
	We can calculate only those mL=L, because L_z commute with 1/r_{12} and thus is still degenerate
	"""
	if LS2e_.S != LS2e.S or LS2e_.mS != LS2e.mS or LS2e_.L != LS2e.L or LS2e_.mL != LS2e.mL:
		return 0
	n1_, l1_, n2_, l2_ = LS2e_.e_config
	n1,  l1,  n2,  l2  = LS2e.e_config
	v1,rv1 = request_ur(n1_,l1_)
	v2,rv2 = request_ur(n2_,l2_)
	u1,ru1 = request_ur(n1,l1)
	u2,ru2 = request_ur(n2,l2)
	normas = LS2e_.normalization_factor()*LS2e.normalization_factor()
	if normas==0:
		return 0
	if l1==l2:
		res = 0
		for l in range(min(abs(l1-l1_),abs(l2-l2_)), max(l1_+l1,l2_+l2)+1):
			angula = angularIntegral_V12(l, l1_,l2_,LS2e_.L,LS2e_.mL, l1,l2,LS2e.L,LS2e.mL)
			if abs(angula) < 1e-15:
				continue
			M = gridMaxR_l(l,rv1,rv2)
			radial = radialIntegralSandwich(M,v1,rv1,v2,rv2,u1,ru1,u2,ru2)
			if n1==n2: 
				# assert (L+S)%2==0
				radial += radial
			else:
				radial += (-1 if (LS2e.L+LS2e.S)%2==1 else 1)*radialIntegralSandwich(M,v1,rv1,v2,rv2,u2,ru2,u1,ru1)
			res += radial*angula
		return res * 2/normas
	elif l1_==l2_:
		return np.conj(V_mat_element_LScoupledStates(LS2e,LS2e_))
	else:
		# print("l1!=l2")
		res = 0
		Mdict = {}
		for l in range(min(abs(l1-l1_),abs(l2-l2_)), max(l1_+l1,l2_+l2)+1):
			# print("l =",l)
			angula = angularIntegral_V12(l, l1_,l2_,LS2e_.L,LS2e_.mL, l1,l2,LS2e.L,LS2e.mL)
			if abs(angula) < 1e-15:
				# print('angula==0')
				continue
			M = gridMaxR_l(l,rv1,rv2)
			Mdict[l] = M
			radial = radialIntegralSandwich(M,v1,rv1,v2,rv2,u1,ru1,u2,ru2)
			res += radial*angula
		res_ex = 0
		# swap (l1,l2), (n1,n2), (u1,u2), (ru1,ru2)
		for l in range(min(abs(l2-l1_),abs(l1-l2_)), max(l1_+l2,l2_+l1)+1):
			angula = angularIntegral_V12(l, l1_,l2_,LS2e_.L,LS2e_.mL, l2,l1,LS2e.L,LS2e.mL)
			if abs(angula) < 1e-15:
				continue
			if l in Mdict.keys():
				M = Mdict[l]
			else:
				M = gridMaxR_l(l,rv1,rv2)
			radial = radialIntegralSandwich(M,v1,rv1,v2,rv2,u2,ru2,u1,ru1)
			res_ex += radial*angula
		return (res + (-1 if (LS2e.S+LS2e.L-l1-l2)%2==1 else 1)*res_ex)*2 / normas

def inner_prod(LS2e_, LS2e):
	"""<LS2e_|LS2e>
	"""
	if LS2e_.S != LS2e.S or LS2e_.mS != LS2e.mS or LS2e_.L != LS2e.L or LS2e_.mL != LS2e.mL:
		return 0
	normas = LS2e_.normalization_factor()*LS2e.normalization_factor()
	if normas==0:
		return 0
	n1_, l1_, n2_, l2_ = LS2e_.e_config
	n1,  l1,  n2,  l2  = LS2e.e_config
	v1,rv1 = request_ur(n1_,l1_)
	v2,rv2 = request_ur(n2_,l2_)
	u1,ru1 = request_ur(n1,l1)
	u2,ru2 = request_ur(n2,l2)
	res = 0
	# first term
	if l1_==l1 and l2_==l2:
		res += uInnerProd_1d(v1, rv1, u1, ru1)*uInnerProd_1d(v2, rv2, u2, ru2)
	# exchanged term
	if l1_==l2 and l2_==l1:
		res += (-1 if (LS2e.S+LS2e.L-l1-l2)%2==1 else 1)*\
			   uInnerProd_1d(v1, rv1, u2, ru2)*uInnerProd_1d(v2, rv2, u1, ru1)
	return res*2 / normas

def angularIntegral_V12(l,l1_,l2_,L_,mL_,l1,l2,L,mL):
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
			res += N(clebsch_gordan(l1_,l2_,L_, m1_,mL_-m1_,mL_)*clebsch_gordan(l1, l2, L,  m1, mL -m1, mL )*gaunt(l1_,l,l1,-m1_,m,m1)*gaunt(l2_,l,l2,-(mL_-m1_),-m,mL-m1))\
				 * (-1 if (mL+m)%2 == 1 else 1)
			# print("\t\t", res)
		# print("")
	return res

def radialIntegralSandwich(Ml,v1,rv1,v2,rv2,u1,ru1,u2,ru2):
	vM = (v1.reshape(-1,1)*v2.reshape(1,-1))*Ml
	u= (u1.reshape(-1,1)*u2.reshape(1,-1))
	return uInnerProd_2d(vM,rv1,rv2, u,ru1,ru2)

def gridMaxR_l(l,r1,r2):
	r1g,r2g = np.meshgrid(r1,r2,indexing='ij')
	rmax = np.maximum(r1g,r2g)
	rmin = np.minimum(r1g,r2g)
	M = ((rmin/rmax)**l)/rmax
	return M*4*np.pi/(2*l+1)

def uInnerProd_2d(v,av,bv,u,au,bu):
	"""
	two 2D functions v & u are gridded on b1\otimes b2 and a1\otimes a2 respectively
	return the integral \int_0^{+\infty}dr_1\int_0^{+\infty}dr_2, a1 & b1 are grids on the r_1 axis, a2 & b2 are grids on the r_2 axis
	use trapezoid integral on the internal nodes
	use triangular integral between 0 and the first node
	"""
	assert v.shape == (av.size, bv.size)
	assert u.shape == (au.size, bu.size)
	res, current_a, last_intg, av_last,v_last, au_last,u_last = (0,0,0,0,0,0,0)
	kv, ku = (0,0)
	while (kv<av.size and ku<au.size):
		if (av[kv]<=au[ku]):
			a_incr = av[kv] - current_a
			current_a = av[kv]
			if (a_incr>0):
				intg = uInnerProd_1d(v[kv], bv, u[ku]+(u[ku]-u_last)/(au[ku]-au_last)*(current_a - au_last), bu)
				res += a_incr*0.5*(last_intg+intg)
			av_last = av[kv]
			v_last = v[kv]
			kv += 1
		else:
			a_incr = au[ku] - current_a
			current_a = au[ku]
			if (a_incr>0):
				intg = uInnerProd_1d(v[kv]+(v[kv]-v_last)/(av[kv]-av_last)*(current_a - av_last), bv, u[ku], bu)
				res += a_incr*0.5*(last_intg+intg)
			au_last = au[ku]
			u_last = u[ku]
			ku += 1
		last_intg = intg
	return res

if __name__ == '__main__':

	# print(angularIntegral_V12(2,2,2,4,4,2,2,4,4))
	# u1, r1 = request_ur(4,1)
	# u2, r2 = request_ur(4,2)
	# M = np.ones((r1.size, r2.size))
	# print(uInnerProd_1d(u1,r1,u2,r2)**2)

	# print(radialIntegralSandwich(M, u1,r1, u2,r2,u2,r2,u1,r1))
	# M = gridMaxR_l(0,r1,r2)

	ground = LS2eState((4,0,4,0),0,0,0,0)
	ground.defineBasis_LSrestricted((0,10),(3,6),(3,6))
	ground.displayBasis()
	w,v = ground.diagonalize()
	print(ground._Energy)

	# highR1 = LS2eState((4,0,18,0),0,0,0,0)
	# highR1.defineBasis_LSrestricted((0,5),(3,8),(16,21))
	# highR1.displayBasis()
	# w,v = highR1.diagonalize()
	# print(highR1._Energy)
	# np.savez('4s18s.npz', eigvals=w, eigvecs=v)

	# excite1 = LS2eState((4,0,4,1),1,1,0,0)
	# excite2 = LS2eState((4,0,5,1),1,1,0,0)
	# print(V_mat_element_LScoupledStates(excite1,excite2))
	# excite1_ = LS2eState((4,0,4,1),1,0,0,0)
	# excite2_ = LS2eState((4,0,5,1),1,0,0,0)
	# print(V_mat_element_LScoupledStates(excite1_,excite2_))
	pass

