# -*- coding: utf-8 -*-
import numpy as np
from sys import path
path.append("./Numerov")
import pickle
from numerov import integrate, eigensolve, socXi, uInnerProd_1d
from sympy.physics.wigner import gaunt, clebsch_gordan
from sympy import N, sqrt
from scipy.linalg import eigh
from tqdm import tqdm
from time import time
from matplotlib import pyplot as plt 


data_ebar = np.load('specs/CaII_ebar.npz')
with open("specs/ebar_quant_defect.pickle",'rb') as f:
    ebar_qd = pickle.load(f)
# paramss = np.load('modelPotUparams/5param/Feb2.npy')
paramss = np.load('modelPotUparams/4param/M_Aymar_1991_J_Phys_B_24.npy')

n_lowest_valence = 4
dx = 5e-3

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

# def level_order(n,l, low_bound=n_lowest_valence):
# 	res = n+0.7*l
# 	if res >= low_bound:
# 		return res
# 	else:
# 		return None

def levelAhigher(nlA,nlB,low_bound = n_lowest_valence):
	a = 10*nlA[0] + 7*nlA[1]
	b = 10*nlB[0] + 7*nlB[1]
	if a < low_bound or b < low_bound:
		return None
	return a > b or a==b and nlA[0] > nlB[0]

def unperturbedEnergy(gamma):
	return request_energy(*(gamma[:2]))+request_energy(*(gamma[2:]))

class LS2eState:
	def __init__(self, L, S, J, e_config, gamma_basis=None,coordinates=None):
		self.L = L
		self.S = S
		self.J = J
		if levelAhigher(e_config[:2],e_config[2:]):
			e_config = (*(e_config[2:]),*(e_config[:2]))
		self.e_config = e_config
		self.setGammaBasis(gamma_basis)
		self.coordinates = coordinates
		assert self.normalizationFactor(e_config) > 0
		self.SOCfree = (self.S ==0 or self.L==0)
		self._h0diag = None
		self._Vmat = None
		self._Smat = None
		self._SOCmat = None

	def setGammaBasis(self,gb):
		reset = False
		if gb is None:
			self.__gamma_basis = [self.e_config]
			self.idx_in_basis = 0
			self.coordinates = np.ones((1,))
			reset = True
		else:
			if gb != self.__gamma_basis:
				self.__gamma_basis = []
				idx_in_basis = None
				# print(gb)
				for g in (gb):
					n1,l1,n2,l2 = g
					if self.L > l1+l2 or self.L < abs(l1-l2):
						continue
					if self.normalizationFactor(g) == 0:
						continue
					if levelAhigher((n1,l1),(n2,l2)): # then swap
						g = (*(g[2:]),*(g[:2]))
					if g == self.e_config:
						idx_in_basis = len(self.__gamma_basis)
					self.__gamma_basis.append(g)
				if idx_in_basis is None:
					raise ValueError("%s is not found in these electron configurations"%str(self.e_config))
				# self.__gamma_basis = gb
				self.idx_in_basis = idx_in_basis
				self.coordinates = None
				reset = True
		if reset:
			self._h0diag = None
			self._Vmat = None
			self._Smat = None
			self._SOCmat = None

	def appendGammaBasis(self,gb):
		appending = []
		for g in gb:
			n1,l1,n2,l2 = g
			if self.L > l1+l2 or self.L < abs(l1-l2):
				continue
			if self.normalizationFactor(g) == 0:
				continue
			if levelAhigher((n1,l1),(n2,l2)):
				g = (*(g[2:]),*(g[:2]))
			for g1 in self.__gamma_basis:
				if g==g1:
					continue
			appending.append(g)
		if len(appending) > 0:
			pass # extend the matrices
		self.__gamma_basis = self.__gamma_basis + appending

	def normalizationFactor(self, gamma):
		if gamma[:2]==gamma[2:]:
			if (self.S+self.L)%2==1:
				return 0
			else:
				return 2
		else:
			return 2**0.5
	
	def defineBasis_LSrestricted(self, l_range, n1_range, n2_range):
		n1, l1, n2, l2 = self.e_config
		if not (n1 in range(*n1_range) and n2 in range(*n2_range)):
			raise ValueError("n ranges do not cover this state (n1,n2)=(%d,%d)"%(n1,n2))
		if not (l1 in range(*l_range) and l2 in range(*l_range)):
			raise ValueError("l range does not cover l1=%d or l2=%d"%(l1,l2))
		g_basis = []
		for nn1 in range(*n1_range):
			for ll1 in range(l_range[0],min(l_range[1],nn1)):
				if levelAhigher((n_lowest_valence,0),(nn1,ll1)):
					continue
				# nn1 + 0.7 ll1 <= nn2 + 0.7 ll2 <= nn2 + 0.7 (nn2 - 1) = 1.7 nn2 -0.7
				for nn2 in range(n2_range[0], n2_range[1]):
					# nn1 + 0.7 ll1 <= nn2 + 0.7 ll2
					for ll2 in range(max(l_range[0],abs(ll1-self.L)),min(l_range[1],nn2,ll1+self.L+1)):
						if levelAhigher((nn1,ll1),(nn2,ll2)):
							continue
						g_basis.append((nn1,ll1,nn2,ll2))
		self.setGammaBasis(g_basis)

	def displayBasis(self):
		print("%d e configs in the subspace of (L,S,J)=(%d,%d,%d) are relevant to"%(len(self.__gamma_basis), self.L,self.S,self.J),self.e_config)
		print(self.__gamma_basis)

	def _h0diagConstruct(self):
		self._h0diag = np.asarray([unperturbedEnergy(g) for g in self.__gamma_basis])
		return self._h0diag

	def _matConstruct(self,attrname):
		"""matname in ['_Vmat','_SOCmat','_Smat']"""
		# attrname = "_%smat"%matname
		print("Construct",attrname[1:])
		if self.SOCfree and attrname=='_SOCmat':
			return None
		setattr(self,attrname,np.empty((len(self.__gamma_basis),len(self.__gamma_basis)),'d'))
		for i,bv in enumerate(self.__gamma_basis):
			for j_i,bu in tqdm(enumerate(self.__gamma_basis[i:]), desc='Row %d'%i, unit='mat.entry'):
				j = i+j_i
				getattr(self,attrname)[i,j] = getattr(self,attrname[1:]+"Element")(bv,bu)
				if j!=i:
					getattr(self,attrname)[j,i] = getattr(self,attrname)[i,j]
		return getattr(self,attrname)

	# def _VmatConstruct(self):
	# 	print("Construct Vmat")
	# 	self._Vmat = np.empty((len(self.__gamma_basis),len(self.__gamma_basis)),'d')
	# 	for i,bv in enumerate(self.__gamma_basis):
	# 		for j_i,bu in tqdm(enumerate(self.__gamma_basis[i:]), desc='Row %d'%i, unit='mat.entry'):
	# 			j = i+j_i
	# 			self._Vmat[i,j] = self.VmatElement(bv,bu)
	# 			if j!= i:
	# 				self._Vmat[j,i] = self._Vmat[i,j].conj()
	# 	return self._Vmat

	# def _SOCmatConstruct(self):
	# 	self._SOCmat = np.empty((len(self.__gamma_basis),len(self.__gamma_basis)),'d')
	# 	for i,bv in enumerate(self.__gamma_basis):
	# 		for j_i,bu in tqdm(enumerate(self.__gamma_basis[i:]), desc='Row %d'%i, unit='mat.entry'):
	# 			j = i+j_i
	# 			self._SOCmat[i,j] = self.SOCmatElement(bv,bu)
	# 			if j!= i:
	# 				self._SOCmat[j,i] = self._SOCmat[i,j].conj()
	# 	return self._SOCmat

	# def _SmatConstruct(self):
	# 	self._Smat = np.identity(len(self.__gamma_basis))
	# 	for i, bi in enumerate(self.__gamma_basis):
	# 		for j_i_1, bj in enumerate(self.__gamma_basis[i+1:]):
	# 			j = i+j_i_1+1
	# 			self._Smat[i,j] = self.SmatElement(bi,bj)
	# 			self._Smat[j,i] = self._Smat[i,j]
	# 	return self._Smat

	def diagonalizeLS(self):
		if self._h0diag is None or len(self.__gamma_basis)!=self._h0diag.size:
			self._h0diagConstruct()
		for attrname in ["_Vmat","_SOCmat","_Smat"]:
			if getattr(self,attrname) is None or len(self.__gamma_basis)!=getattr(self,attrname).shape[0]:
				self._matConstruct(attrname)
		# try:
		H = self._Vmat + np.diag(self._h0diag)
		if not self.SOCfree:
			H += self._SOCmat
		# except:
		# 	print(self._Vmat is None)
		# 	exit(-2)
		print("Unperturbed energy = %.4f, 1st order perturbed = %.4f"%(self._h0diag[self.idx_in_basis], H[self.idx_in_basis,self.idx_in_basis]))
		print("Max amplitude of residue S due to radial functions = %.2g"%(abs(self._Smat-np.identity(len(self.__gamma_basis))).max()))
		w, v = eigh(H, b=self._Smat)
		col_idx_in_transmat = np.argmax(abs(v[self.idx_in_basis]))
		print("Col. Index in transition mat v is", col_idx_in_transmat)
		self._EnergyLS = w[col_idx_in_transmat]
		return w, v

	def SmatElement(self, bv, bu):
		if bv==bu:
			return 1
		normas = self.normalizationFactor(bv)*self.normalizationFactor(bu)
		# if normas==0:
		# 	return 0
		n1_, l1_, n2_, l2_ = bv
		n1,  l1,  n2,  l2  = bu
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
			res += (-1 if (self.S+self.L-l1-l2)%2==1 else 1)*\
				   uInnerProd_1d(v1, rv1, u2, ru2)*uInnerProd_1d(v2, rv2, u1, ru1)
		return res*2 / normas

	def VmatElement(self, bv, bu, ML=None, MS=None):
		"""<bv,L,ML,S,MS|\frac{1}{r_{12}}|bu,L,ML,S,MS>
		We can calculate only those ML=L, because L_z commute with 1/r_{12} and thus is still degenerate
		"""
		n1_, l1_, n2_, l2_ = bv
		n1,  l1,  n2,  l2  = bu
		normas = self.normalizationFactor(bv)*self.normalizationFactor(bu)
		v1,rv1 = request_ur(n1_,l1_)
		v2,rv2 = request_ur(n2_,l2_)
		if ML is None:
			ML = self.L
		if MS is None:
			MS = self.S
		if l1_==l2_ and l1!=l2:
			return np.conj(self.VmatElement(bu,bv,ML,MS))
		res_direct = 0
		anguladict = {}
		Mdict = {}
		for l in range(min(abs(l1-l1_),abs(l2-l2_)), max(l1_+l1,l2_+l2)+1):
			# print("l =",l)
			angula = angulaV(l, l1_,l2_,l1,l2,self.L,ML)
			if abs(angula) < 1e-15:
				# print('angula==0')
				anguladict[l] = 0
				continue
			else:
				anguladict[l] = angula
			M = gridMaxR_l(l,rv1,rv2)
			Mdict[l] = M
			radial = radiaV(M,(n1_,l1_),(n2_,l2_),(n1,l1),(n2,l2))
			res_direct += radial*angula
		res_exchang = 0
		if l1==l2:
			if n1==n2: # swapping makes no difference, reuse res_direct
				res_exchang = res_direct
			else: # reuse angular integration
				for l in range(min(abs(l2-l1_),abs(l1-l2_)), max(l1_+l2,l2_+l1)+1):
					if anguladict[l] == 0:
						continue
					M = Mdict[l]
					radial = radiaV(M,(n1_,l1_),(n2_,l2_),(n2,l2),(n1,l1))
					res_exchang += radial*anguladict[l]
			return (res_direct + (-1 if (self.S+self.L)%2==1 else 1)*res_exchang)*2 / normas
		del anguladict
		for l in range(min(abs(l2-l1_),abs(l1-l2_)), max(l1_+l2,l2_+l1)+1):
			angula = angulaV(l, l1_,l2_,l2,l1,self.L,ML)
			if abs(angula) < 1e-15:
				continue
			if l in Mdict.keys():
				M = Mdict[l]
			else:
				M = gridMaxR_l(l,rv1,rv2)
			radial = radiaV(M,(n1_,l1_),(n2_,l2_),(n2,l2),(n1,l1))
			res_exchang += radial*angula
		return (res_direct + (-1 if (self.S+self.L-l1-l2)%2==1 else 1)*res_exchang)*2 / normas

	def SOCmatElement(self, bv, bu, MJ=None):
		if self.S ==0:
			return 0
		n1_,l1_,n2_,l2_ = bv
		n1, l1, n2, l2  = bu
		normas = self.normalizationFactor(bv)*self.normalizationFactor(bu)
		if MJ is None:
			MJ = self.J
		if l1_==l2_ and l1!=l2:
			return np.conj(self.SOCmatElement(bu,bv,MJ))
		res_direct = 0
		anguladict = {}
		if l1_==l1 and l2_==l2:
			for k,lk in enumerate([l1,l2]):
				angula = angulaSO(l1,l2,self.L,self.S,self.J,MJ,k)
				if abs(angula) < 1e-15:
					anguladict[k] = 0
					continue
				else:
					anguladict[k] = angula
				radial = radiaSO((n1_,l1_),(n2_,l2_),(n1,l1),(n2,l2),k,lk)
				res_direct += radial*angula
		res_exchang = 0
		if l1_==l2 and l2_==l1:
			if l1==l2: 
				if n1==n2: # swapping makes no difference, reuse res_direct
					res_exchang = res_direct
				else: # reuse angula
					for k,lk in enumerate([l1,l2]):
						if anguladict[k]==0:
							continue
						radial = radiaSO((n1_,l1_),(n2_,l2_),(n2,l2),(n1,l1),k,lk)
						res_exchang += radial*anguladict[k]
				return (res_direct + (-1 if (self.S+self.L)%2==1 else 1)*res_exchang)*2 / normas
			del anguladict
			for k,lk in enumerate([l1,l2]):
				angula = angulaSO(l1,l2,self.L,self.S,self.J,MJ,k)
				if abs(angula) < 1e-15:
					continue
				radial = radiaSO((n1_,l1_),(n2_,l2_),(n2,l2),(n1,l1),k,lk)
				res_exchang += radial*angula
		return (res_direct + (-1 if (self.S+self.L-l1-l2)%2==1 else 1)*res_exchang)*2 / normas

	def save(self,fname, eigvals, eigvecs):
		dct = {"eigvals":eigvals,"eigvecs":eigvecs,"V":self._Vmat,"S":self._Smat,"basis":self.__gamma_basis}
		if not self.SOCfree:
			dct["SOC"] = self._SOCmat
		np.savez(fname,**dct)


def angulaV(l,l1_,l2_,l1,l2, L,ML):
	# if ML != ML_ or L!=L_:
	# 	return 0
	if l > max(l1_+l1,l2_+l2) or l < min(abs(l1-l1_),abs(l2-l2_)):
		return 0
	res = 0
	# print(range(max(-l1,ML-l2),min(l1,ML+l2)+1))
	for m1 in range(max(-l1,ML-l2),min(l1,ML+l2)+1):
		cf_ = clebsch_gordan(l1, l2, L, m1, ML-m1, ML)
		if cf_ == 0:
			continue
		res_ = 0
		# print("m1 =",m1,range(max(-l1_,ML_-l2_,m1-l),min(l1,ML_+l2_,m1+l)+1))
		for m1_ in range(max(-l1_,ML-l2_,m1-l),min(l1_,ML+l2_,m1+l)+1):
			m = m1_-m1
			res_ += N(clebsch_gordan(l1_,l2_,L, m1_,ML-m1_,ML)*gaunt(l1_,l,l1,-m1_,m,m1)*gaunt(l2_,l,l2,-(ML-m1_),-m,ML-m1))\
				 * (-1 if (ML+m)%2 == 1 else 1)
			# print("\t\t", res)
		res += res_*cf_
	return N(res)

def radiaV(Ml,nl1_,nl2_,nl1,nl2):
	# Since Hashing the dict is faster that integration
	# the radial wavefunctions are only requested when I need the radial integration
	v1,rv1 = request_ur(*nl1_)
	v2,rv2 = request_ur(*nl2_)
	u1,ru1 = request_ur(*nl1)
	u2,ru2 = request_ur(*nl2)
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

def angulaSO(l1,l2,L,S,J,MJ,k):
	if k==1:
		return angulaSO(l1,l2,L,S,J,MJ,0)
	elif k!=0:
		raise ValueError("k is not in {0,1}, but there are only 2 valence electrons.")
	if l1==0:
		return 0
	if S==0:
		return 0
	res = 0
	for MS_ in range(-S, S+1):
		for MS in range(-S,S+1):
			ML_ = MJ-MS_
			ML  = MJ-MS
			# print("ML_=%d, MS_=%d, ML=%d, MS=%d"%(ML_,MS_,ML,MS),end='\t')
			cf_ = clebsch_gordan(L,S,J,ML_,MS_,MJ)*clebsch_gordan(L,S,J,ML,MS,MJ)
			# print("coefficient =",cf_)
			if cf_==0:
				continue
			res_ = 0
			for ml2 in range(-l2,l2+1):
				res__ = 0
				cf__ = clebsch_gordan(l1,l2,L,ML_-ml2,ml2,ML_)*clebsch_gordan(l1,l2,L,ML-ml2,ml2,ML)
				# print("\tml2=%d,ml1_=%d,ml1=%d\tcoefficient ="%(ml2,ML_-ml2,ML-ml2),cf__)
				if cf__==0:
					continue
				for ms2 in [-0.5,0.5]:
					res___ = 0
					cf___ =clebsch_gordan(0.5,0.5,S,MS_-ms2,ms2,MS_)*clebsch_gordan(0.5,0.5,S,MS-ms2,ms2,MS)
					# print("\t\tms2=%.1f,ms1_=%.1f,ms1=%.1f\tcoefficient"%(ms2,MS_-ms2,MS-ms2),cf___)
					if cf___==0:
						continue
					if MS==MS_:
						res___ = (ML-ml2)*(MS-ms2)
					elif MS-1==MS_:
						res___ = sqrt((l1*(l1+1)-(ML-ml2)*(ML-ml2+1))*(0 if MS-ms2==-0.5 else 1))/2
					elif MS+1==MS_:
						res___ = sqrt((l1*(l1+1)-(ML-ml2)*(ML-ml2-1))*(0 if MS-ms2==0.5 else 1))/2
					# res___ = sum([ldots*clebsch_gordan(l1,0.5,j1,ML_-ml2,MS_-ms2,MJ-ml2-ms2)*clebsch_gordan(l1,0.5,j1,ML-ml2,MS-ms2,MJ-ml2-ms2) for j1,ldots in [(l1-0.5,-0.5*(l1+1)),(l1+0.5,0.5*l1)]])
					# print("\t\t\t",res___)
					res__ += res___*cf___
				# print("\tml2=%d,ml1_=%d,ml1=%d\tres__*cf__ ="%(ml2,ML_-ml2,ML-ml2), res__*cf__)
				res_ += res__*cf__
			# print("ML_=%d, MS_=%d, ML=%d, MS=%d"%(ML_,MS_,ML,MS),"res_ =", res_)
			res += N(res_*cf_)
	return res

def radiaSO(nl1_,nl2_,nl1,nl2, k,lk):
	# Since Hashing the dict is faster that integration
	# the radial wavefunctions are only requested when I need the radial integration
	v1,rv1 = request_ur(*nl1_)
	v2,rv2 = request_ur(*nl2_)
	u1,ru1 = request_ur(*nl1)
	u2,ru2 = request_ur(*nl2)
	if k==1:
		return radiaSO(v2,rv2,v1,rv1,u2,ru2,u1,ru1,0,lk)
	elif k!=0:
		raise ValueError("k is not in {0,1}, but there are only 2 valence electrons.")
	u1 *= socXi(ru1,paramss[lk])
	return uInnerProd_1d(v1,rv1,u1,ru1)*uInnerProd_1d(v2,rv2,u2,ru2)

if __name__ == '__main__':

	# print(angulaV(2,2,2,4,4,2,2,4,4))
	# u1, r1 = request_ur(4,1)
	# u2, r2 = request_ur(4,2)
	# M = np.ones((r1.size, r2.size))
	# print(uInnerProd_1d(u1,r1,u2,r2)**2)

	# print(radiaV(M, u1,r1, u2,r2,u2,r2,u1,r1))
	# M = gridMaxR_l(0,r1,r2)

	# trial = 2
	# ground = LS2eState(0,0,0,(4,0,4,0))
	# ground.defineBasis_LSrestricted((0,7),(3,7),(3,7))
	# ground.displayBasis()
	# w,v = ground.diagonalizeLS()
	# ground.save("4s4s_trial%d.npz"%trial,w,v)
	# print(ground._EnergyLS)
	# ax = plt.subplot(1,2,1)
	# V = ax.matshow(ground._Vmat)
	# plt.colorbar(V)
	# ax.set_title(r"$V$")

	# ax = plt.subplot(1,2,2)
	# v = ax.matshow(np.log10(abs(v)))
	# plt.colorbar(v)
	# ax.set_title(r"$\log_{10}(|$eigvecs$|)$")
	# plt.subplots_adjust(bottom=0,top=0.99,right=0.95,left=0.05)
	# plt.savefig('4s4s_trial%d.png'%trial)
	
	excite1 = LS2eState(1,0,1,(4,0,4,1))
	trial = 0
	excite1 = LS2eState(1,0,1,(4,0,4,1))
	excite1.defineBasis_LSrestricted((0,6),(3,7),(3,7))
	excite1.displayBasis()
	# t0 = time()
	w,v = excite1.diagonalizeLS()
	excite1.save("4s4p_trial%d.npz"%trial,w,v)
	print(excite1._EnergyLS)

	ax = plt.subplot(1,2,1)
	V = ax.matshow(excite1._Vmat)
	plt.colorbar(V)
	ax.set_title(r"$V$")
	ax = plt.subplot(1,2,2)
	v = ax.matshow(np.log10(abs(v)))
	plt.colorbar(v)
	ax.set_title(r"$\log_{10}(|$eigvecs$|)$")
	plt.subplots_adjust(bottom=0,top=0.99,right=0.95,left=0.05)
	plt.savefig('4s4p_trial%d.png'%trial)
	# print("Total duration", time()-t0)

	# trial = 5
	# highR1 = LS2eState(0,0,(4,0,18,0))
	# highR1.defineBasis_LSrestricted((0,1),(4,6),(11,25))
	# highR1.appendGammaBasis([(3,2,n2,2) for n2 in range(11,25)])
	# highR1.displayBasis()
	# w,v = highR1.diagonalizeLS()
	# print(highR1._EnergyLS)
	# np.savez('4s18s_trial%d.npz'%trial, eigvals=w, eigvecs=v)
	# plt.matshow((abs(v)))
	# plt.savefig("4s18s_trial%d.png"%trial)

	# excite1 = LS2eState((4,0,4,1),1,1,0,0)
	# excite2 = LS2eState((4,0,5,1),1,1,0,0)
	# print(V_mat_element_LScoupledStates(excite1,excite2))
	# excite1_ = LS2eState((4,0,4,1),1,0,0,0)
	# excite2_ = LS2eState((4,0,5,1),1,0,0,0)
	# print(V_mat_element_LScoupledStates(excite1_,excite2_))
	# print(angulaSO(1,0,1,0,2,2,1,0,True))
	# print(angulaSO(0,0,0,0,1,1,0,0,True))
	pass

