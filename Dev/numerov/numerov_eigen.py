# -*- coding: utf-8 -*-

from numpy import *
import scipy.sparse as sps
from scipy.constants import physical_constants, h,c,e
from scipy.sparse.linalg import inv as spsinv

hc_in_eVcm = h*c/e*100
elim = {'CaI':49305.96, 'CaII':95751.87}

alpha2 = physical_constants['fine-structure constant'][0]**2

def cm_1_to_eV(e, limit):
	return hc_in_eVcm*(e-elim[limit])

class Numerov:
	"""
	in (Hatree) Atomic units
	"""
	Z = 20
	_CoreValency = 2
	alphaD   = 3.26
	_mass  = 39.96259098 * physical_constants['atomic mass constant'][0]/ physical_constants['atomic unit of mass'][0]
	_mu = (_mass - _CoreValency) / (_mass - _CoreValency +1)
	dx = 1e-3

	def __init__(self, n,l,j=None):
		self.n = n
		self.l = l
		self.j = j
		if j is not None:
			assert abs(abs(l-j)-0.5)<1e-5
		# else the instance is j independent

	def setCoreValency(self, CV):
		self._CoreValency = CV
		self._mu = (self._mass - CV) / (self._mass - CV +1)

	def setMass(self, mass):
		self._mass = mass
		self._mu = (mass - self._CoreValency) / (mass - self._CoreValency +1)
	
	def radEffPot(self, r, params):
		"""
		Radial Effective Potential
		Veff^{(lj)}(r) = U(r) + \frac{l(l+1)}{2\mu r} + \frac{\alpha^2}{4}(j(j+1)-l(l+1)-s(s+1))\frac{d U(r)}{r dr}
		"""
		a1,a2,a3,a4,rc = params
		effCharge = self._CoreValency+(self.Z-self._CoreValency)*exp(-a1*r)-r*(a3+a4*r)*exp(-a2*r)
		# corePotU
		res = effCharge / r - self.alphaD/(2*(r**4)) * (1-exp((r/rc)**6))
		# centrifugal
		res += 0.5*(self.l)*(self.l+1) / (self._mu*r*r)
		if self.j is None:
			return res
		# SOC term
		ddrEffCharge = -a1*(self.Z-self._CoreValency)*exp(-a1*r) - ((a3+2*a4*r) - a2*r*(a3+a4*r))*exp(-a2*r)
		xi = effCharge/(r*r) -ddrEffCharge/r
		aCovr5 = self.alphaD/(r**5)
		rovrc6 = (r/rc)**6
		xi += aCovr5 * (exp(-rovrc6)*(3*rovrc6-2)+2)
		xi /= r
		return res + alpha2 / 4 * (self.j*(self.j)-self.l*(self.l+1)-0.75) * xi

	def eigen(self, params, Einit=None):
		"""
		with x=\sqrt{r}, y(x)=x^{3/2}R(x^2), solve
		\frac{d^2 y(x)}{dx^2} = -g(x) y(x)
			g(\sqrt{r}) = 8r\mu(E-Veff(r))-\frac{3}{4r}
		with Numerov algorithm
		"""

		# set mesh
		rmax = 2*self.n*(self.n+15)
		rmin = self.dx**2
		x_ = arange(sqrt(rmin), sqrt(rmax), self.dx)
		n_samp = x_.size
		r_ = x_**2
		
		# compute matrices
		dx212 = self.dx**2/12.0
		f_ = 1 - (self.radEffPot(r_, params)*8*self._mu*(r_)+0.75/r_)*dx212
		dat= asarray([(-10*f_+12)/r_, append(f_[1:],0)/r_, f_/append(r_[1:],0)])
		A = sps.dia_matrix((dat, [0,1,-1]), shape=(n_samp, n_samp))
		dat= asarray([10*ones(n_samp), append(r_[1:],0)/r_, r_/append(r_[1:],0)])
		M = sps.dia_matrix((dat, [0,1,-1]), shape=(n_samp, n_samp))
		# Miv = spsinv(M)

		if Einit is None:
			# eigen solve the lowest 20
			energies = sps.linalg.eigs(A,20,M)
		else:
			energies = sps.linalg.eigs(A,1,M, Einit)
		return energies
		# return M

if __name__ == '__main__':
	params_ = array([[ 4.0099,  2.1315, 13.023 ,  0.    ,  1.6352],\
       				 [ 4.2056,  2.0186, 12.658 ,  0.    ,  1.5177],\
       				 [ 3.5058,  2.2648, 12.399 ,  0.    ,  1.6187],\
       				 [ 3.7741,  3.1848, 13.232 ,  0.    ,  0.715 ]])
	neuman = Numerov(8,0,0.5)
	# M = neuman.eigen()
	# M = neuman.eigen(params_[0])