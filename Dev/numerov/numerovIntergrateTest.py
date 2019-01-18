from numerov import integrate
from numpy import *
from scipy.constants import h,c,e, physical_constants
from matplotlib import pyplot as plt

cm_1_in_au = 100*h*c/physical_constants['atomic unit of energy'][0]
iLimit = 95751.87

n = 10
l = 2
j = 1.5
Z = 20
CV= 2

a1 = array([4.0099, 4.2056, 3.5058, 3.7741])
a2 = array([13.023, 12.658, 12.399, 13.232])
a3 = array([2.1315, 2.0186, 2.2648, 3.1848])
rc = array([1.6352, 1.5177, 1.6187, 0.7150])

alphaD = 3.26
mass = 39.96259098 * physical_constants['atomic mass constant'][0]/ physical_constants['atomic unit of mass'][0]
mu = (mass - CV)/(mass-CV+1)

innerLmt = alphaD**(1./3)
outerLmt = 2*n*(n+15)
step = 1e-3
init1=init2=0.001

stateE = (90753.92-iLimit)*cm_1_in_au


def radialufunc(n,l,j, init1, init2):
	assert abs(abs(l-j)-0.5)<0.01
	d = integrate(innerLmt, outerLmt, step, init1,init2, l, 0.5, j,\
			 stateE, alphaD, Z, CV, a1[l], a2[l], a3[l], 0, rc[l], mu)
	nmlizer = trapz(d[0]**2, x=d[1]) **0.5
	return d[0]/nmlizer, d[1]