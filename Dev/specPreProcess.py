from scipy.constants import h,c,e, physical_constants
import pandas as pd
import numpy as np

hc_in_eVcm = h*c/e*100
elim = {'CaI':-(49305.96 + 95751.87), 'CaII':-95751.87}

def cm_1_to_eV(energy, limit='CaII'):
	return hc_in_eVcm*(energy+elim[limit])

eV_in_au = e/physical_constants['atomic unit of energy'][0]

def appendColeV(limit, lmax):
	npz = {}
	for l in range(lmax+1):
		fp = open("specs/"+limit+'_L%d.dat'%l,'r')
		st = fp.read().split('\n');
		dat = []
		for s in st:
			dt = [eval(it) for it in s.split(' ')]
			# print dt[-1]
			dt.append(cm_1_to_eV(dt[-1], limit))
			dat.append(tuple(dt))
			# print(dat)
		dat = np.asarray(dat,dtype=[('n','i'),('j','f'),('spec/cm_1','d'),('energy/eV','d')])
		npz['%d'%l] = dat.copy()
		dat = pd.DataFrame(dat)
		dat.to_csv("specs/"+limit+'_L%d.csv'%l, index=False)
	np.savez("specs/"+limit+'.npz', **npz)

if __name__ == '__main__':
	appendColeV('CaII',4)