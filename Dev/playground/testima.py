# -*- coding: utf-8 -*-
from sys import path
path.append("./build/lib.linux-x86_64-3.6/")
from testa import test
import numpy as np
if __name__ == '__main__':
	x=np.arange(0,3.14,0.1)
	print(x.size,flush=True)
	res = test(x,1.5)
	print(res[0],res[1])
