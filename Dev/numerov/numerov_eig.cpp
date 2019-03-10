#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>
#include <Vec.h>
#include <tridiag.h>
#include <cmath>
// #include <fstream>
// #include <string>
#include <stdio.h>
#include <cstdlib>
// #include <random>

#define INVPOW_MAXITER 50
#define INVPOW_DV_TOL 1e-10
#define EIGE_MAXITER 3
#define SIMANN_MAXITER 100
#define INTEGRATE_FROMLASTMAX_SUP 50

// #define DEBUG_OUTPUT

static PyObject * NumerovIntegration(PyObject *self, PyObject *args);
static PyObject * NumerovEigensolve(PyObject *self, PyObject *args);
static PyObject * NumerovSOCXi(PyObject *self, PyObject *args);
static PyObject * UInnerProd1d(PyObject *self, PyObject *args);
// static PyObject * NumerovDEBUG(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
  // "Python name", C function,  arg_representation, doc
  {"integrate", (PyCFunction)NumerovIntegration, METH_VARARGS, 
  "numerov.integrate(l, j, energy, Uparams, rmin, rmax, dx, init1, init2):\n"
  "Given energy, inward Numerov integrate radial wavefunction u\n"
  "Parameters:\n"
  "\tl: Orbit ang-momentum number\n"
  "\tj: Total ang-momentum number. If ignore SOC, set a negative value\n"
  "\tenergy: Exact value of enery level, directly used in integration\n"
  "\tUparams: numpy 1D array of length 4 or 5. Parameters in the model potential. If Uparams.size==4, a4=0 by default, else if Uparams.size==5, fill in (a1,a2,a3,a4,rc) correspondingly\n"
  "\trmin, rmax: Boundaries of the integration. Empirically rmax = n*2*(n+15) is enough\n"
  "\tdx: Step length of x = r**0.5"
  "\tinit1,init2: Two initial values for the integration that are assigned as\n"
  "\t\ty[-1] = init2; y_[-2] = init1;\n"
  "Return two 1D arrays: u_k & r_k"},
  {"eigensolve", (PyCFunction)NumerovEigensolve, METH_VARARGS, 
  "numerov.eigensolve(l, j, e_estimate, Uparams, dx, rmax, return_vec):\n"
  "Solve the eigen function & eigen energy (near e_estimate) with Numerov difference form.\n"
  "Parameters:\n"
  "\tl: Orbit ang-momentum number\n"
  "\tj: Total ang-momentum number. If ignore SOC, set a negative value\n"
  "\te_estimate: Estimated value of eigenenergy, which is the shift before inverse power iteration.\n"
  "\tUparams: numpy 1D array of length 4 or 5. Parameters in the model potential. If Uparams.size==4, a4=0 by default, else if Uparams.size==5, fill in (a1,a2,a3,a4,rc) correspondingly\n"
  "\tdx: Step length of x = r**0.5"
  "\trmax: Boundary of the integration. Empirically rmax = n*2*(n+15) is enough\n"
  "\treturn_vec: Bool value indicating whether to return the eigen vector.\n"
  "Return: eigen_energy, u_k, r_k if return_vec else eigen_energy"},
  {"socXi",(PyCFunction)NumerovSOCXi, METH_VARARGS, "numerov.socXi"},
  {"uInnerProd_1d",(PyCFunction)UInnerProd1d, METH_VARARGS, "numerov.uInnerProd"},
  // {"debug",(PyCFunction)NumerovDEBUG, METH_VARARGS, "numerov.debug"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "numerov",
    "Numerov method for the single electron in a central potential by a full-shell nuclei",
     -1, module_methods, NULL, NULL, NULL, NULL, };

  PyMODINIT_FUNC PyInit_numerov(void) {
    PyObject * m;
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;
    // import Numpy API
    import_array();
    //  creating numerovError
    // numerovError = PyErr_NewException("numerov.error", NULL, NULL);
    // Py_INCREF(numerovError);
    // PyModule_AddObject(m, "error", numerovError);
    return m;
  }
#else
  PyMODINIT_FUNC initnumerov(void) {
    PyObject * m;
    m = Py_InitModule3("numerov", module_methods, "Numerov method for the single electron in a central potential by a full-shell nuclei");
    if (m == NULL) return;
    //  import Numpy API
    import_array();
    //  creating numerovError
    // numerovError = PyErr_NewException("numerov.error", NULL, NULL);
    // Py_INCREF(numerovError);
    // PyModule_AddObject(m, "error", numerovError);
  }
#endif

// ================== Global Variables ==================
int Z = 20, CoreValency = 2;
double alphaD = 3.5; // 3.26 from Opik 1967
double alpha2 = 5.325135447834466e-5;
size_t l=0;
double j=0.5, mu=0.9999862724756698;
// to ignore SOC, set j<0
double SoL=0.5*(j*(j+1)-l*(l+1)-0.75), hfL2=0.5*l*(l+1);

double dx, rmax, xmin;
double dx2t2o3;
size_t n_samp=0;

// parameters in the modeled potential
double a1,a2,a3,a4,rc;

// Auxilaries
void setUParams(PyArrayObject * const Uparams);
size_t calcAuxParams(double _xmin);
void doMesh(Vec<double> & r_);
PyObject * castNPYArray(const Vec<double> & v_);

// Core function
double trapzNorm(const Vec<double> & u_, const Vec<double> & r_);
void integ(double energy, const Vec<double> & r_, Vec<double> & y_, double init1, double init2);
static PyObject * NumerovIntegration(PyObject * self, PyObject * args){
	double rmin, energy, init1, init2;
	PyArrayObject * Uparams;
	if(!(PyArg_ParseTuple(args,"iddOddddd", &l, &j, &energy, &Uparams, &rmin, &rmax, &dx, &init1, &init2)))
		return NULL;
	setUParams(Uparams);
	
	calcAuxParams(sqrt(rmin));
	Vec<double> r_(n_samp);
	doMesh(r_);
	Vec<double> y_(n_samp);
	// main function for integration
	integ(energy, r_, y_, init1, init2);
	// convert y_ that is at the moment R(r)*r^{3/4} into u(r) = R(r)*r
	for(size_t k=0; k<n_samp; k++) y_[k] = y_[k]*pow(r_[k],0.25);
	y_ /= sqrt(trapzNorm(y_, r_));
	PyObject * u_ = castNPYArray(y_);
	PyObject * ri_= castNPYArray(r_);
	y_.destruct();
	r_.destruct();
	return Py_BuildValue("OO",u_,ri_);
}

// Core functions
double invPower(const Triband<double> & T, Vec<double> & V, const Triband<double> & pM, double eval0, bool & success);
double eigen(double Einit, const Vec<double> & r_, Vec<double> & v);
static PyObject * NumerovEigensolve(PyObject * self, PyObject * args){
	double einit, eigval;
	PyArrayObject * Uparams;
	bool ret_vec;
	if(!(PyArg_ParseTuple(args,"iddOddp", &l, &j, &einit, &Uparams, &dx, &rmax, &ret_vec)))
		return NULL;
	setUParams(Uparams);
	calcAuxParams(dx);

	Vec<double> y_(n_samp);
	Vec<double> r_(n_samp);
	doMesh(r_);

	eigval = eigen(einit, r_, y_);
	if(!ret_vec)
		return Py_BuildValue("d", eigval);
	// convert y_ that is at the moment R(r)*r^{3/4} into u(r) = R(r)*r
	for(size_t k=0; k<n_samp; k++) y_[k] = y_[k]*pow(r_[k],0.25);
	y_ /= sqrt(trapzNorm(y_, r_));
	PyObject * u_ = castNPYArray(y_);
	PyObject * ri_ = castNPYArray(r_);
	r_.destruct();
	y_.destruct();
	return Py_BuildValue("(dOO)", eigval, u_, ri_);
}

void setUParams(PyArrayObject * const Uparams){
	double * uparams = (double *)(PyArray_GETPTR1(Uparams,0));
	if(PyArray_SIZE(Uparams)==5)
		a1 = uparams[0], a2 = uparams[1], a3 = uparams[2], a4 = uparams[3], rc = uparams[4];
	else
		a1 = uparams[0], a2 = uparams[1], a3 = uparams[2], a4 = 0, rc = uparams[3];
}

size_t calcAuxParams(double _xmin){
	SoL=0.5*(j*(j+1)-l*(l+1)-0.75), hfL2=0.5*l*(l+1);
	dx2t2o3 = dx*dx*2/3;
	xmin = _xmin;
	int nsam = (int)((sqrt(rmax)-xmin)/dx);
	if(nsam<=0)
		PyErr_SetString(PyExc_RuntimeError, "n_samp <= 0. Check rmax and dx");
	return n_samp = (size_t)nsam;
}

void doMesh(Vec<double> & r_){
	// r_.resiz(n_samp);
	double xt;
	for(size_t i=0; i<n_samp; i++){
		xt = xmin + i*dx;
		r_[i] = xt*xt;
	}
	return;
}

// Implementation
void initVecWOver8mur(const Vec<double> & r_, Vec<double> & xpot_, double * const ebound_=NULL);
void initVecF(const Vec<double> & r_, const Vec<double> & xpot_, Vec<double> & f_);
void initVecP(double energy, const Vec<double> & r_, const Vec<double> & xpot_, Vec<double> & p_);
void integ(double energy, const Vec<double> & r_, Vec<double> & y_, double init1, double init2){
	Vec<double> xpot_(n_samp);
	initVecWOver8mur(r_, xpot_);
	Vec<double> gh2o12_(n_samp);
	initVecP(energy, r_, xpot_, gh2o12_);
	size_t br = n_samp -1,checkPoint=0,fromLastMax=0,divergePoint;
	double maxValue=0, temp; // max value of u(r) = y(x)*sqrt(x)
	y_[br] = init2;
	y_[--br] = init1;
	while(br>checkPoint && br>0){
		br--;
		y_[br] = (2*(1-5*gh2o12_[br+1])*y_[br+1]-(1+gh2o12_[br+2])*y_[br+2])/(1+gh2o12_[br]);
		// maxValue traces the maximum amplitude of u
		if((temp=fabs(y_[br]*pow(r_[br],0.25)))>maxValue){
			maxValue = temp;
		}
		else{
			fromLastMax++;
			if(fromLastMax>INTEGRATE_FROMLASTMAX_SUP)
				checkPoint = br;
		}
	}
	divergePoint = 0;
	// u_1 = y_[br]*pow(r_[br],0.25), u_2=0;
	// if (br<n_samp-1){
	// 	u_2 = y_[br+1]*pow(r_[br+1],0.25);
	// }
	while (br>0 && divergePoint==0){
        br--;
		y_[br] = (2*(1-5*gh2o12_[br+1])*y_[br+1]-(1+gh2o12_[br+2])*y_[br+2])/(1+gh2o12_[br]);
		temp = y_[br]*pow(r_[br],0.25);
		// if ((temp-u_1)*(u_1-u_2)< 0 && u_1*temp>=0 && u_2*u_1>=0 && fabs(u_1)<=fabs(temp) && fabs(u_1)<=fabs(u_2)){
		// 	divergePoint = br;
		// }
		// else{
			// Supposedly, the highest maxima of u should correspond to the largest r
	        if (fabs(temp) > maxValue){
	            divergePoint = br;
	        }
	        // else{
	        // 	u_2 = u_1;
	        // 	u_1 = temp;
	        // }
		// }
    }
    while ((fabs(y_[divergePoint+1])<fabs(y_[divergePoint])) && (divergePoint<n_samp-1)){
        divergePoint +=1;
    }
    if(divergePoint < checkPoint){
    	
	    // if (divergePoint>checkPoint){
	    //       PyErr_SetString(PyExc_RuntimeError, "divergePoint not found");
	    //       return;
	    // }
	    // RETURN RESULT - but set to zero divergent part (to prevent integration there)
		for (br=0; br<divergePoint; br++) y_[br] = 0;
    }
    #ifdef DEBUG_OUTPUT
    std::cout<<"n_samp="<<n_samp<<", checkPoint="<<checkPoint<<", divergePoint="<<divergePoint<<std::endl;
    #endif

	gh2o12_.destruct();
	xpot_.destruct();
	return;
}

double socXi(double r){
	double effCharge, ddrEffCharge, aCovr5, xi, rovrc6;
	effCharge = CoreValency+(Z-CoreValency)*exp(-a1*r)-r*(a3+a4*r)*exp(-a2*r);
	ddrEffCharge = -a1*(Z-CoreValency)*exp(-a1*r) - ((a3+2*a4*r) - a2*r*(a3+a4*r))*exp(-a2*r);
	xi = effCharge/(r*r) -ddrEffCharge/r;
	aCovr5 = alphaD/(pow(r,5));
	rovrc6 = pow(r/rc,6);
	xi += aCovr5 * (exp(-rovrc6)*(3*rovrc6-2)+2);
	xi /= r;
	return 0.5*alpha2*xi;
}

static PyObject * NumerovSOCXi(PyObject * self, PyObject * args){
	PyArrayObject * Uparams;
	PyArrayObject * rs;
	if(!(PyArg_ParseTuple(args,"OO", &rs, &Uparams)))
		return NULL;
	setUParams(Uparams);
	PyObject * results = PyArray_NewLikeArray(rs, NPY_ANYORDER, NULL, 0);
	if(!results)
		return NULL;
	double * rs_ptr = (double *)PyArray_GETPTR1(rs,0);
	double * res_ptr = (double *)PyArray_GETPTR1((PyArrayObject *)results,0);
	for(long k=0; k<PyArray_SIZE(rs); k++)
		res_ptr[k] = socXi(rs_ptr[k]);
	return results;
}

double radEffPot(double r){
	/*
	Radial Effective Potential
	Veff^{(lj)}(r) = U(r) + \frac{l(l+1)}{2\mu r} + \frac{\alpha^2}{4}(j(j+1)-l(l+1)-s(s+1))\frac{d U(r)}{r dr}
	*/
	double effCharge,res;
	effCharge = CoreValency+(Z-CoreValency)*exp(-a1*r)-r*(a3+a4*r)*exp(-a2*r);
	// corePotU
	res = - effCharge / r - alphaD/(2*pow(r,4)) * (1-exp(-pow(r/rc,6)));
	// centrifugal
	res += hfL2 / (mu*r*r);
	if(j<0) // SOC ignored
		return res;
	return res + SoL * socXi(r);
}

double WOver8mur(double r){
	/*
	Return W(\sqrt{r})/(8\mu r) = U(r) + \frac{l(l+1)}{2\mu r} + \frac{3}{32\mu r^2} + \frac{\alpha^2}{4}(j(j+1)-l(l+1)-s(s+1))\frac{d U(r)}{r dr}
	*/
	return radEffPot(r)+3/(32*mu*r*r);
}

void initVecWOver8mur(const Vec<double> & r_, Vec<double> & xpot_, double * const ebound_){
	// xpot_.resiz(n_samp);
	size_t i=0;
	if(ebound_!=NULL){
		double elw, eup, et;
		elw = eup = xpot_[i] = WOver8mur(r_[i]);
		for(i=1;i<n_samp;i++){
			et = xpot_[i] = WOver8mur(r_[i]);
			if(et<elw) elw = et;
			if(et>eup) eup = et;
		}
		ebound_[0] = elw;
		ebound_[1] = eup;
	}
	else{
		for(i=0; i<n_samp; i++)
			xpot_[i] = WOver8mur(r_[i]);
	}
}

void initVecF(const Vec<double> & r_, const Vec<double> & xpot_, Vec<double> & f_){
	// f_.resiz(n_samp);
	for(size_t i=0; i<n_samp; i++)
		f_[i] = 1 - dx2t2o3*mu*r_[i]*xpot_[i];
}

void initVecP(double energy, const Vec<double> & r_, const Vec<double> & xpot_, Vec<double> & p_){
	for(size_t i=0; i<n_samp; i++)
		p_[i] = dx2t2o3*mu*r_[i]*(energy-xpot_[i]);
}

// static PyObject * NumerovDEBUG(PyObject *self, PyObject *args){
// 	double rmin, energy, init1, init2;
// 	PyArrayObject * Uparams;
// 	if(!(PyArg_ParseTuple(args,"iddOddddd", &l, &j, &energy, &Uparams, &rmin, &rmax, &dx, &init1, &init2)))
// 		return NULL;
// 	setUParams(Uparams);
	
// 	calcAuxParams(sqrt(rmin));
// 	Vec<double> r_(n_samp);
// 	doMesh(r_);

// 	Vec<double> xpot_(n_samp);
// 	for(size_t i=0; i<n_samp; i++){
// 		xpot_[i] = WOver8mur(r_[i]);
// 	}
// 	// initVecWOver8mur(r_, xpot_);
// 	// Vec<double> gh2o12_(n_samp);
// 	// initVecP(energy, r_, xpot_, gh2o12_);
// 	return Py_BuildValue("(OO)", castNPYArray(xpot_), castNPYArray(r_));
// }

double eigen(double Einit, const Vec<double> & r_, Vec<double> & v){
	/*
	with x=\sqrt{r}, y(x)=x^{3/2}R(x^2), solve
		\frac{d^2 y(x)}{dx^2} = -g(x) y(x)
			g(\sqrt{r}) = 8r\mu(E-Veff(r))-\frac{3}{4r}
		with Numerov algorithm
	*/
	Vec<double> xpot_(n_samp);
	initVecWOver8mur(r_,xpot_);
	Vec<double> f_(n_samp);
	initVecF(r_,xpot_,f_);
	xpot_.destruct();

	Triband<double> A(n_samp);
	Triband<double> M(n_samp);
	size_t k=0;
	double rk=r_[k], eval;
	M.d.setAll(10.);
	A.d[k] = -(12-10*f_[k])/rk; A.pu[k] = f_[k+1]/rk;
	M.pu[k]= r_[k+1]/rk;
	for(k++; k<n_samp-1; k++){
		rk = r_[k];
		A.pd[k-1] = f_[k-1]/rk; A.d[k] = -(12-10*f_[k])/rk; A.pu[k] = f_[k+1]/rk;
		M.pd[k-1] = r_[k-1]/rk; M.pu[k] = r_[k+1]/rk;
	}
	rk = r_[k];
	A.pd[k-1] = f_[k-1]/rk; A.d[k] = -(12-10*f_[k])/rk;
	M.pd[k-1] = r_[k-1]/rk;

	// std::cout<<"Inverse Power Method"<<std::endl;
    // std::cout<<A.d;
    bool success = false;
    k=0;
	eval = -mu*dx2t2o3*Einit;
	while((!success) && k<EIGE_MAXITER){
		// std::cout<<"Iter k="<<k<<std::endl;
		eval = invPower(A, v, M, eval, success);
		k++;
		// if(success) std::cout<<"success"<<std::endl;
	}
	if(!success) 
		PyErr_WarnEx(PyExc_RuntimeWarning, "Reached Maxiter for InversePower Method.",1);
	A.destruct();
	M.destruct();
	f_.destruct();
	return -eval/(mu*dx2t2o3);
}

template <typename T1>
T1 maxbar(const Vec<T1> & v){
    T1 res = 0;
    for(size_t i=0; i< v.siz(); i++){
        if(abs(v[i]) > abs(res)) res = v[i];
    }
    return res;
}

template <typename T1>
T1 maxabs(const Vec<T1> & v){
    T1 res = 0, tmp;
    for(size_t i=0; i< v.siz(); i++){
        if((tmp=abs(v[i])) > res) res = tmp;
    }
    return res;
}

double invPower(const Triband<double> & T, Vec<double> & V, const Triband<double> & TM, double eval0, bool & success){
    // The eigenvalue in the vicinity of eval0 will be returned
    // and the corresponding eigenvector will be stored in v
    double eigval=0;
    size_t n=T.dim(), niter=0, i;

    V.setAll(n, 1.);
    Vec<double> u(n), w(n);
    double dv_norm = 1., tmp;

    // Tc = T - e_est * M
    Triband<double> Tc(n);
    for(i=0; i<n-1; i++){
    	Tc.d[i] = T.d[i] - eval0*TM.d[i];
    	Tc.pd[i]= T.pd[i]- eval0*TM.pd[i];
    	Tc.pu[i]= T.pu[i]- eval0*TM.pu[i];
    }
	Tc.d[i] = T.d[i] - eval0*TM.d[i];

	// LU decompose Tc on site
    LUdcmp(Tc);
    // Now Tc is no longer T - e_est * M
    success = true;
    while(dv_norm>INVPOW_DV_TOL){
        u = V;
        MVprod(TM,V,w);
        // std::cout<<maxabs(w)<<std::endl;
        LUsolve(Tc,w);
        // std::cout<<maxabs(w)<<std::endl;
        V.fromCMult(eigval = 1./maxbar(w), w);
        // std::cout<<V.norm()<<std::endl;
        // std::cout<<"deigval="<<eigval<<std::endl;
        dv_norm = 0.;
        for(i=0; i<n; i++)
            if( (tmp=abs(V[i]-u[i]))>dv_norm ) dv_norm=tmp;
        if(niter++>=INVPOW_MAXITER){
            // std::cout<<"InvPow Warning: Reached Maxiter "<<INVPOW_MAXITER<<", while "<<INVPOW_DV_TOL<<" cannot be satisfied. ";
            // std::cout<<"Finally |dv| = "<<dv_norm<<std::endl;
            success = false;
            break;
        }
    }
    V /= V.norm();
//    cout<<"The while loop finished"<<endl;
    Tc.destruct();
    u.destruct();
    w.destruct();
    return eigval+eval0;
}

double trapzNorm(const Vec<double> & u_, const Vec<double> & r_){
	double uj_1_square=0, ret=0, r_temp=0, uj_square;
	size_t j;
	for(j=0; j<n_samp; j++){
		uj_square = u_[j]*u_[j];
		ret += (r_[j]-r_temp)*(uj_square+uj_1_square)*0.5;
		r_temp = r_[j];
		uj_1_square = uj_square;
	}
	return ret;
}

PyObject * castNPYArray(const Vec<double> & v_){
    npy_intp m[1] = {v_.siz()};
    PyObject * out = PyArray_SimpleNew(1,m,NPY_DOUBLE);
    double * ptr = (double *)(PyArray_GETPTR1((PyArrayObject *)out,0));
    for(long k=0; k<m[0]; k++)
        ptr[k] = v_[k];
    return out;
}

static PyObject * UInnerProd1d(PyObject * self, PyObject * args){
	PyArrayObject * pyv;//, * pyav, * pyu, * pyau;
	PyArrayObject * pyav;
	PyArrayObject * pyu;
	PyArrayObject * pyau;
	if(!(PyArg_ParseTuple(args, "OOOO", &pyv, &pyav, &pyu, &pyau)))
		return NULL;
	if(PyArray_SIZE(pyv)!=PyArray_SIZE(pyav))
		PyErr_SetString(PyExc_RuntimeError, "v.size!=av.size");
	if(PyArray_SIZE(pyu)!=PyArray_SIZE(pyau))
		PyErr_SetString(PyExc_RuntimeError, "u.size!=au.size");

	double * v = (double *)(PyArray_GETPTR1(pyv,0));
	double * u = (double *)(PyArray_GETPTR1(pyu,0));
	double * av = (double *)(PyArray_GETPTR1(pyav,0));
	double * au = (double *)(PyArray_GETPTR1(pyau,0));
	long kv = 0, ku = 0;
	double res, last_prod, current_r, av_last, v_last, au_last, u_last, prod, r_incr;
	res= last_prod= current_r= av_last= v_last= au_last= u_last = 0;
	while(kv<PyArray_SIZE(pyv) && ku<PyArray_SIZE(pyu)){
		if(av[kv] <= au[ku]){
			r_incr = av[kv]-current_r;
			current_r = av[kv];
			if(r_incr>0){
				prod = v[kv]* (u_last + (u[ku]-u_last)/(au[ku]-au_last)*(current_r-au_last));
				res += r_incr*0.5*(prod + last_prod);
	        }
            av_last = av[kv];
            v_last = v[kv++];
		}
		else{
			r_incr = au[ku]-current_r;
			current_r = au[ku];
			if(r_incr>0){
				prod = u[ku]* (v_last + (v[kv]-v_last)/(av[kv]-av_last)*(current_r-av_last));
	            res += r_incr*0.5*(prod + last_prod);
			}
            au_last = au[ku];
            u_last = u[ku++];
		}
        last_prod = prod;
	}
	return Py_BuildValue("d",res);
}

// static PyObject * NumerovXi(PyObject * self, PyObject * args){
// 	double r;
// 	PyArrayObject * Uparams;
// 	if(!(PyArg_ParseTuple(args,"didO", &r, &l, &j, &Uparams)))
// 		return NULL;
// 	setUParams(Uparams);
// 	calcAuxParams();
// 	return Py_BuildValue("d", socXi(r));
// }
