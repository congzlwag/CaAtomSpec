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

#define DEBUG_OUTPUT

static PyObject * NumerovIntegration(PyObject *self, PyObject *args);
static PyObject * NumerovEigensolve(PyObject *self, PyObject *args);
static PyObject * NumerovSOCXi(PyObject *self, PyObject *args);
// static PyObject * NumerovDEBUG(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
  // "Python name", C function,  arg_representation, doc
  {"integrate", (PyCFunction)NumerovIntegration, METH_VARARGS, 
  "numerov.integrate(l, j, energy, Uparams, rmin, rmax, dx, init1, init2):\n"
  "Given energy, inward integrate Numerov wavefunction\n"
  "Return 2 arrays: u_k & r_k"},
  {"eigensolve", (PyCFunction)NumerovEigensolve, METH_VARARGS, 
  "numerov.eigensolve(l, j, e_estimate, Uparams, dx, rmax):\n"
  "Solve the eigen function & eigen energy (near e_estimate) with Numerov difference form.\n"
  "Return: eigen energy, u_k, r_k"},
  {"socXi",(PyCFunction)NumerovSOCXi, METH_VARARGS, "numerov.socXi"},
  // {"debug",(PyCFunction)NumerovDEBUG, METH_VARARGS, "numerov.debug"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "numeroveig",
    "Numerov method for the single electron in a central potential by a full-shell nuclei",
     -1, module_methods, NULL, NULL, NULL, NULL, };

  PyMODINIT_FUNC PyInit_numeroveig(void) {
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
  PyMODINIT_FUNC initnumeroveig(void) {
    PyObject * m;
    m = Py_InitModule3("numeroveig", module_methods, "Numerov method for the single electron in a central potential by a full-shell nuclei");
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
double alphaD = 3.26, alpha2 = 5.325135447834466e-5;
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
	if(!(PyArg_ParseTuple(args,"iddOdd", &l, &j, &einit, &Uparams, &dx, &rmax)))
		return NULL;
	setUParams(Uparams);
	calcAuxParams(dx);

	Vec<double> y_(n_samp);
	Vec<double> r_(n_samp);
	doMesh(r_);

	eigval = eigen(einit, r_, y_);
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
	while ((br>0)&&(divergePoint == 0)){
        br--;
		y_[br] = (2*(1-5*gh2o12_[br+1])*y_[br+1]-(1+gh2o12_[br+2])*y_[br+2])/(1+gh2o12_[br]);
        if ((temp=fabs(y_[br]*pow(r_[br],0.25)))>maxValue){
            divergePoint = br;
            while ((fabs(y_[divergePoint+1])<fabs(y_[divergePoint])) && (divergePoint<checkPoint)){
                divergePoint +=1;
            }
            if (divergePoint>checkPoint){
                  PyErr_SetString(PyExc_RuntimeError, "divergePoint not found");
                  return;
            }
        }
    }
    #ifdef DEBUG_OUTPUT
    std::cout<<"n_samp="<<n_samp<<", checkPoint="<<checkPoint<<", divergePoint="<<divergePoint<<std::endl;
    #endif
    // RETURN RESULT - but set to zero divergent part (to prevent integration there)
	for (br=0; br<divergePoint; br++) y_[br] = 0;

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

// static PyObject * NumerovXi(PyObject * self, PyObject * args){
// 	double r;
// 	PyArrayObject * Uparams;
// 	if(!(PyArg_ParseTuple(args,"didO", &r, &l, &j, &Uparams)))
// 		return NULL;
// 	setUParams(Uparams);
// 	calcAuxParams();
// 	return Py_BuildValue("d", socXi(r));
// }
