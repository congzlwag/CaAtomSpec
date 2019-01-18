
#include <Python.h>
// http://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#define ABS(a)     (((a) < 0) ? -(a) : (a))
#define MIN(a,b)   (((a) < (b)) ? (a) : (b))
#define MAX(a,b)   (((a) > (b)) ? (a) : (b))

int l;
double s,j;
double stateEnergy;
double alphaD;
double alpha2 = 5.325135447834466e-5; // square of fine structure constant
int Z, CoreValency;
double a1,a2,a3,a4,rc; // depends on l - determined by Python in advance
double mu;

#define DEBUG_OUTPUT

static PyObject * numerovError;

static PyObject * NumerovIntegration(PyObject *self, PyObject *args);
static PyObject * NumerovEigensolve(PyObject *self, PyObject *args);
static PyObject * Test(PyObject * self);

static PyMethodDef module_methods[] = {
  // "Python name", C function,  arg_representation, doc
  {"integrate", NumerovIntegration, METH_VARARGS, 
  "numerov.integrate(innerLimit, outerLimit, step, init1, init2, "
  "l, s, j, stateEnergy, alphaD, Z, CoreValency, a1, a2, a3, a4, rc, mu):\n"
  "Given energy, inward integrate Numerov wavefunction"
  ""},
  // {"eigensolve", NumerovEigensolve, METH_VARARGS, "Solve the eigen function & eigen energy with Numerov algorithm"},
  // {"test", (PyCFunction)Test, METH_NOARGS, "Playground"},
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
    if (m == NULL) return;
    // import Numpy API
    import_array();
    //  creating numerovError
    numerovError = PyErr_NewException("numerov.error", NULL, NULL);
    Py_INCREF(numerovError);
    PyModule_AddObject(m, "error", numerovError);

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
    numerovError = PyErr_NewException("numerov.error", NULL, NULL);
    Py_INCREF(numerovError);
    PyModule_AddObject(m, "error", numerovError);
  }
#endif

// =========== Playground ===================

static PyObject * Test(PyObject * self){
  std::cout<<"Hello World!";
  return Py_None;
}


// =========== variable definition ===========
int divergencePoint, checkPoint, fromLastMax;
int totalLength;
double* sol;
double* r;
double xmin, dx, ddx12, rmin, rmax;

// double (*RadEffPot)(double);

npy_intp dims[2];
PyObject* narray;

double commonTerm1=0;
double commonTerm2=0;
// would be
  // commonTerm1 = (j*(j+1.0)-(l)*(l+1.0)-s*(s+1.))/2.0;
  // commonTerm2 = 0.5*(l+.25)*(l+.75);
// after j,l being passed in

// =========== Numerov integration implementation ===========

__inline double EffectiveCharge(double r){
  // returns effective charge of the core
  return CoreValency+(Z-CoreValency)*exp(-a1*r)-r*(a3+a4*r)*exp(-a2*r);
}

__inline double DDrEffectiveCharge(double r){
  // returns \frac{d Z(r)}{dr}
  return -a1*(Z-CoreValency)*exp(-a1*r) - ((a3+2*a4*r) - a2*r*(a3+a4*r))*exp(-a2*r);
}

__inline double CorePotentialU(double r){
    // l dependent core potential (angular momentum of e) at radius r
    // returns Core potential
    return -EffectiveCharge(r)/r-alphaD/(2.*pow(r,4))*(1.-exp(-pow(r/rc,6)));
}

__inline double CorePotentialXi(double r){
  // returns \frac{1}{r}\frac{dU}{dr}
  double ddr = EffectiveCharge(r)/(r*r)-DDrEffectiveCharge(r)/r;
  double aCovr5 = alphaD/(pow(r,5)), rovrc6 = pow(r/rc,6);
  ddr += 2*aCovr5 + exp(-rovrc6)*aCovr5*(3*rovrc6-2);
  return ddr/r;
}

double RadialEffPotential(double r){
  // l<4
    return CorePotentialU(r)+alpha2/2.0*commonTerm1*CorePotentialXi(r) + commonTerm2/(mu*r*r);
}

double RadialEffPotential2(double r){
  // l>=4
    // act as if it is a Hydrogen atom, include spin-orbit coupling
    return -CoreValency/r  + alpha2/2.0*commonTerm1*CorePotentialXi(r) + commonTerm2/(mu*r*r);
}

double gFun(double r){ // the d2y/dx2 = g(r) y(x)
  // with potential for l<4
  return 8*r*mu*(stateEnergy-RadialEffPotential(r));
}

double gFun2(double r){ // the d2y/dx2 = g(r) y(x)
  // with potential for l>=4
  return 8*r*mu*(stateEnergy-RadialEffPotential2(r));
}

// void doMesh(double* x){
//   for(int j=0; j<totalLength; j++){
//     x[j] = xmin+j*dx;
//     r[j] = x[j]*x[j]
//   }
//   return;
// }

// double initPot(double* vpot){
//   // calculate the gridded (radial effective) potential, return the lower bound
//   int j=totalLength-1;
//   double elw = vpot[j] = (*Pot)(r[j]);
//   for(j--; j>=0; j--){
//     vpot[j] = (*Pot)(r[j]);
//     elw = MIN(elw, vpot[j]);
//   }
//   return elw;
// }

// int initVecF(double E, double* r, int totalLength, double* vpot, double* vecf){
//   // calculate the f vector for the numerov integration, return the 
//   int icl = -1;
//   double g;
//   for(j=0; j<totalLength; j++){
//     g = 8*r[j]*mu*(E-vpot[j]);
//     f[j] = 
//     if()
//   }
//   return icl;
// }

static PyObject * NumerovIntegration(PyObject *self, PyObject *args) {
  // Numerov arguments: innerLimit,outerLimit,gFun,step,init1,init2
  double innerLimit,outerLimit,step,init1,init2;
  double x,step2,maxValue,r;

    if (!(PyArg_ParseTuple(args, "dddddiddddiidddddd", 
      &innerLimit, &outerLimit, &step, &init1, &init2,
      &l, &s, &j, &stateEnergy, &alphaD,
      &Z, &CoreValency, &a1, &a2, &a3, &a4, &rc, &mu))) return NULL;


#ifdef DEBUG_OUTPUT
    printf("innerLimit\t=\t%.3f\nouterLimit\t=\t%.3f\nstep\t=\t%.3f\ninit1\t=\t%.3f\ninit2\t=\t%.3f\n",innerLimit,outerLimit,step,init1,init2);
    printf("l\t=\t%i\ns\t=\t%.1f\nj\t=\t%.1f\n",l,s,j);
    printf("stateEnergy\t=\t%.7f\nalphaD\t=\t%.3f\nalpha\t=\t%.3f\nZ\t=\t%i\n",stateEnergy,alphaD,sqrt(alpha2),Z);
    printf("a1\t\t%.4f\na2\t\t%.4f\na3\t\t%.4f\na4\t\t%.4f\nrc\t\t%.4f\n",a1,a2,a3,a4,rc);
    printf("mu\t\t%.4f",mu);
#endif

  // let's speed up calculation by calculating some common terms beforehand
  commonTerm1 = (j*(j+1.0)-((double)l)*(l+1.0)-s*(s+1.))/2.0;
  commonTerm2 = 0.5*(l+0.25)*(l+0.75);

  totalLength =  (int)((sqrt(outerLimit)-sqrt(innerLimit))/step);
  step2 = step*step;

#ifdef DEBUG_OUTPUT
  printf("Index = %i\n",totalLength);
  printf("Index should be about = %.2f\n",(sqrt(outerLimit)-sqrt(innerLimit)/step));
#endif

  br = totalLength;
  sol = (double*) malloc(2*br*sizeof(double));
  if (!sol){
// #ifdef DEBUG_OUTPUT
//     printf("Memory allocaiton failed! Aborting.");
// #endif
    PyErr_SetString(numerovError, "Memory allocaiton failed! Aborting");
    return NULL;
  }

  for(br=0; br<totalLength; br++) // x grid
    sol[br+totalLength] = sqrt(innerLimit)+br*step;

// for l<4
  if (l<4){

      br = totalLength-1;
      x = sol[br+totalLength];
      sol[br] = (2*(1-5.0/12.0*step2*gFun(x))*init1-(1+1/12.0*step2*gFun(x+step))*init2)/(1+1/12.0*step2*gFun(x-step));
      
      br = br-1;
      x = sol[br+totalLength];
      sol[br] = (2*(1-5.0/12.0*step2*gFun(x))*sol[br+1]-(1+1/12.0*step2*gFun(x+step))*init1)/(1+1/12.0*step2*gFun(x-step));

      maxValue = 0; // max value of u(r) = y(x)*sqrt(x)
      checkPoint = 0;
      fromLastMax = 0;

      while (br>checkPoint){
          br = br-1;
          x = sol[br+totalLength];
          sol[br] = (2*(1-5.0/12.0*step2*gFun(x))*sol[br+1]-(1+1/12.0*step2*gFun(x+step))*sol[br+2])/(1+1/12.0*step2*gFun(x-step));
          if (fabs(sol[br]*sqrt(x))>maxValue){
              maxValue = fabs(sol[br]*sqrt(x));
          }
          else{
              fromLastMax += 1;
              if (fromLastMax>50){
                  checkPoint = br;
              }
          }
      }
#ifdef DEBUG_OUTPUT
      printf("checkPoint = %d\n",checkPoint);
#endif
      divergencePoint = 0;
      while ((br>0)&&(divergencePoint == 0)){
        br = br-1;
        x = sol[br+totalLength];
        sol[br] = (2*(1-5.0/12.0*step2*gFun(x))*sol[br+1]-(1+1/12.0*step2*gFun(x+step))*sol[br+2])/(1+1/12.0*step2*gFun(x-step));

        if (fabs(sol[br]*sqrt(x))>maxValue){
            divergencePoint = br;
            while ((fabs(sol[divergencePoint])>fabs(sol[divergencePoint+1])) && (divergencePoint<checkPoint)){
                divergencePoint +=1;
            }
            if (divergencePoint>checkPoint){
                  PyErr_SetString(numerovError, "");
                  return NULL;
            }
        }
      }
#ifdef DEBUG_OUTPUT
      printf("divergencePoint = %d\n",divergencePoint);
#endif
  } // end of if l<4
  else{ //l>=4

      br = totalLength-1;
      x = sol[br+totalLength];
      sol[br] = (2*(1-5.0/12.0*step2*gFun2(x))*init1-(1+1/12.0*step2*gFun2(x+step))*init2)/(1+1/12.0*step2*gFun2(x-step));

      br = br-1;
      x = sol[br+totalLength];
      sol[br] = (2*(1-5.0/12.0*step2*gFun2(x))*sol[br+1]-(1+1/12.0*step2*gFun2(x+step))*init1)/(1+1/12.0*step2*gFun2(x-step));

      maxValue = 0;
      checkPoint = 0;
      fromLastMax = 0;

      while (br>checkPoint){
          br = br-1;
          x = sol[br+totalLength];
          sol[br] = (2*(1-5.0/12.0*step2*gFun2(x))*sol[br+1]-(1+1/12.0*step2*gFun2(x+step))*sol[br+2])/(1+1/12.0*step2*gFun2(x-step));

          if (fabs(sol[br]*sqrt(x))>maxValue){
              maxValue = fabs(sol[br]*sqrt(x));
          }
          else{
               fromLastMax += 1;
               if (fromLastMax>50){
                   checkPoint = br;
               }
          }
      }

      divergencePoint = 0;
      while ((br>0)&&(divergencePoint == 0)){
          br = br-1;
          x = sol[br+totalLength];
          sol[br] = (2*(1-5.0/12.0*step2*gFun2(x))*sol[br+1]-(1+1/12.0*step2*gFun2(x+step))*sol[br+2])/(1+1/12.0*step2*gFun2(x-step));

          if (fabs(sol[br]*sqrt(x))>maxValue){
              divergencePoint = br;
              while ((fabs(sol[divergencePoint])>fabs(sol[divergencePoint+1])) && (divergencePoint<checkPoint)){
                  divergencePoint +=1;
              }
              if (divergencePoint>checkPoint){
#ifdef DEBUG_OUTPUT
                  printf("ERROR: Numerov error\n");
#endif
                  PyErr_SetString(numerovError, "");
                  return NULL;
              }
          }
      }

  }

  // RETURN RESULT - but set to zero divergent part (to prevent integration there)
  for (i =0; i<divergencePoint; i++) sol[i] = 0;

  // convert sol that is at the moment R(r)*r^{3/4} into R(r)*r
  for (i=0; i<totalLength; i++)  sol[i]=sol[i]*sqrt(sol[i+totalLength]);
  // convert coordinates from sqrt(r) into r
  for (i=totalLength; i<2*totalLength; i++)  sol[i]=sol[i]*sol[i];

  // return the array as a numpy array (numpy will free it later)
  dims[0] = 2;
  dims[1] = totalLength;
  narray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, sol);
  //free(sol); # freeing of solution array should be done from Numpy
  // this is the critical line - tell numpy it has to free the data
  PyArray_ENABLEFLAGS((PyArrayObject*)narray, NPY_ARRAY_OWNDATA);
  return narray;
  // return 0;
}


// static PyObject * NumerovEigensolve(PyObject * self, PyObject *args){
//   if (!(PyArg_ParseTuple(args, "dddiddddiidddddd", 
//       &innerLimit, &outerLimit, &step,
//       &l, &s, &j, &stateEnergy, &alphaD,
//       &Z, &CoreValency, &a1, &a2, &a3, &a4, &rc, &mu))) return NULL;
  
//   /* Local variables */
//     static double eps=1e-10;
//     static int n_iter=100;
//     int i, j;
//     double e, de, fac;
//     int icl, kkk;
//     double x2l2, elw, eup, ddx12, norm;
//     int nodes;
//     double sqlhf, ycusp, dfcusp;
//     int ncross;
//     double *f;

// /* --------------------------------------------------------------------- */

// /* solve the schroedinger equation in radial coordinates on a 
//    logarithmic grid by Numerov method - atomic (Ry) units */

//     ddx12 = step * step / 12.;
// /* Computing 2nd power */
//     sqlhf = (l + 0.5) * (l + 0.5);
//     x2l2 = (double) (2*l+ 2);
// /* set initial lower and upper bounds to the eigenvalue */

//     eup = vpot[mesh];
//     elw = eup;
//     for (i = 0; i <= mesh; ++i) {
// /*      if ( elw > sqlhf / r2[i] + vpot[i] ) 
//   elw = sqlhf / r2[i] + vpot[i] ; */
//         elw = MIN ( elw, sqlhf / r2[i] + vpot[i] );
//     }
//     if (eup - elw < eps) {
//       fprintf (stderr, "%25.16e 25.16e\n", eup, elw);
//       fprintf (stderr, "solve_sheq: lower and upper bounds are equal\n");
//       exit(1);
//     }
//     e = (elw + eup) * .5;
//     f = malloc( (mesh+1) * sizeof(double) );

// /* start loop on energy */

//     de = 1e+10; /* any number larger than eps */
//     for ( kkk = 0; kkk < n_iter && ABS(de) > eps ; ++kkk ) {

// /* set up the f-function and determine the position of its last */
// /* change of sign */
// /* f < 0 (approximately) means classically allowed   region */
// /* f > 0         "         "        "      forbidden   " */

//       icl = -1;
//       f[0] = ddx12 * (sqlhf + r2[0] * (vpot[0] - e));
//       for (i = 1; i <= mesh; ++i) {
//         f[i] = ddx12 * (sqlhf + r2[i] * (vpot[i] - e));
// /* beware: if f(i) is exactly zero the change of sign is not observed */
// /* the following line is a trick to prevent missing a change of sign */
// /* in this unlikely but not impossible case: */
//         if (f[i] == 0.) {
//             f[i] = 1e-20;
//         }
//         if (f[i] != copysign(f[i], f[i - 1])) {
//             icl = i;
//         }
//       }
//       if (icl < 0 || icl >= mesh - 2) {
//         fprintf (stderr, "%4d %4d\n", icl, mesh);
//         fprintf (stderr, "error in solve_sheq: last change of sign too far");
//         exit(1);
//       }

// /* f function as required by numerov method */

//       for (i = 0; i <= mesh; ++i) {
//         f[i] = 1. - f[i];
//         y[i] = 0.;
//       }

// /* determination of the wave-function in the first two points */

//       nodes = n - l - 1;
//       y[0] = pow (r[0], l+1) * (1. - zeta * 2. * r[0] / x2l2) / sqr[0];
//       y[1] = pow (r[1], l+1) * (1. - zeta * 2. * r[1] / x2l2) / sqr[1];

//  outward integration, count number of crossings 

//       ncross = 0;
//       for (i = 1; i <= icl-1; ++i) {
//         y[i + 1] = ((12. - f[i] * 10.) * y[i] - f[i - 1] * y[i - 1])
//            / f[i + 1];
//         if (y[i] != copysign(y[i],y[i+1]) ) {
//             ++ncross;
//         }
//       }
//       fac = y[icl];

// /* check number of crossings */

//       if (ncross != nodes) {
//         /* incorrect number of nodes: adjust energy bounds */
//         if (ncross > nodes) {
//             eup = e;
//         } else {
//             elw = e;
//         }
//         e = (eup + elw) * .5;
//       }
//       else {
//         /* correct number of nodes: perform inward iteration */

// /* determination of the wave-function in the last two points */
// /* assuming y(mesh+1) = 0 and y(mesh) = dx */

//         y[mesh] = dx;
//         y[mesh - 1] = (12. - f[mesh] * 10.) * y[mesh] / f[mesh - 1];

// /* inward integration */

//         for (i = mesh - 1; i >= icl+1; --i) {
//           y[i - 1] = ((12. - f[i] * 10.) * y[i] - f[i + 1] * y[i + 1])
//            / f[i - 1];
//           if (y[i - 1] > 1e10) {
//             for (j = mesh; j >= i-1; --j) y[j] /= y[i - 1];
//           }
//         }

// /* rescale function to match at the classical turning point (icl) */

//         fac /= y[icl];
//         for (i = icl; i <= mesh; ++i) {
//           y[i] *= fac;
//         }

// /* normalize on the segment */

//         norm = 0.;
//         for (i = 1; i <= mesh; ++i) {
//           norm += y[i] * y[i] * r2[i] * dx;
//         }
//         norm = sqrt(norm);
//         for (i = 0; i <= mesh; ++i) {
//           y[i] /= norm;
//         }

// /* find the value of the cusp at the matching point (icl) */

//         i = icl;
//         ycusp = (y[i - 1] * f[i - 1] + f[i + 1] * y[i + 1] + f[i] * 10. 
//         * y[i]) / 12.;
//         dfcusp = f[i] * (y[i] / ycusp - 1.);

// /* eigenvalue update using perturbation theory */

//         de = dfcusp / ddx12 * ycusp * ycusp * dx;
//         if (de > 0.) {
//           elw = e;
//         }
//         if (de < 0.) {
//           eup = e;
//         }

// /* prevent e to go out of bounds, i.e. e > eup or e < elw */
// /* (might happen far from convergence) */

//         e = e + de;
//         e = MIN (e,eup);
//         e = MAX (e,elw);
//         /* if ( e > eup ) e=eup;
//            if ( e < elw ) e=elw; */
//       }
//     }
// /* ---- convergence not achieved ----- */
//     if ( ABS(de) > eps ) {
//       if ( ncross != nodes ) {
//          fprintf(stderr, "ncross=%4d nodes=%4d icl=%4d e=%16.8e elw%16.8e eup=%16.8e\n", 
//       ncross, nodes, icl,  e, elw, eup);
//       } else {
//          fprintf(stderr, "e=%16.8e  de= %16.8e\n", e, de);
//       }
//       fprintf(stderr, " solve_sheq not converged after %d iterations\n",n_iter);
//       exit (1);
//     }
// /* ---- convergence has been achieved ----- */
//     fprintf(stdout, "convergence achieved at iter # %4d, de = %16.8e\n",
//       kkk, de);
//     free(f);
//     return e;
// }