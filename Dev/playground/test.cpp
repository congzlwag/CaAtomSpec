#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>
#include <cmath>
#include <Vec.h>

PyObject * castNPYArray(const Vec<double> & v_){
    npy_intp m[1] = {v_.siz()};
    PyObject * out = PyArray_SimpleNew(1,m,NPY_DOUBLE);
    double * ptr = (double *)(PyArray_GETPTR1((PyArrayObject *)out,0));
    for(size_t k=0; k<m[0]; k++)
        ptr[k] = v_[k];
    return out;
}

static PyObject * Test(PyObject *self, PyObject *args){
    PyArrayObject *in_array;
    PyObject      *out_array;
    // PyArrayIterObject *in_iter;
    // PyArrayIterObject *out_iter;
    double dd;
    Vec<double> vv(10);
    for(size_t k=0; k<10; k++){
        vv[k] = 1.5*k;
    }
    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "Od", &in_array, &dd))
        return NULL;

    /*  construct the output array, like the input array */
    // out_array = PyArray_NewLikeArray(in_array, NPY_ANYORDER, NULL, 0);
    // if (out_array == NULL)
    //     return NULL;

    /*  create the iterators */
    /* TODO: this iterator API is deprecated since 1.6
     *       replace in favour of the new NpyIter API */
    // in_iter  = (PyArrayIterObject *)PyArray_IterNew((PyObject*)in_array);
    // size_t N = PyArray_SIZE(in_array);
    // npy_intp m[1] = {N};
    // out_array = PyArray_SimpleNew(1,m,NPY_DOUBLE);
    // // PyArray_ENABLEFLAGS((PyArrayObject*)out_array, NPY_ARRAY_OWNDATA);
    // if (out_array == NULL)
    //     return NULL;
    // // out_iter = (PyArrayIterObject *)PyArray_IterNew(out_array);
    // // if (in_iter == NULL || out_iter == NULL)
    // //     goto fail;

    // /*  iterate over the arrays */
    // double * in_ptr = (double *)(PyArray_GETPTR1(in_array,0));
    // double * out_ptr= (double *)(PyArray_GETPTR1((PyArrayObject *)out_array,0));
    // for(size_t kl=0; kl<N; kl++){
    //     *(out_ptr + kl) = cos(*(in_ptr + kl));
    // }
    // while (in_iter->index < in_iter->size
    //         && out_iter->index < out_iter->size) {
    //     /* get the datapointers */
    //     double * in_dataptr = (double *)in_iter->dataptr;
    //     double * out_dataptr = (double *)out_iter->dataptr;
    //     /* cosine of input into output */
    //     *out_dataptr = cos(*in_dataptr);
    //     /* update the iterator */
    //     PyArray_ITER_NEXT(in_iter);
    //     PyArray_ITER_NEXT(out_iter);
    // }

    /*  clean up and return the result */
    // Py_DECREF(in_iter);
    // Py_DECREF(out_iter);
    out_array = castNPYArray(vv);
    vv.destruct();
    Py_INCREF(out_array);
    return Py_BuildValue("(dO)",0.5,out_array);

        // /*  in case bad things happen */
        // fail:
        //     Py_XDECREF(out_array);
        //     Py_XDECREF(in_iter);
        //     Py_XDECREF(out_iter);
        //     return NULL;
}

static PyMethodDef module_methods[] = {
  // "Python name", C function,  arg_representation, doc
  {"test", (PyCFunction)Test, METH_VARARGS,"Playground function"},
  {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "testa", "Testing Playground",
     -1, module_methods, NULL, NULL, NULL, NULL, };

  PyMODINIT_FUNC PyInit_testa(void) {
    // import Numpy API
    import_array();
    return PyModule_Create(&moduledef);
  }
#else
  PyMODINIT_FUNC inittesta(void) {
    PyObject * m;
    m = Py_InitModule3("testa", module_methods, "Testing Playground");
    if (m == NULL) return;
    //  import Numpy API
    import_array();
  }
#endif

// static PyObject * Test(PyObject * self, PyObject * args){
	// double x;
	// if(!(PyArg_ParseTuple(args,"d",&x)))
	// 	return NULL;
	// Vec<double> v1(2,x);
 //  v1[1] += 1;
 //  x = v1[0]*v1[1];
 //  v1.destruct();
	// return Py_BuildValue("d", x*x);
// }
/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */