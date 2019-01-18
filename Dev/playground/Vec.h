#pragma once
// Linear algebra vector
#include <iostream>
#include <cmath>
// using namespace std;
// T should be long, double or complex<double>
typedef double DataType;

template <typename T>
class Vec{
private:
    T * p;
    size_t n;
public:
    Vec(): p(NULL),n(0){}
    Vec(const size_t & m): p(new T[m]), n(m){}
    Vec(const size_t & m, const T & x): p(new T[m]), n(m){for(size_t i=0; i<m; i++) p[i]=x;}
    Vec(const Vec & v1): p(new T[v1.siz()]), n(v1.siz()){for(size_t i=0; i<n; i++) p[i]=v1[i];}
    size_t siz() const {return n;}
    void resiz(const size_t & nn){
        if(n!=nn){
            T* pnew = new T[nn];
            // size_t ncut = (nn<n)?nn:n;
            // for(size_t i=0; i<ncut; i++)
            //     pnew[i] = p[i];
            delete[] p;
            p = pnew;
            n = nn;
        }
    }
    T & operator [] (size_t i){return p[i];}
    const T & operator [] (size_t i) const {return p[i];}
    void operator = (const Vec & mv){
        resiz(mv.siz());
        for(size_t i=0; i<n; i++){
            p[i] = mv[i];
        }
    }
    void setAll(const size_t & nn, const T & x){
        this->resiz(nn);
        for(size_t i=0; i<nn; i++)
            p[i] = x;
    }
    void setAll(const T & x){for(size_t i=0; i<n; i++) p[i] = x;}

    template <typename T1>
    void operator +=  (const Vec<T1> & x){
        if(x.siz()==n) for(size_t k=0; k<n; k++) p[k] += x[k];
        else std::cout<<"Error: incompatible shape"<<std::endl;
    }
    template <typename T1>
    void operator -= (const Vec<T1> & x){
        if(x.siz()==n) for(size_t k=0; k<n; k++) p[k] -= x[k];
        else std::cout<<"Error: incompatible shape"<<std::endl;
    }
    template <typename T1>
    void operator *= (const T1 & x){
        for(size_t k=0; k<n; k++) p[k] *= x;
    }
    template <typename T1>
    void operator /= (const T1 & x){
        for(size_t k=0; k<n; k++) p[k] /= x;
    }
    template <typename T1>
    void fromCopy(const Vec<T1> & v){
        if(n!=(v.siz())) this->resiz(v.siz());
        for(size_t k=0; k<n; k++) p[k] = (v[k]);
    }
    template <typename T1, typename T2>
    void fromCMult(const T1 & c, const Vec<T2> & v){
        if(n!=(v.siz())) this->resiz(v.siz());
        for(size_t k=0; k<n; k++) p[k] = (c*v[k]);
    }
    template <typename T1, typename T2>
    void fromCDivd(const T1 & c, const Vec<T2> & v){
        if(n!=(v.siz())) this->resiz(v.siz());
        for(size_t k=0; k<n; k++) p[k] = (v[k]/c);
    }
    template <typename T1, typename T2>
    void fromVPlus(const Vec<T1> & v1, const Vec<T2> & v2){
        if((v1.siz())==v2.siz()){
            this->resiz(v1.siz());
            for(size_t k=0; k<n; k++) p[k] = v1[k]+v2[k];
        }
        else std::cout<<"Error: incompatible shape"<<std::endl;
    }
    template <typename T1>
    T dot(const Vec<T1> & v1) const{
        T res =0;
        if((v1.siz())==n)
            for(size_t k=0; k<n; k++)
                res += p[k]*v1[k];
        else std::cout<<"Error: incompatible shape"<<std::endl;
        return res;
    }
    DataType norm() const{
        DataType res=0;
        for(size_t k=0; k<n; k++)
            res += pow(abs(p[k]), 2);
        return sqrt(res);
    }
    friend std::ostream & operator<<(std::ostream & os, const Vec & v){
        size_t n = v.siz();
        size_t k=0;
        for(; k+1<n; k++)
            os<<v.p[k]<<",";
        os<<v.p[k]<<std::endl;
        return os;
    }
    void destruct(){
        if(p!=NULL) delete[] p;
        p=NULL;
    }
    ~Vec(){}
};

// template <typename T1>
// CDOUBLE hdot(const Vec<CDOUBLE> & a, const Vec<T1> & b){
//     CDOUBLE res = 0.0;
//     size_t n;
//     if((n=a.siz())==b.siz())
//         for(size_t k=0; k<n; k++) res += conj(a[k])*b[k];
//     return res;
// }

// void conjTo(Vec<CDOUBLE> & vdst, const Vec<CDOUBLE> & vsrc){
//     size_t n;
//     if((n=vsrc.siz())!=vdst.siz()) vdst.resiz(n);
//     for(size_t p=0; p<n; p++)
//         vdst[p] = conj(vsrc[p]);
// }

//void copyTo(Vec<CDOUBLE> & vdst, const Vec<DataType> & vsrc){
//    size_t n;
//    if((n=vsrc.siz())!=vdst.siz()) vdst.resiz(n);
//    for(size_t p=0; p<n; p++)
//        vdst[p] = vsrc[p];
//}

// template <typename T1>
// void ensure(Vec<T1> * & pv, size_t nn){
//         if(pv!=NULL && pv->siz()!=nn){
//             pv->resiz(nn);
//         }
//         if(pv==NULL)
//             pv=new Vec<T1>(nn);
// }
//template <class T>
//class Mat: public Bvec<
