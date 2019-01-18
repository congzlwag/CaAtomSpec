#pragma once

#include "Vec.h"
#include <stddef.h>
#include <cstdlib>

template <typename T>
class Triband
{
public:
    Vec<T> d;
    Vec<T> pd;
    Vec<T> pu;
    
    Triband(){};
    Triband(const size_t & _n):d(Vec<T>(_n)),pd(Vec<T>(_n-1)),pu(Vec<T>(_n-1)){}
    // Triband(const Triband<T> & _trb):d(_trb.d),pu(_trb.pu),pd(_trb.pd){}
    Triband(const Vec<T> & _d, const Vec<T> & _pu, const Vec<T> & _pd):d(_d),pu(_pu),pd(_pd){}
    size_t dim() const {return d.siz();}
    const T loc(const size_t & i, const size_t & j) const {
        if(i==j) return d[i];
        if(j==i+1) return pu[i];
        if(i==j+1) return pd[j];
        std::exit(-3);
    }
    template <typename T1>
    void operator = (const Triband<T1> & trb){
        d = trb.d;
        pu = trb.pu;
        pd = trb.pd;
    }
    template <typename T1>
    void operator *= (const T1 & x){
        d *= x;
        pu *= x;
        pd *= x;
    }
    template <typename T1>
    void operator += (const Triband<T1> & trb){
        d += trb.d;
        pu += trb.pu;
        pd += trb.pd;
    }
    void fromCMult(const T & x, const Triband & trb){
        d.fromCMult(x, trb.d);
        pu.fromCMult(x, trb.pu);
        pd.fromCMult(x, trb.pd);
    }
    void destruct(){
        d.destruct();
        pu.destruct();
        pd.destruct();
    }
    ~Triband(){
//        cout<<"destructed as 3band"<<endl;
    }
};

template <typename T1>
void MVprod(const Triband<T1> & T, const Vec<T1> & v, Vec<T1> & vout){
    size_t nn = T.dim();
    if(nn==v.siz()){
        size_t i =0;
        vout[0] = T.d[0]* v[0] + T.pu[0]* v[1];
        for(i=1; i<nn-1; i++){
            vout[i] = T.d[i]*v[i] + T.pu[i]*v[i+1] + T.pd[i-1] * v[i-1];
        }
        vout[i] = T.d[i]*v[i] +T.pd[i-1] * v[i-1];
    }
    else std::exit(-3);
}

template <typename T1>
void LUdcmp(Triband<T1> & Tdcmp){
    size_t n = Tdcmp.dim(), k=0;
    T1 u;
    u= (Tdcmp.pu[k] /= Tdcmp.d[k]);
    for(k=1; k<n-1; k++){
        Tdcmp.d[k] -= (Tdcmp.pd[k-1]*u);
        u= (Tdcmp.pu[k] /= Tdcmp.d[k]);
    }
    Tdcmp.d[k] -= (Tdcmp.pd[k-1]*u);
}

template <typename T1, typename T2>
void LUsolve(const Triband<T1> & Tdcmp, Vec<T2> & v){
    size_t n = Tdcmp.dim(), k=0;
    v[0] /= Tdcmp.d[0];
    for(k=1; k<n; k++){
        v[k] -= v[k-1]*(Tdcmp.pd[k-1]);
        v[k] /= Tdcmp.d[k];
    }
    for(k=n-1; k>0; k--){
        v[k-1] -= v[k]*(Tdcmp.pu[k-1]);
    }
}

// template <typename T1>
// class Thomas
// {
// private:
//     Triband<T1> * Tdcmp;
//     bool onst;
// public:
//     Thomas(Triband<T1> & T, bool onsite){
//         size_t n = T.dim(), k=0;
//         onst = onsite;
//         if(onsite){Tdcmp = &T;}
//         else{
//             Tdcmp = new Triband<T1>(n);
//             (*Tdcmp) = T;
//         }
//         T1 u;
//         u= (Tdcmp->pu[k] /= Tdcmp->d[k]);
//         for(k=1; k<n-1; k++){
//             Tdcmp->d[k] -= (Tdcmp->pd[k-1]*u);
//             u= (Tdcmp->pu[k] /= Tdcmp->d[k]);
//         }
//         Tdcmp->d[k] -= (Tdcmp->pd[k-1]*u);
//     }
//     void solve(Vec<T1> & v){
//         size_t n = Tdcmp->dim(), k=0;
//         v[0] /= Tdcmp->d[0];
//         for(k=1; k<n; k++){
//             v[k] -= v[k-1]*(Tdcmp->pd[k-1]);
//             v[k] /= Tdcmp->d[k];
//         }
//         for(k=n-1; k>0; k--){
//             v[k-1] -= v[k]*(Tdcmp->pu[k-1]);
//         }
//     }
//     const Triband<T1> & LUdcmp(){
//         return *Tdcmp ;
//     }
//     ~Thomas(){
//         if(!onst&& Tdcmp!=NULL){delete Tdcmp;}
//     }
// };

// class TridiagRS:public Triband<DataType>
// {
// public:
//     TridiagRS(size_t nn): Triband(){
//         this->d = new Vec<DataType>(nn);
//         this->pu = new Vec<DataType>(nn-1);
//         this->pd = NULL;
//         this->n = nn;
//     }
//     TridiagRS(const TridiagRS & T): Triband(){
//         this->d = new Vec<DataType>(*(T.d));
//         this->pu = new Vec<DataType>(*(T.pu));
//         this->pd = NULL;
//         this->n = T.n;
//     }
//     void operator *= (const DataType & x){
//         (*(this->d)) *= x;
//         (*(this->pu)) *= x;
//     }
//     ~TridiagRS(){    }
// };

// template <typename T1>
// Vec<T1> * MVprod(const TridiagRS & T, const Vec<T1> & v){
//     size_t nn = T.dim();
//     if(nn==v.siz()){
//         size_t i=0;
//         Vec <T1>* res = new Vec<T1>(nn);
//         (*res)[i] = T.diag()[i]* v[i] + T.parau()[i]* v[i+1];
//         for(i=1; i<nn-1; i++){
//             (*res)[i] = T.diag()[i]*v[i] +T.parau()[i] * v[i+1]+T.parau()[i-1] * v[i-1];
//         }
//         (*res)[i]= T.diag()[i]*v[i]+T.parau()[i-1]*v[i-1];
//         return res;
//     }
//     else cout<<"Error: incompatible length"<<endl;
//     return NULL;
// }

// class TridiagC: public Triband<CDOUBLE>
// {
// public:
//     TridiagC(size_t nn):Triband<CDOUBLE>(nn){}
//     TridiagC(const TridiagC & T):Triband<CDOUBLE>(T.n){
//         (*(this->d)) = *(T.d);
//         (*(this->pu)) = *(T.pu);
//         (*(this->pd)) = *(T.pd);
//     }
//     void copyFrom(const TridiagRS & Trs){
//         (this->d)->fromCopy(Trs.diag());
//         (this->pu)->fromCopy(Trs.parau());
//         (this->pd)->fromCopy(Trs.parau());
//         this->n = Trs.dim();
//     }
//     Vec<CDOUBLE> & parad(){return *pd;}
//     const Vec<CDOUBLE> & parad() const {return *pd;}
//     void operator *= (const CDOUBLE & x){
//         (*(this->d)) *= x;
//         (*(this->pu)) *= x;
//         (*(this->pd)) *= x;
//     }
//     ~TridiagC(){}
// };

// template <typename T1>
// Vec<CDOUBLE> * MVprod(const TridiagC & T, const Vec<T1> & v){
//         size_t nn = T.dim();
//         if(nn==v.siz()){
//             size_t i =0;
//             Vec<CDOUBLE> * res = new Vec<CDOUBLE>(nn);
//             (*res)[0] = T.diag()[0]* v[0] + T.parau()[0]* v[1];
//             for(i=1; i<nn-1; i++){
//                 (*res)[i] = T.diag()[i]*v[i] +T.parau()[i] * v[i+1]+T.parad()[i-1] * v[i-1];
//             }
//             (*res)[i] = T.diag()[i]*v[i] +T.parad()[i-1] * v[i-1];
//             return res;
//         }
//         else cout<<"Error: incompatible length"<<endl;
//         return NULL;
// }

// void testTridiagC(size_t N){
//     TridiagC T(N);
//     T.diag().setAll(1);
//     T.parau().setAll(0.);
//     T.parau()[0] = 0.5i;
//     T.parau()[1] = 0.1-0.1i;
//     T.parad().setAll(0.);
//     T.parad()[0]= -0.5i;
//     T.parad()[1] = 0.1+0.1i;
//     T *= 1i;
//     TridiagC T1(T);
//     T1.diag().setAll(0.);
//     cout<<T.diag()<<T.parau()<<T.parad()<<endl;
// }
