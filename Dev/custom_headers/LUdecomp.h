#pragma once

#include <iostream>
#include <cmath>
#include "tridiag.h"

class Chol_3band{
public:
    Chol_3band(TridiagRS & T, bool onsite){
        n=T.dim();
        onst = onsite;
        if(onsite){Tdcmp = &T;}
        else{Tdcmp = new TridiagRS(T);}
        DataType u; size_t k=0;
        u= (Tdcmp->parau()[k] /= Tdcmp->diag()[k]);
        for(k=1; k+1<n; k++){
            Tdcmp->diag()[k] -= (Tdcmp->diag()[k-1]*u*u);
            u = (Tdcmp->parau()[k] /= Tdcmp->diag()[k]);
        }
        Tdcmp->diag()[k] -= (Tdcmp->diag()[k-1]*u*u);
    }
    void solve(Vec<DataType> & v){
        size_t k;
        for(k=1; k<n; k++)
            v[k] -= (Tdcmp->parau()[k-1])*v[k-1];
    //    cout<<v<<endl;
        for(k=0; k<n; k++)
            v[k] /= Tdcmp->diag()[k];
        for(k=n-1; k>0; k--)
            v[k-1] -= Tdcmp->parau()[k-1]*v[k];
    }
    ~Chol_3band(){   if(!onst){delete Tdcmp;}    }
private:
    TridiagRS * Tdcmp;
    size_t n;
    bool onst;
};

class Thomas{
public:
    Thomas(TridiagC & T, bool onsite){
        n=T.dim();
        onst = onsite;
        if(onsite){Tdcmp = &T;}
        else{Tdcmp = new TridiagC(T);}
        size_t k=0; CDOUBLE u;
        u= (Tdcmp->parau()[k] /= Tdcmp->diag()[k]);
        for(k=1; k<n-1; k++){
            Tdcmp->diag()[k] -= Tdcmp->parad()[k-1]*u;
            u= (Tdcmp->parau()[k] /= Tdcmp->diag()[k]);
        }
        Tdcmp->diag()[k] -= T.parad()[k-1]*u;
    }
    void solve(Vec<CDOUBLE> & v){
        size_t k=0;
        v[0] /= Tdcmp->diag()[0];
        for(k=1; k<n; k++){
            v[k] -= v[k-1]*(Tdcmp->parad()[k-1]);
            v[k] /= Tdcmp->diag()[k];
        }
        for(k=n-1; k>0; k--){
            v[k-1] -= v[k]*(Tdcmp->parau()[k-1]);
        }
    }
    ~Thomas(){
        if(!onst){delete Tdcmp;}
    }
private:
    TridiagC * Tdcmp;
    size_t n;
    bool onst;
};


void testChol3band(size_t N){
TridiagRS T(N);
for(size_t i=0; i<N; i++){T.diag()[i] = (i+1)*(i+1); if(i<N-1) T.parau()[i] = 0.1*i;}
//T.parau()[2]=0.3; T.parau()[2]=0.1;
Chol_3band core(T, false);
Vec<DataType> b(N, 2.);
b[2]=0.5; b[0]=-2;
cout<<"Originally, b="<<b;
core.solve(b);
cout<<"Solution: x="<<b;
Vec<DataType> * rec = MVprod(T, b);
cout<<"Recovery: b="<<(*rec);
delete rec;
}

void testThomas(size_t N){
    TridiagC T(N);
    T.diag().setAll(1);
    T.parau().setAll(0.);
    T.parau()[0] = 0.5i;
    T.parau()[1] = 0.1-0.1i;
    T.parad().setAll(0.);
    T.parad()[0]= -0.5i;
    T.parad()[1] = 0.1+0.1i;
    T *= 1i;
    cout<<T.diag()<<T.parau()<<T.parad()<<endl;
    Vec<CDOUBLE> b(N, 2.0);
    Thomas thms(T, false);
    thms.solve(b);
    cout<<T.diag()<<T.parau()<<T.parad()<<endl;
//    cout<<T.diag()<<T.parau()<<endl;
    cout<<b<<endl;
    Vec<CDOUBLE> * r = MVprod(T, b);
    cout<<(*r)<<endl;
    delete r;
}

//void LUdcomp(Mat & a, Vec & indx)
//{
//    // Partial pivoting Crout algorithm
//    // Finally replaces a with the LU decomposition of
//    // a rowwise permutation of a
//    // The permutation has its effect recorded in indx
//    const double tol = 1e-30;
//    int i, imax, j, k;
//    double big, dum, sum, tmp;
//
//    int n= a.dim(0);
//    Vec u(n); // scaler container
//    for(i=0; i<n; i++){
//        big = 0.;
//        for(j=0; j<n; j++)
//            if( (tmp=fabs(a.loc(i,j)))>big ) big=tmp;
//        if(big==0.){std::cout<<"Error: Singular Matrix"<<endl; return;}
//        u[i] = 1./big; // scaled by infty-norm
//    }
//    for(j=0; j<n; j++){// col-j in Crout algorithm
//        for(i=0; i<j; i++){ // beta_{ij}
//            sum = a.loc(i,j);
//            for(k=0; k<i; k++) sum -= (a.loc(i,k)*a.loc(k,j));
//            a.loc(i,j) = sum;
//        }
//        big = 0.; imax=j;
//        for(i=j; i<n; i++){
//            sum = a.loc(i,j);
//            for(k=0; k<j; k++) sum -= (a.loc(i,k)*a.loc(k,j));
//            a.loc(i,j) = sum;
//            if( (dum=u[i]*fabs(sum)) >= big )
//                big = dum, imax=i;
//        }
//        if(j!=imax){ // then we need to swap row j and row imax
//            for(k=0; k<n; k++)
//                dum=a.loc(imax, k), a.loc(imax, k) = a.loc(j,k), a.loc(j,k)=dum;
//            u[imax] = u[j];
//        }
//        indx[j] = imax; //record
//        if(a.loc(j,j)==0.) a.loc(j,j)=tol;
//        if(j != n-1){
//            dum = 1./a.loc(j,j);
//            for(i=j+1; i<n; i++) a.loc(i,j) *= dum;
//        }
//    }
//}
//
//void LUsolve(const Mat & LU, const Vec & indx, Vec & b){
//    int i, ii=0, ip, j;
//    double sum;
//    int n = indx.dim();
//    for(i=0; i<n; i++){
//        ip = int(indx[i]);
//        sum=b[ip];
//        b[ip]=b[i];
//        if(ii!=0)
//            for(j=ii-1; j<i; j++) sum -=LU.loc(i,j)*b[j];
//        else{
//            if(sum != 0.0) ii=i+1;
//        }
//        b[i] = sum;
//    }
//    for(i=n-1; i>=0; i--){
//        sum = b[i];
//        for(j=i+1; j<n; j++)
//            sum -= LU.loc(i,j)*b[j];
//        b[i]  = sum/LU.loc(i,i);
//    }
//}

//void testLUsolve(int N){
//    Mat A(N,N);
//    A.setAll(0.);
//    for(int i=0; i<N; i++) A.loc(i,i) = 0.5*(10-i);
//    A.loc(0,1) = 0.9; A.loc(1,0)=0.9;
//    A.loc(N-1,N-2) = 1; A.loc(N-2, N-1)=1;
//    Vec idx(N);
//
//    Vec b(N);
//    for(int i=0; i<N; i++) b[i]=i;
//
//    cout<<"Original A="; print(A);
//    cout<<"\t b = "; print(b);
//    Mat LU(A);
//    LUdcomp(LU, idx);
//    LUsolve(LU, idx, b);
//
//    cout<<"Solution to Ax=b, x="; print(b);
//    Vec r(N); r.fromMVProd(A, b);
//    cout<<"recovery b="; print(r);
//}
