#pragma once
#define INVPOW_MAXITER 20

#include "LUdecomp.h"
#include "Lst.h"
#include <fstream>
#include <iomanip>
#define KMAX 1.5
//#define SIGMAX 0.2
#define ORDERSUP3 16
#define ORDERSUP4 21
#define SIGMAT 10

using namespace std;

double invPower(const TridiagRS & T, Vec<DataType> & V, double eval0, double rtol=1e-14);
void wriToFile(const char * nm, const char * title, const Vec<DataType> & v, const double & h);
void wriToFile(const char * nm, const char * title, const Vec<DataType> & v, const Vec<DataType> & x);

CDOUBLE ii = 1.0i;

class H1D{
public:
    size_t nx;
    double hx;
    size_t nt;
    double ht;
    H1D(size_t Nx, double xtep):psi0(Nx), psic(Nx), ham0(Nx), xi(Nx), iHt_half(Nx){
        nx = Nx;
        hx = xtep;
        if(nx%2){
            for(size_t k=0; k<nx; k++) xi[k] = hx*k-hx*(nx/2);
        }
        else{
            for(size_t k=0; k<nx; k++) xi[k] = hx*k-hx*((nx/2)+0.5);
        }
        omega = 0;
        xcAbsorb = -0.75*(xi[0]);
        dpl = NULL;
        acc = NULL;
    }
    double ground1s(bool write=false){
        size_t i; DataType xii;
        for(i=0; i+1<nx; i++){
            xii = xi[i];
            ham0.diag()[i] = 1/(hx*hx)-1/sqrt(1+xii*xii);
            ham0.parau()[i] = -0.5/(hx*hx);
        }
        xii = xi[i];
        ham0.diag()[i] = (1/(hx*hx)-1/sqrt(1+xii*xii));
//        cout<<"Now H_0 is constructed"<<endl;
        double E0 = invPower(ham0, psi0, -0.67);
//        cout<<"E0 obtained"<<endl;
        psi0 /= psi0.norm();
        if(write){
            cout<<"Writing out"<<endl;
            ofstream ofile("quest1.csv");
            ofile<<"x, psi, prob"<<endl;
            for(size_t i=0;i<nx; i++){
                ofile<<xi[i]<<","<<psi0[i]<<","<<pow(psi0[i],2)<<endl;
            }
            ofile<<"E_0,"<<setprecision(11)<<E0;
            ofile.close();
        }
        return E0;
    }

    void setInfrared(double Intens, double lambda, short Ncar, double phi0=0){
        // Intens in 10^{16}W/cm^2
        E0 = sqrt(Intens/3.5094448314);
        omega = 45.5633525316/lambda;
        Nc = Ncar;
        ph0 = phi0;
        cout<<"Laser field Set: E0="<<E0<<", omega="<<omega<<", Nc="<<Nc<<endl;
    }
    void setTimeGrid(double tfinal, size_t Nt_1){
        nt = Nt_1+1;
        ht = tfinal/Nt_1;
        initime();
        cout<<"Time grid Set. dt*nt = "<<ht<<"*"<<nt<<endl;
    }
    void setTimeGrid(double dt){
        if(omega==0.){cout<<"Infrared not defined"<<endl; return;}
        double tf = Nc*2*M_PI/omega;
        size_t Ntminus1 = tf/dt;
        setTimeGrid(tf, Ntminus1);
    }

    Vec<DataType> * evolve_for_p1s(bool write=false){
        if(jt!=0) initime();
        Vec<DataType> * p1s= new Vec<DataType>(nt);
        while(jt+1<nt){
//            cout<<jt<<"\t";
            (*p1s)[jt] = norm( hdot(psic, psi0) );
            CrankNicol();
//            psic /= psic.norm();
        }
        (*p1s)[jt] = norm( hdot(psic, psi0) );
        if(write){
            cout<<"Writing out"<<endl;
            wriToFile("quest2_P1s-t.csv","t, P1s(t)",*p1s, ht);
        }
        return p1s;
    }
    Vec<DataType> * q2_momentSpec(size_t nk, bool write=false){
        if(jt+1!=nt){cout<<"Evolution is insufficient."<<endl; return NULL;}
        Vec<CDOUBLE> psif(psic), psi0tmp(nx);
        Vec<DataType> ki(nk);
        psi0tmp.fromCopy(psi0);
        psi0tmp *= conj(hdot(psif, psi0));
        psif -= psi0tmp;
        Vec<DataType> * Pk = new Vec<DataType>(nk);
        size_t ik, l; double k, hk=KMAX*2/(nk-1);
        CDOUBLE f;
        for(ik=0; ik<nk; ik++){
            f=0.;
            if(nk%2){ k = hk*ik-hk*(nk/2); }
            else{k = hk*ik-hk*((nk/2)+0.5);}
            ki[ik] = k;
            for(l=0; l<nx; l++)
                f += psif[l]*exp(-k*(xi[l])* ii );
            (*Pk)[ik] = norm(f)*hx;
        }
        if(write){
            wriToFile("quest2_Pk.csv", "k,P(k)", *Pk, ki);
        }
        return Pk;
    }

    void evolve_for_A(bool write=false, bool q3=true){
        if(jt!=0) initime();
        bool rec_dpl = (dpl!=NULL), rec_acc=(acc!=NULL);
        //Evolution, recording d(t) and a(t)
        while(jt<nt-1){
            if(rec_dpl) (*dpl)[jt] = dipole();
            if(rec_acc) (*acc)[jt] = accele();
            CrankNicol();
            boundAbsorb();
            psic /= psic.norm();
//            if(jt%200==0) cout<<jt<<endl;
        }
        if(rec_dpl) (*dpl)[jt] = dipole();
        if(rec_acc) (*acc)[jt] = accele();
        if(write){
            if(rec_dpl) wriToFile(q3?"q3_dipole.csv":"q4_dipole.csv", "t,d(t)", *dpl, ht);
            if(rec_acc) wriToFile(q3?"q3_accele.csv":"q4_accele.csv", "t,a(t)", *acc, ht);
        }
    }

    void q3_spec(size_t nw, Vec<DataType> & dplSpec, Vec<DataType> & accSpec, bool write=false){
        ensure(dpl, nt); ensure(acc, nt);
        evolve_for_A(write);
        double tmp, wm, hw=ORDERSUP3*omega/(nw-1);
        for(size_t iw=0; iw<nw; iw++){
            wm = iw*hw;
            tmp = tFourierNormSq(wm, *dpl);
            tmp *= pow(wm, 4);
            dplSpec[iw] = log10(tmp);
            tmp = tFourierNormSq(wm, *acc);
            accSpec[iw] = log10(tmp);
        }
        if(write){
            cout<<"Writing out"<<endl;
            wriToFile("quest3-dipoleSpec.csv", "omega,log_10(|A(omega)|^2)", dplSpec, hw/omega);
            wriToFile("quest3-acceleSpec.csv", "omega,log_10(|A(omega)|^2)", accSpec, hw/omega);
        }
    }


    void q4_spec(int nt00, size_t nw, Lst< Vec<DataType> > & res, bool write=false){
        ensure(acc, nt);
        if(dpl!=NULL){delete dpl; dpl=NULL;}
        evolve_for_A(write, false);
        double hw = ORDERSUP4*omega/(nw-1), ht0;
        size_t it0, iw;
        size_t nt0 = (nt00<=0)?nt:nt00;
        ht0 = 2*Nc*M_PI/omega/(nt0-1);
        res.resiz(nt0);
        for(it0=0; it0<nt0; it0++){
//            cout<<it0<<"\t";
            res[it0].resiz(nw);
            for(iw=0; iw<nw; iw++)
                res[it0][iw] = log10(norm(tfAnna(it0*ht0, iw*hw)));
        }
        if(write){
            hw /= omega; ht0 *=omega/(2*M_PI);
            cout<<"Writing out"<<endl;
            ofstream ofile("quest4_tfAnalysis.csv");
            ofile<<"t0";
            for(iw=0; iw<nw; iw++) ofile<<","<<iw*hw;
            ofile<<endl;
            for(it0=0;it0<nt0; it0++){
                ofile<<it0*ht0;
                for(iw=0;iw<nw;iw++)
                    ofile<<","<<res[it0][iw];
                if(it0+1<nt0) ofile<<endl;
            }
            ofile.close();
        }
        cout<<"time-freq analysis finished"<<endl;
        hw *= omega;
        Vec<DataType> accSpec(nw);
        for(iw=0; iw<nw; iw++){
            accSpec[iw] = log10(tFourierNormSq(iw*hw, *acc));
        }
        if(write) wriToFile("quest4-acceleSpec.csv", "omega,log_10(|A(omega)|^2)", accSpec, hw/omega);
    }

    ~H1D(){
        if(dpl!=NULL) delete dpl;
        if(acc!=NULL) delete acc;
    }

protected:
    Vec<DataType> psi0;
    size_t jt; // current time slice index
    Vec<CDOUBLE> psic; // current state
    TridiagRS ham0;
    Vec<DataType> xi;
    Vec<DataType> * dpl;
    Vec<DataType> * acc;

    void CrankNicol(){//evolve from jt slice to jt+1
        size_t i;
//        for(i=0; i<nx; i++)
//                iHt_half.diag()[i] = (ham0.diag()[i] + xi[i]*elecField((jt+0.5)*ht))*0.5*ht*ii ;
        TridiagC hamtmp(iHt_half);
        for(i = 0; i<nx; i++) hamtmp.diag()[i] = (ham0.diag()[i] + xi[i]*elecField((jt+0.5)*ht))*0.5*ht*ii;
        Vec<CDOUBLE> * const tmp = MVprod(hamtmp, psic); // iHt_half  is \frac{i}{2}\tau H(t_j)
        psic -= (*tmp); // (1-\frac{i}{2}\tau H(t_j))\psi^{(j)}
        // j+1 slice
        // iHt_half obtains \frac{i}{2}\tau H(t_{j+1})
        for(i = 0; i<nx; i++) hamtmp.diag()[i] += 1;
        Thomas evolvCore(hamtmp, true);
        evolvCore.solve(psic); // psic obtains \psi^{(j+1)}
//        for(i=0; i<nx; i++)
//                iHt_half.diag()[i] -= 1.0;// recovered, ready for j+1->j+2
        delete tmp;
        jt++;
    }
private:
    double E0;
    double omega;
    short Nc;
    double ph0;
    double xcAbsorb;
    TridiagC iHt_half;

    void initime(){
        psic.fromCopy(psi0); //copy the initial state to the recorder
        iHt_half.copyFrom(ham0); //since E(t=0) =0, no other term
        iHt_half  *= 0.5*ht*ii;
        jt = 0; // initialize the time pointer
    }
    double elecField(const double & t) const {
        double s=sin(omega*t/(2*Nc));
        return E0*s*s*sin(omega*t+ph0);
    }
    void boundAbsorb(){
        double xx, sx=0.2;
        size_t i=0;
        while(  (xx= abs(xi[i])) > xcAbsorb  ){
            psic[i] *= exp(-pow((xx-xcAbsorb)/sx,2));
            i++;
        }
        i=nx-1;
        while(  (xx= abs(xi[i])) > xcAbsorb  ){
            psic[i] *= exp(-pow((xx-xcAbsorb)/sx,2));
            i--;
        }
    }

    DataType dipole() const {
        DataType tmp=0;
        for(size_t ix=0; ix<nx; ix++) tmp += norm(psic[ix])*(xi[ix]);
        return tmp*hx;
    }
    DataType accele() const {
        DataType tmp=0, xii;
        for(size_t ix=0; ix<nx; ix++){
            xii  = xi[ix];
            tmp += norm(psic[ix])*(-xii/pow(1+xii*xii, 1.5)+elecField(jt*ht));// \int dx
        }
        return tmp*hx;
    }
    double tFourierNormSq(const double & w, const Vec<DataType> & timeSieries) const {
    // squared norm of t-omega Fourier Transform of the expectation of (diagonal) operator A
        CDOUBLE res = 0, www=exp(-w*ht*ii); //double tmp;
        size_t jtsup, jjt;
        if(jt+1<nt){
            cout<<"Warning: evolution is insufficient!"<<endl;
            jtsup = jt+1;
        }
        else{jtsup=nt;}
        for(jjt=0; jjt<jtsup; jjt++)// \int dt
            res += pow(www, jjt)*(timeSieries[jjt]);
        res *= ht;
        return norm(res)/(2*M_PI);
    }
    CDOUBLE tfAnna(const double & t0, const double & w, double st=SIGMAT) const {
        CDOUBLE www=exp(-w*ht*ii), res=0; double tti;
        for(size_t it=0; it<nt; it++){
            tti = (it*ht-t0)/st;
            res += pow(www, it)*exp(-0.5*tti*tti)*((*acc)[it]);
        }
        return res*ht;
    }
};

template <typename T1>
T1 maxbar(const Vec<T1> & v){
    T1 res = 0;
    for(size_t i=0; i< v.siz(); i++){
        if(abs(v[i]) > abs(res)) res = v[i];
    }
    return res;
}

double invPower(const TridiagRS & T, Vec<DataType> & V, double eval0, double rtol){
    //In order for the precision, use the original Hamiltonian
    //rather than the tridiagonal one.
    // The eigenvalue in the vicinity of eval0 will be returned
    // and the corresponding eigenvector will be stored in v
    double eigval;
    TridiagRS T1(T);
    size_t n = V.siz(), i, niter=0;
    V.setAll(1.);
    Vec<DataType> u(n);
    DataType dv_norm = 1., tmp;
    for(i=0; i<n; i++) T1.diag()[i] -= eval0;
    Chol_3band clsk3bd(T1, true);
    while(dv_norm>rtol){
        u = V;
        clsk3bd.solve(V);
//        cout<<V;
        V *= (eigval = 1./maxbar(V));
        dv_norm = 0.;
        for(i=0; i<n; i++)
            if( (tmp=abs(V[i]-u[i]))>dv_norm ) dv_norm=tmp;
        if(niter++>=INVPOW_MAXITER){
            cout<<"InvPow Warning: Reached Maxiter. "<<rtol<<" cannot be satisfied. ";
            cout<<"Finally |dv| = "<<dv_norm<<endl;
            return eval0;
        }
    }
//    cout<<"The while loop finished"<<endl;
    return eigval+eval0;
}

void wriToFile(const char * nm, const char * title, const Vec<DataType> & v, const double & h){
    ofstream ofile(nm);
    ofile<<title<<endl;
    size_t i;
    for(i=0; i+1<v.siz(); i++){
        ofile<<(i*h)<<","<<v[i]<<endl;
    }
    ofile<<(i*h)<<","<<v[i];
    ofile.close();
}

void wriToFile(const char * nm, const char * title, const Vec<DataType> & v, const Vec<DataType> & x){
    ofstream ofile(nm);
    ofile<<title<<endl;
    size_t i;
    for(i=0; i+1<v.siz(); i++){
        ofile<<x[i]<<","<<v[i]<<endl;
    }
    ofile<<x[i]<<","<<v[i];
    ofile.close();
}

//void testInvPow(size_t N){
//    TridiagRS T(N);
//    T.diag()[0]=-1;
//    T.diag()[1]=2;
//    T.diag()[2]=3;
//    T.diag()[3]=-4;
//    T.parau().setAll(-0.4);
//    cout<<T.diag()<<"\t"<<T.parau()<<endl;
//    Vec<DataType> v(N);
//    double res = invPower(T, v, 1.9);
//    cout<<res<<endl;
//    cout<<v<<endl;
//    Vec<DataType> * recov = MVprod(T, v);
//    v *= res;
//    cout<<v<<(*recov)<<endl;
//    delete recov;
//}
