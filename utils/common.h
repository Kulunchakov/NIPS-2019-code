#ifndef COMMON_H
#define COMMON_H
#include "linalg.h"
//#include "decomp.h"
#define PI T(3.14159265359)

#ifdef TIMINGS 
#define NTIMERS 15
Timer time_mat[NTIMERS];

#ifdef CUDA
#define RESET_TIMERS \
   for (int llll = 0; llll < NTIMERS; ++llll) { \
      cudaDeviceSynchronize(); \
      time_mat[llll].stop(); \
      time_mat[llll].reset(); \
   }
#define START_TIMER(i) \
   cudaDeviceSynchronize(); \
   time_mat[i].start(); 
#define STOP_TIMER(i) \
   cudaDeviceSynchronize(); \
   time_mat[i].stop(); 
#else
#define RESET_TIMERS \
   for (int llll = 0; llll < NTIMERS; ++llll) { \
      time_mat[llll].stop(); \
      time_mat[llll].reset(); \
   }
#define START_TIMER(i) \
   time_mat[i].start(); 
#define STOP_TIMER(i) \
   time_mat[i].stop(); 
#endif

#define PRINT_TIMERS \
   for (int llll = 0; llll < NTIMERS; ++llll) { \
      cout << "Timer " << llll << endl; \ 
      time_mat[llll].printElapsed(); \
   }
#else
#define START_TIMER(i) 
#define RESET_TIMERS
#define STOP_TIMER(i) 
#define PRINT_TIMER(i) 
#endif

//#ifdef MKL_DNN
//#include "common_dnn.h"
//#endif
//

template <typename T>
inline T compute_sigma(const Matrix<T>& X, const T quantile) {
   T* prX=X.rawX();
   const INTM m = X.m();
   const INTM n = X.n();
   const INTM n2=static_cast<INTM>((n/2));
   Vector<T> tmpT(n2);
   T* tmp=tmpT.rawX();
#pragma omp parallel for
   for (INTM ii=0; ii<n2; ++ii) {
      tmp[ii]=0;
      for (INTM jj=0; jj<m; ++jj)  {
         const T tmp2=prX[ii*m+jj]-prX[(ii+n2)*m+jj];
         tmp[ii]+=tmp2*tmp2;
      }
      tmp[ii]=sqr<T>(tmp[ii]);
   }
   tmpT.sort(true);
   const int ind=floor(quantile*n2);
   return (ind == n2-1) ? tmpT[ind] : 0.5*(tmpT[ind]+tmpT[ind+1]);
};

template <typename T>
inline void centering(Matrix<T>& X, const INTM V = 1) {
   T* prX=X.rawX();
   const INTM n = X.n();
   const INTM m = X.m();
#pragma omp parallel for
   for (INTM ii=0;ii <n; ++ii) {
      Vector<T> mean(V);
      mean.setZeros();
      for (INTM jj=0;jj<m;jj+=V) /// assumes R,G,B,R,G,B,R,G,B
         for (INTM kk=0; kk<V; ++kk) 
            mean[kk]+=prX[ii*m+jj+kk];
      for (INTM kk=0; kk<V; ++kk)
         mean[kk] *= V/static_cast<T>(m);
      for (INTM jj=0;jj<m;jj+=V) 
         for (INTM kk=0; kk<V; ++kk) 
            prX[ii*m+jj+kk] -= mean[kk];
   }
};

#define EPS_NORM 0.00001

template <typename T>
inline void normalize(Matrix<T>& X, Vector<T>& norms) {
   T* prX=X.rawX();
   const INTM n = X.n();
   const INTM m = X.m();
   norms.resize(n);
//#pragma omp parallel for
   for (INTM ii=0;ii <n; ++ii) {
      norms[ii]=cblas_nrm2<T>(m,prX+m*ii,1);
      cblas_scal<T>(m,T(1.0)/MAX(norms[ii],T(EPS_NORM)),prX+ii*m,1);
   }
};

template <typename T>
inline void normalize(Vector<T>& X) {
   T* prX=X.rawX();
   const INTM m = X.n();
   T nrm=cblas_nrm2<T>(m,prX,1);
   cblas_scal<T>(m,T(1.0)/MAX(nrm,T(EPS_NORM)),prX,1);
}

template <typename T>
inline void normalize(Matrix<T>& X) {
   T* prX=X.rawX();
   const INTM n = X.n();
   const INTM m = X.m();
//#pragma omp parallel for
   for (INTM ii=0;ii <n; ++ii) {
      T nrm=cblas_nrm2<T>(m,prX+m*ii,1);
      cblas_scal<T>(m,T(1.0)/MAX(nrm,T(EPS_NORM)),prX+ii*m,1);
   }
};

template <typename T>
void inline whitening(Matrix<T>& X) {
   Matrix<T> Wfilt;
   Vector<T> mu;
   X.meanCol(mu);
   Vector<T> ones(X.n());
   ones.set(T(1.0));
   X.rank1Update(mu,ones,-T(1.0));
   Matrix<T> U;
   Vector<T> S;
   X.svd2(U,S,X.m(),2);
   const T maxS=S.fmaxval();
   for (int ii=0; ii<S.n(); ++ii)
      S[ii] = (S[ii] > maxS*1e-8) ? T(1.0)/alt_sqrt<T>(S[ii]) : 0;
   Matrix<T> U2;
   U2.copy(U);
   U2.multDiagRight(S);
   U2.mult(U,Wfilt,false,true);
   Matrix<T> tmp, tmp2;
   for (INTM ii = 0; ii<X.n(); ii+=10000) {
      const INTM size_block=MIN(10000,X.n()-ii);
      X.refSubMat(ii,size_block,tmp);
      tmp2.copy(tmp);
      Wfilt.mult(tmp2,tmp);
   }
};




template <typename T>
void inline whitening(Matrix<T>& X, Matrix<T>& Wfilt, Vector<T>& mu) {
   X.meanCol(mu);
   Vector<T> ones(X.n());
   ones.set(T(1.0));
   X.rank1Update(mu,ones,-T(1.0));
   Matrix<T> U;
   Vector<T> S;
   X.svd2(U,S,X.m(),2);
   const T maxS=S.fmaxval();
   for (int ii=0; ii<S.n(); ++ii)
      S[ii] = (S[ii] > maxS*1e-8) ? T(1.0)/sqr<T>(S[ii]) : 0;
   Matrix<T> U2;
   U2.copy(U);
   U2.multDiagRight(S);
   U2.mult(U,Wfilt,false,true);
   Matrix<T> tmp, tmp2;
   for (INTM ii = 0; ii<X.n(); ii+=10000) {
      const INTM size_block=MIN(10000,X.n()-ii);
      X.refSubMat(ii,size_block,tmp);
      tmp2.copy(tmp);
      Wfilt.mult(tmp2,tmp);
   }
};

template <typename T> struct Map {
   public:
      Map() : _x(0), _y(0), _z(0) { };
      Map(T* X, const INTM x, const INTM y, const INTM z) { 
         this->setData(X,x,y,z);
      };
      virtual ~Map() {  };
      void copy(const Map<T>& map) { 
         _x=map._x;
         _y=map._y;
         _z=map._z;
         _vec.copy(map._vec);
      }
      void resize(const INTM x,const INTM y,const INTM z) {
         _x=x;
         _y=y;
         _z=z;
         _vec.resize(x*y*z);
      }
      void refSubMapZ(const INTM ind, Map<T>& map) const {
         map._x=_x;
         map._y=_y;
         map._z=1;
         map._vec.setData(_vec.rawX()+_x*_y*ind,_x*_y);
      }
      void setData(T* X, const INTM x, const INTM y, const INTM z) { 
         _vec.setData(X,x*y*z);
         _x=x;
         _y=y;
         _z=z;
      };
      inline INTM x() const {return _x; };
      inline INTM y() const {return _y; };
      inline INTM z() const {return _z; };
      inline void print() const { _vec.print("map"); };
      inline void print_size() const { printf("%d x %d x %d \n",_x,_y,_z); };
      inline T* rawX() const {return _vec.rawX(); };
      void two_dim_gradient(Map<T>& DX, Map<T>& DY) const;
      void subsampling_new(Map<T>& out,const int factor, const T beta, const bool verbose = false) const;
      void im2col(Matrix<T>& out,const int e, const bool zero_pad = false, const int stride = 1, const bool normal = false) const;
      void col2im(const Matrix<T>& in,const int e, const bool zero_pad = false, const bool normal = true, const int stride = 1);
      void refMat(Matrix<T>& out) const { out.setData(_vec.rawX(),_x,_y*_z); };
      void refVec(Vector<T>& out) const { out.setData(_vec.rawX(),_x*_y*_z); };
   private:
      INTM _x;
      INTM _y;
      INTM _z;
      Vector<T> _vec;
};

template <typename T>
void Map<T>::two_dim_gradient(Map<T>& DX, Map<T>& DY) const {
   assert(_x==1);
   DX.resize(_x,_y,_z);
   DY.resize(_x,_y,_z);
   T* dx = DX.rawX();
   T* dy = DY.rawX();
   T* X = _vec.rawX();
   const INTM m = _y;
   // compute DX;
   cblas_copy<T>(m*(_z-1),X+m,1,dx,1);
   cblas_copy<T>(m,X+(_z-1)*m,1,dx+(_z-1)*m,1);
   cblas_axpy<T>(m,-T(1.0),X,1,dx,1);
   cblas_axpby<T>(m*(_z-2),-T(0.5),X,1,T(0.5),dx+m,1);
   cblas_axpy<T>(m,-T(1.0),X+(_z-2)*m,1,dx+(_z-1)*m,1);
   // compute DY;
#pragma omp parallel for
   for (INTM ii=0; ii<_z; ++ii) {
      const INTM ind=ii*m;
      dy[ind]=X[ind+1]-X[ind];
      cblas_copy<T>(m-2,X+ind+2,1,dy+ind+1,1);
      cblas_axpby<T>(m-2,-T(0.5),X+ind,1,T(0.5),dy+ind+1,1);
      dy[ind+m-1]=X[ind+m-1]-X[ind+m-2];
   }
};

template <typename T>
void Map<T>::im2col(Matrix<T>& out,const int e, const bool zero_pad, const int stride, const bool normal) const {
   const INTM n1=zero_pad ? _y : _y-e+1;
   const INTM n2=zero_pad ? _z : _z-e+1;
   const INTM s=e*e*_x;
   T* X=_vec.rawX();
   const INTM nn1=ceil(n1 / static_cast<double>(stride));
   const INTM nn2=ceil(n2 / static_cast<double>(stride));
   out.resize(s,nn1*nn2);
   T* Y=out.rawX();
   out.setZeros();
   INTM num_patch=0;
   if (zero_pad) {
      const INTM ed2=e/2;
      for (INTM jj=0; jj<n2; jj+=stride) {
         for (INTM ii=0; ii<n1; ii+=stride) {
            const int kmin = MAX(jj-ed2,0);
            const int kmax = MIN(jj-ed2+e,_z);
            const int imin = MAX(ii-ed2,0);
            const int imax = MIN(ii-ed2+e,_y);
            const INTM ex=(imax-imin)*_x;
            if (normal) {
               for (INTM kk=kmin; kk<kmax; ++kk) {
                  cblas_axpy<T>(ex,static_cast<T>(e*e)/((kmax-kmin)*(imax-imin)),X+(kk*_y+imin)*_x,1,Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,1);
               }
            } else {
               for (INTM kk=kmin; kk<kmax; ++kk) {
                  memcpy(Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,X+(kk*_y+imin)*_x,ex*sizeof(T));
               }
            }
            ++num_patch;       
         }
      }
   } else {
      const INTM ex=e*_x;
      for (INTM jj=0; jj<n2; jj+=stride) {
         for (INTM ii=0; ii<n1; ii+=stride) {
            for (INTM kk=0; kk<e; ++kk) {      
               memcpy(Y+num_patch*s+kk*ex,X+((jj+kk)*_y+ii)*_x,ex*sizeof(T));      
            }  
            ++num_patch;       
         }     
      }
   }
}

template <typename T>
void Map<T>::col2im(const Matrix<T>& in,const int e, const bool zero_pad, const bool norm, const int stride) {
   const INTM n1=zero_pad ? _y : _y-e+1;
   const INTM n2=zero_pad ? _z : _z-e+1;
   const INTM s=e*e*_x;
   _vec.setZeros();
   Vector<T> count;
   count.copy(_vec);
   T* X=_vec.rawX();
   T* pr_count=count.rawX();
   T* Y=in.rawX();
   _vec.setZeros();
   INTM num_patch=0;
   if (zero_pad) {
      const INTM ed2=e/2;
      for (INTM jj=0; jj<n2; jj+=stride) {
         for (INTM ii=0; ii<n1; ii+=stride) {
            const int kmin = MAX(jj-ed2,0);
            const int kmax = MIN(jj-ed2+e,_z);
            const int imin = MAX(ii-ed2,0);
            const int imax = MIN(ii-ed2+e,_y);
            const INTM ex=(imax-imin)*_x;
            for (INTM kk=kmin; kk<kmax; ++kk) {
               cblas_axpy<T>(ex,T(1.0),Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,1,X+(kk*_y+imin)*_x,1);
            }
            if (norm) {
               for (INTM kk=kmin; kk<kmax; ++kk) {
                  for (INTM ll=0; ll<ex; ++ll)
                     pr_count[(kk*_y+imin)*_x+ll]++;
               }
            }
            ++num_patch;       
         }
      }
   } else {
      const INTM ex=e*_x;
      for (INTM jj=0; jj<n2; jj+=stride) {
         for (INTM ii=0; ii<n1; ii+=stride) {
            for (INTM kk=0; kk<e; ++kk) {
               cblas_axpy<T>(ex,T(1.0),Y+num_patch*s+kk*ex,1,X+((jj+kk)*_y+ii)*_x,1);
            }  
            if (norm) {
               for (INTM kk=0; kk<e; ++kk) {
                  for (INTM ll=0; ll<ex; ++ll)
                     pr_count[((jj+kk)*_y+ii)*_x+ll]++;
               }
            }
            ++num_patch;       
         }     
      }
   }
   if (norm)
      _vec.div(count);
}

template <typename T>
void Map<T>::subsampling_new(Map<T>& out,const int sub, const T sigma, const bool verbose) const {
   const INTM sizeimage=_y*_z;
   const INTM h = _y;
   const INTM w = _z;
   const INTM hout = ceil((float(h))/sub);
   const INTM wout = ceil((float(w))/sub);
   const INTM diffh = h- ((hout-1)*sub+1);
   const bool even = diffh % 2 == 1;
   const INTM offset = even ? diffh/2 + 1 : diffh/2;
   const INTM s = even ? 2*sub: 2*sub+1;
   if (verbose && even)
      printf("Even filter\n");
   if (verbose)
      printf("Offset %d\n",offset);
   T* filt = new T[s];
   T sum = 0;
   if (even) {
      for(int i=-sub; i<sub; ++i){
         const T ind=i+T(0.5);
         filt[i+sub] = exp(-(1.0/(2*sigma*sigma))*ind*ind);
         sum += filt[i+sub];
      }
   } else {
      for(int i=-sub; i<=sub; ++i){
         filt[i+sub] = exp(-(1.0/(2*sigma*sigma))*i*i);
         sum += filt[i+sub];
      }
   }
   for(int i=0; i<s; ++i){
      filt[i] /= sum;
   }
   const INTM m = _x;
   out.resize(_x,hout,wout);
   T* prIn =_vec.rawX();
   T* prOut =out._vec.rawX();
   memset(prOut, 0, sizeof(T)*hout*wout*m);
   T* buff = new T[m*w*hout];
   const T subdisc = even ? sub-1 : sub;

#pragma omp parallel for
   for(INTM ii = 0; ii<w; ++ii) {
      for(INTM jj = 0; jj<hout; ++jj) {
         const INTM dx=jj*sub+offset;
         const INTM discardx =-MIN(dx - sub,0);
         const INTM discardy =MAX(dx + subdisc - h +1,0);
         cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),prIn + (ii*h+dx -sub + discardx)*m,m,filt+discardx,1,0,buff+(jj*w+ii)*m,1);
      }
   }

#pragma omp parallel for
   for (INTM ii=0; ii<hout; ++ii) {
      for (INTM jj=0; jj<wout; ++jj) {
         const INTM dx=jj*sub+offset;
         const INTM discardx =-MIN(dx - sub,0);
         const INTM discardy =MAX(dx + subdisc - w +1,0);
         cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),buff + (ii*w+dx -sub + discardx)*m,m,filt+discardx,1,0,prOut+(jj*hout+ii)*m,1);
      }
   }
   delete[](filt);
   delete[](buff);
};

template <typename T>
inline T convert_image_data(const double in) {
   return static_cast<T>(in);
};

template <typename T>
inline T convert_image_data(const float in) {
   return static_cast<T>(in);
};

template <typename T>
inline T convert_image_data(const unsigned char in) {
   return static_cast<T>(in)/255;
};

template <typename Tin, typename Tout>
inline void convert_image_data_map(const Tin* input, Tout* output, const int n) {
   for (int ii=0; ii<n; ++ii) output[ii]=convert_image_data<Tout>(input[ii]);
};

template <typename Tin, typename Tout>
inline void convert_image_data_map_switch(const Tin* input, Tout* output, const int nc, const int channels, const int nimages, const bool augment = false) {
   if (augment) {
      const int h = static_cast<int>(sqrt(nc));
      for (int ii=0; ii<nimages; ++ii)  {
         const bool flip = (random() % 2) == 0;
         const int offset1 = static_cast<int>((random() % 5))-2;
         const int offset2 = static_cast<int>((random() % 5))-2;
         for (int jj=0; jj<channels; ++jj) 
            for (int kk=0; kk<h; ++kk) 
               for (int ll=0; ll<h; ++ll) {
                  const int ind_output= flip ? (kk)*h + h-1 -ll  : kk*h+ll;
                  const int ind_input= (kk+offset1)*h+ll+offset2;
                  if (kk + offset1 < 0 || kk + offset1 >= h || ll + offset2 < 0 || ll + offset2 >= h) {
                     output[ii*nc*channels+ind_output*channels+jj] = 0;
                  } else {
                     output[ii*nc*channels+ind_output*channels+jj]=convert_image_data<Tout>(input[ii*nc*channels+jj*nc+ind_input]);
                  }
            }
      }
   } else {
      for (int ii=0; ii<nimages; ++ii) 
         for (int jj=0; jj<channels; ++jj) 
            for (int kk=0; kk<nc; ++kk) 
               output[ii*nc*channels+kk*channels+jj]=convert_image_data<Tout>(input[ii*nc*channels+jj*nc+kk]);
   }
};

template <typename T>
inline void convert_nchw_to_nhwc(T* output,const T* input,const int h, const int w, const int c, const int n) {
   const int s=h*w;
   const int nc=h*w*c;
   for (int ii=0; ii<n; ++ii) 
      for (int jj=0; jj<c; ++jj) 
         for (int kk=0; kk<s; ++kk) 
            output[ii*nc+kk*c+jj]=input[ii*nc+jj*s+kk];
};



template <typename Tin, typename T>
inline void get_zeromap(const Map<Tin>& mapin, Map<T>& map, const int type_layer) {
   Tin* pr_im = mapin.rawX();
   const INTM ex = mapin.x(); // image is assumed to be e x e x mapin.z
   const INTM ey = mapin.y(); // image is assumed to be e x e x mapin.z
   if (ey == 3*ex) { // assumes this means RGB  (RRR,GGG,BBB)
      if (type_layer==4) {
         map.resize(1,ex,ex); // extract green channel
         T* X = map.rawX();
         for (INTM ii=0; ii<ex; ++ii)
            for (INTM jj=0; jj<ex; ++jj)
               X[ii*ex+jj]=convert_image_data<T>(pr_im[ii*ex+ex*ex+jj]);
      } else {
         map.resize(3,ex,ex);
         T* X = map.rawX();
         for (INTM kk=0; kk<3; ++kk)
            for (INTM ii=0; ii<ex; ++ii)
               for (INTM jj=0; jj<ex; ++jj)
                  X[(ii*ex+jj)*3+kk]=convert_image_data<T>(pr_im[ii*ex+kk*ex*ex+jj]); // output is R,G,B,R,G,B,R,G,B
      }
   } else {// assumes this means gray scale
      map.resize(1,ex,ey);
      T* X = map.rawX();
      for (INTM ii=0; ii<ey; ++ii)
         for (INTM jj=0; jj<ex; ++jj)
            X[ii*ex+jj]=convert_image_data<T>(pr_im[ii*ex+jj]);
   }
};

typedef enum
{
   SQLOSS = 0,
   ONE_VS_ALL_SQHINGE_LOSS = 1,
   SQLOSS_CONV = 2,
   NEGNORM = 3
} loss_t;

typedef enum
{
   POOL_GAUSSIAN_FILTER = 0,
   POOL_AVERAGE = 1, // not implemented on cpu
} pooling_mode_t;

template <typename T> struct Layer {
   int num_layer;
   int npatch;
   int nfilters;
   int subsampling;
   int stride;
   bool zero_padding;
   int type_layer;
   /// 0 = RAW
   /// 1 = centering
   /// 2 = centering + global whitening
   /// 3 = global whitening
   /// 4 = gradient
   /// 5 = centering + whitening per image
   int type_kernel;
   /// 0 = Gaussian
   /// 1 = polynomial, degree (x'x)^2
   T sigma;
   Matrix<T> W;
   Vector<T> b;
   Matrix<T> Wfilt;
   Vector<T> mu;
   Matrix<T> W2;
   pooling_mode_t pooling_mode;
};

template <typename T>
inline void pre_processing(Matrix<T>& X, const Layer<T>& layer, const int channels) {
   if (layer.type_layer == 1 || layer.type_layer == 2 || layer.type_layer == 5)
      centering(X,channels);
   if (layer.type_layer == 5) 
      whitening(X);
   if (layer.type_layer == 2 || layer.type_layer==3) {
      Vector<T> ones(X.n());
      ones.set(T(1.0));
      X.rank1Update(layer.mu,ones,-T(1.0));
      if (layer.type_layer==2) {
         Matrix<T> Z;
         Z.copy(X);
         layer.Wfilt.mult(Z,X);
      }
   }
}

template <typename T>
inline void encode_layer(const Map<T>& mapin, Map<T>& mapout, const Layer<T>& layer,const bool verbose = false) {
   Map<T> map;
   if (layer.num_layer==1 && layer.type_layer==4) {
      Map<T> DX, DY;
      mapin.two_dim_gradient(DY,DX);
      const INTM num_orients=layer.nfilters;
      const INTM mx = DX.y();
      const INTM my = DX.z();
      map.resize(num_orients,mx,my);
      const T* dx = DX.rawX();
      const T* dy = DY.rawX();
      T* X = map.rawX();
      Vector<T> theta(num_orients);
      Vector<T> costheta(num_orients);
      Vector<T> sintheta(num_orients);
      const T sigma=layer.sigma;
      for (int ii=0; ii<num_orients; ++ii) theta[ii]=(2*PI/num_orients)*ii;
      for (int ii=0; ii<num_orients; ++ii) costheta[ii]=cos(theta[ii]);
      for (int ii=0; ii<num_orients; ++ii) sintheta[ii]=sin(theta[ii]);
#pragma omp parallel for
      for (INTM ii=0; ii<mx; ++ii) {
         for (INTM jj=0; jj<my; ++jj) {
            const T rho = sqr<T>(dx[ii*my+jj]*dx[ii*my+jj]+dy[ii*my+jj]*dy[ii*my+jj]);
            for (INTM kk=0; kk<num_orients; ++kk) {
               const T ddx=(dx[ii*my+jj]/rho-costheta[kk]);
               const T ddy=(dy[ii*my+jj]/rho-sintheta[kk]);
               X[ii*(num_orients*my) +jj*num_orients+kk] = rho ? rho*exp(-(T(1.0)/(2*sigma*sigma))*(ddx*ddx+ddy*ddy)) : 0;
            }
         }
      }
   } else {
      Matrix<T> X;
      mapin.im2col(X,layer.npatch,layer.zero_padding,layer.stride);
      pre_processing(X,layer,layer.num_layer==1 ? mapin.x() : 1);
      const int yyout = layer.zero_padding ? mapin.y() : mapin.y() - layer.npatch + 1;
      const int zzout = layer.zero_padding ? mapin.z() : mapin.z() - layer.npatch + 1;
      const int yout = ceil(yyout/static_cast<double>(layer.stride));
      const int zout = ceil(zzout/static_cast<double>(layer.stride));
      Vector<T> ones(X.n());
      ones.set(T(1.0));
      Vector<T> norms;
      normalize(X,norms);
      map.resize(layer.W.n(), yout, zout);
      Matrix<T> Y(map.rawX(), layer.W.n(), yout*zout);
      layer.W.mult(X,Y,true);
      if (layer.type_kernel==0) {
         Y.rank1Update(layer.b,ones);
         Y.exp();
      } else if (layer.type_kernel==1) {
         Y.pow(T(2.0));
      }
      Y.multDiagRight(norms);
   }
   if (layer.subsampling > 1) {
      const T beta =layer.subsampling/sqr<T>(T(2.0));
      map.subsampling_new(mapout,layer.subsampling,beta,verbose);
   } else {
      mapout.copy(map);
   }
   Matrix<T> Z(mapout.rawX(), mapout.x(), mapout.y()*mapout.z());
   Matrix<T> Z2;
   Z2.copy(Z);
   layer.W2.mult(Z2,Z);
}


template <typename Tin, typename T>
inline void encode_ckn(const Map<Tin>& maps, Layer<T> layers[], const int nlayers, Matrix<T>& psi) {
   Timer time;
   time.start();
   const int n = maps.z();
   int count=0;
#pragma omp parallel for
   for (int ii=0; ii<n; ++ii) {
      Map<T> map;
      Map<Tin> mapii;
      maps.refSubMapZ(ii,mapii); 
      encode_ckn_map(mapii,layers,nlayers,map,false);
      memcpy(psi.rawX()+ii*psi.m(),map.rawX(),psi.m()*sizeof(T));
   }
   time.printElapsed();
};

template <typename Tin, typename T>
inline void encode_ckn_map(const Map<Tin>& mapin, Layer<T> layers[], const int nlayers, Map<T>& mapout, const bool verbose = false) {
   Map<T> maptmp;
   get_zeromap(mapin,mapout,layers[0].type_layer);
   for (int jj=0; jj<nlayers; ++jj) {

      maptmp.copy(mapout);
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0
#endif
      encode_layer(maptmp,mapout,layers[jj],verbose);
   }
};

template <typename Tin, typename T>
inline void extract_dataset(const Map<Tin>& maps, Layer<T> layers[], const int nlayers, Matrix<T>& X, Vector<int>& labels) {
   Timer time;
   time.start();
   const int n = maps.z();
   const int per_image=X.n()/n;
   Layer<T>& last_layer=layers[nlayers-1];
#pragma omp parallel for
   for (int ii=0; ii<n; ++ii) {
      Map<T> map;
      Map<Tin> mapii;
      maps.refSubMapZ(ii,mapii); 
      encode_ckn_map(mapii,layers,nlayers-1,map);
      Matrix<T> psi;
      map.im2col(psi,last_layer.npatch,false,1);

      /// track empty patches
      const int m = psi.n();
      Matrix<T> psiC;
      psiC.copy(psi);
      centering(psiC,nlayers==1 ? map.x() : 1);
      Vector<int> per;
      per.randperm(m);
      int count=0;
      Vector<T> col, col2;
      if (last_layer.type_layer == 5) 
         whitening(psiC);
      for (int kk=0; kk<m; ++kk) {
         X.refCol(ii*per_image+count,col);
         if (labels.n() > 0) labels[ii*per_image+count]=ii;
         if (last_layer.type_layer==1 || last_layer.type_layer==2 || last_layer.type_layer==5) {
            psiC.refCol(per[kk],col2);
         } else {
            psi.refCol(per[kk],col2);
         }
         col.copy(col2);
         ++count;
         if (count==per_image) break;
      }
   }
   time.printElapsed();
};


template <typename T>
inline void whitening_map(Map<T>& map, const bool zero_pad = false) {
   Matrix<T> X;
   map.im2col(X,3,zero_pad);
   centering(X,3);
   whitening(X);
   map.col2im(X,3,zero_pad);
};

template <typename T>
inline void whitening_maps(Map<T>& maps) {
   Timer time;
   time.start();
   const int n = maps.z();
   int count=0;
#pragma omp parallel for
   for (int ii=0; ii<n; ++ii) {
      Map<T> map, map_zero;
      maps.refSubMapZ(ii,map); 
      get_zeromap(map,map_zero,0);
      whitening_map(map_zero);
      T* mapX = map.rawX();
      T* map_zeroX = map_zero.rawX();
      const int c = map_zero.x();
      const int h = map_zero.y();
      const int w = map_zero.z();
      for (int jj = 0; jj<w; ++jj)
         for (int kk=0; kk<h; ++kk)
            for (int ll=0; ll<c; ++ll) 
               mapX[ll*h*w+jj*h+kk]=map_zeroX[jj*(h*c)+kk*c+ll];
   }
   time.printElapsed();
};



#endif
