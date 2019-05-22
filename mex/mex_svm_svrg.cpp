
/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mex.h>
#include <mexutils.h>
#include <svm.h>

// w=mexSvmMiso(y,X,tablambda,param);

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
      const int nlhs) {
   const int EPOCHS_THRESHOLD = 602;
   if (!mexCheckType<T>(prhs[0])) 
      mexErrMsgTxt("type of argument 1 is not consistent");
   if (mxIsSparse(prhs[0])) 
      mexErrMsgTxt("argument 1 should not be sparse");

   if (!mexCheckType<T>(prhs[1])) 
      mexErrMsgTxt("type of argument 2 is not consistent");

   if (!mxIsStruct(prhs[3])) 
      mexErrMsgTxt("argument 4 should be a struct");

   T* pry = reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dimsy=mxGetDimensions(prhs[0]);
   INTM my=static_cast<INTM>(dimsy[0]);
   INTM ny=static_cast<INTM>(dimsy[1]);
   Vector<T> y(pry,my*ny);

   T* prX = reinterpret_cast<T*>(mxGetPr(prhs[1]));
   const mwSize* dimsX=mxGetDimensions(prhs[1]);
   INTM p=static_cast<INTM>(dimsX[0]);
   INTM n=static_cast<INTM>(dimsX[1]);
   Matrix<T> X(prX,p,n);

   T* prw0 = reinterpret_cast<T*>(mxGetPr(prhs[2]));
   const mwSize* dimsw0=mxGetDimensions(prhs[2]);
   INTM pw0=static_cast<INTM>(dimsw0[0]);
   INTM nw0=static_cast<INTM>(dimsw0[1]);
   Vector<T> w0(prw0,pw0*nw0);


//   const int nclasses=y.maxval()+1;
//   plhs[0]=createMatrix<T>(p,nclasses);
//   T* prw=reinterpret_cast<T*>(mxGetPr(plhs[0]));
//   Matrix<T> W(prw,p,nclasses);
   plhs[0]=createMatrix<T>(p,1);
   T* prw=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Vector<T> w(prw,p);
   w.copy(w0);

   srandom(0);
   Vector<T> logs; logs.resize(EPOCHS_THRESHOLD);
   for (int i = 0; i < logs.n(); i++) { logs[i] = T(-1.0); }
   Vector<T> ticks; ticks.resize(EPOCHS_THRESHOLD);
   for (int i = 0; i < ticks.n(); i++) { ticks[i] = -1; }
   

   int threads = getScalarStructDef<int>(prhs[3],"threads",-1);
   const int seed = getScalarStruct<int>(prhs[3],"seed");
   const int loss = getScalarStructDef<int>(prhs[3],"loss",1);
   const int epochs = getScalarStructDef<int>(prhs[3],"epochs",100);
   const int k0 = getScalarStructDef<int>(prhs[3],"k0",300);
   const int budget = getScalarStructDef<int>(prhs[3],"budget",1);
   const int n_stages = getScalarStructDef<int>(prhs[3],"n_stages",1);
   const int eval_freq= getScalarStructDef<int>(prhs[3],"eval_freq",1);
   const int start_iter = getScalarStructDef<int>(prhs[3],"start_iter",0);
   const int epochs_per_stage = getScalarStructDef<int>(prhs[3],"epochs_per_stage",2);
   const int minibatch= getScalarStructDef<int>(prhs[3],"minibatch",1);
   const int workhorse = getScalarStructDef<int>(prhs[3],"workhorse",1);
   const int init_epochs = getScalarStructDef<int>(prhs[3],"init_epochs",0);
   int n_oracle_calls = 0;

   const T l1_scale = getScalarStructDef<T>(prhs[3],"l1_scale",0);
   const T kappa = getScalarStructDef<T>(prhs[3],"kappa",0);
   const T lambda = getScalarStruct<T>(prhs[3],"lambda");
   const T L = getScalarStruct<T>(prhs[3],"L");
   const T dropout = getScalarStructDef<T>(prhs[3],"dropout",0);

   const bool averaging = getScalarStructDef<bool>(prhs[3],"averaging",false);
   const bool decreasing = getScalarStructDef<bool>(prhs[3],"decreasing",false);
   const bool is_warm_yk = getScalarStructDef<bool>(prhs[3],"is_warm_yk",true);
  
   const bool is_fair = getScalarStructDef<bool>(prhs[3],"is_fair",false);
   const bool do_catalyst = getScalarStructDef<bool>(prhs[3],"do_catalyst",false);
   const bool do_acc_svrg = getScalarStructDef<bool>(prhs[3],"do_acc_svrg",false);
   const bool do_acc_sgd = getScalarStructDef<bool>(prhs[3],"do_acc_sgd",false);
   const bool do_saga = getScalarStructDef<bool>(prhs[3],"do_saga",false);
   const bool do_svrg = getScalarStructDef<bool>(prhs[3],"do_svrg",false);
   const bool do_sgd = getScalarStructDef<bool>(prhs[3],"do_sgd",false);
   const bool do_katyusha = getScalarStructDef<bool>(prhs[3],"do_katyusha",false);
   

   const bool verbose = getScalarStructDef<bool>(prhs[3],"verbose",false);
   const bool method_verbose = getScalarStructDef<bool>(prhs[3],"method_verbose",false);
   srandom(seed);
   if (threads == -1) {
      threads=1;
#ifdef _OPENMP
      threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 
   threads=init_omp(threads);
   // we have 6-7 methods to track (question about acc_sgd or sgd?)
   
   if (do_catalyst) 
      // workhorse:
      // 1: SVRG
      // 2: SAGA
      catalyst(y,X,w,L,lambda,kappa,budget,epochs,k0,n_stages,decreasing,dropout,eval_freq,logs,ticks,l1_scale,loss,workhorse,is_fair,verbose,method_verbose,is_warm_yk);
   
   if (do_saga) 
      saga_miso(y,X,w,L,lambda,budget,epochs,init_epochs,averaging,decreasing,dropout,eval_freq,logs,ticks,T(0.0),l1_scale,loss,T(0.0),{},is_fair,verbose,n_oracle_calls);
   if (do_svrg) 
      random_svrg(y,X,w,L,lambda,budget,epochs,init_epochs,averaging,decreasing,dropout,eval_freq,logs,ticks,l1_scale,loss,T(0.0),{},is_fair,verbose,n_oracle_calls);
   if (do_acc_svrg) 
      acc_random_svrg(y,X,w,L,lambda,budget,epochs,init_epochs,decreasing,dropout,eval_freq,logs,ticks,l1_scale,loss,T(0.0),{},verbose);
   if (do_sgd) 
      sgd(y,X,w,L,lambda,budget,epochs,init_epochs,averaging,decreasing,dropout,eval_freq,logs,ticks,l1_scale,loss,T(0.0),{},verbose);
   if (do_acc_sgd and minibatch==1)
      acc_sgd(y,X,w,L,lambda,budget,epochs,init_epochs,decreasing,dropout,eval_freq,logs,ticks,l1_scale,loss,T(0.0),{},verbose);
   if (do_acc_sgd and minibatch > 1) 
      acc_sgd_batch(y,X,w,L,lambda,budget,epochs,init_epochs,decreasing,dropout,eval_freq,logs,ticks,minibatch,l1_scale,loss,T(0.0),{},verbose);
   if (do_katyusha) 
      katyusha(y,X,w,L,lambda,budget,epochs,dropout,eval_freq,logs,ticks,l1_scale,loss,verbose);
   // postprocessing (we erase all excessive -1):

   int trim_position = logs.n() + 1;
   for (int i = 0; i < logs.n(); i++) {
      if (logs[i] == T(-1.0)) trim_position = min(trim_position, i);
   }

   plhs[1]=createMatrix<T>(trim_position,1);
   T* prlogs=reinterpret_cast<T*>(mxGetPr(plhs[1]));
   Vector<T> output_logs(prlogs,trim_position);
   for (int i = 0; i < output_logs.n(); i++) {
      output_logs[i] = logs[i];
   }

   plhs[2]=createMatrix<T>(trim_position,1);
   T* prticks=reinterpret_cast<T*>(mxGetPr(plhs[2]));
   Vector<T> output_ticks(prticks,trim_position);
   for (int i = 0; i < output_ticks.n(); i++) {
      output_ticks[i] = ticks[i];
   }
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 4)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 3) 
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs);
      }
   }


