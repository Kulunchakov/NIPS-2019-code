#include <mexutils.h>
#include <common.h>

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],const int nrhs, const int nlhs) {
   Matrix<T> X;
   getMatrix<T>(prhs[0],X);
   if (nlhs == 1) {
      Vector<T> nrms;
      plhs[0]=createMatrix<T>(1,X.n());
      getVector<T>(plhs[0],nrms);
      normalize(X,nrms);
   } else {
      normalize(X);
   }
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
   if (nrhs != 1)
      mexErrMsgTxt("Bad number of inputs arguments");

   if (nlhs != 1 && nlhs != 0)
      mexErrMsgTxt("Bad number of output arguments");

   if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      callFunction<double>(plhs,prhs,nrhs,nlhs);
   } else {
      callFunction<float>(plhs,prhs,nrhs,nlhs);
   }
}

