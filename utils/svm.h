#ifndef SVM_H
#define SVM_H  

#include "linalg.h"
#include <cmath>

template <typename T>
T power(T a, int b){
   T output = T(1.0);
   for (int ii = 0; ii < b; ++ii) { output *= a; }
   return output;
}

template <typename T>
T compute_next_alpha(const T prev_alpha, const T q) {
   // solve a quadratic equation x^2 + b x + c = 0
   const T c = - prev_alpha * prev_alpha;
   const T b = - c - q;
   
   const T discr = b * b - T(4.0) * c;
   const T root = T(0.5) * (sqrt(discr) - b);
   return root;
}

// calculates l1 norm of a vector
template <typename T>
T nrm1(const Vector<T>& tmp){
   const int n = tmp.n();
   T nrm1 = T(0);
   for (int ii=0; ii<n; ++ii)
      nrm1 += ABS(tmp[ii]);
   return nrm1;
}

template <typename T>
void lasso_prox(Vector<T>& tmp, const T scale) {
   const int n = tmp.n();
   for (int ii=0; ii<n; ++ii)
      tmp[ii] = T(ABS(tmp[ii]) >= scale) * (tmp[ii] - T(SIGN(tmp[ii])) * scale);
}

template <typename T>
void dropout_vec(Vector<T>& tmp, const T dropout) {
   const int n = tmp.n();
   if (dropout)
      for (int ii=0; ii<n; ++ii)
         if (random() <= RAND_MAX*dropout) tmp[ii]=0;
}


template <typename T>
T compute_loss(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq, const int loss, const T l1_scale) {
   if (loss==0) {
      return compute_loss_logistic(y,X,w,lambda,dropout,freq,l1_scale);
   } else {
      return compute_loss_sqhinge(y,X,w,lambda,dropout,freq,l1_scale);
   }
}

/// ok no dropout, fine.
template <typename T>
T compute_loss_logistic(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq, const T l1_scale) {
   const int n = y.n();
   T loss=0;
   Vector<T> tmp;
   if (dropout) {
      for (int ll=0; ll<freq; ++ll) {
         for (int kk=0; kk<n; ++kk) {
            X.copyCol(kk,tmp);
            dropout_vec(tmp,dropout);
            loss += logexp(-y[kk]*tmp.dot(w));
         }
      }
      loss *= T(1.0)/(freq*n);
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         loss += logexp(-y[kk]*tmp[kk]);
      }
      loss *= T(1.0)/(n);
   } 
   loss += T(0.5)*lambda*w.nrm2sq();
   return loss;
}


template <typename T>
T compute_loss_sqhinge(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq, const T l1_scale) {
   const int n = y.n();
   T loss=0;
   Vector<T> tmp;
   if (dropout) {
      for (int ll=0; ll<freq; ++ll) {
         for (int kk=0; kk<n; ++kk) {
            X.copyCol(kk,tmp);
            dropout_vec(tmp,dropout);
            const T los=MAX(0,1-y[kk]*tmp.dot(w));
            loss += los*los;
         }
      }
      loss *= T(0.5)/(freq*n);
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         const T los=MAX(0,1-y[kk]*tmp[kk]);
         loss += los*los;
      }
      loss *= T(0.5)/(n);
   } 
   loss += T(0.5)*lambda*w.nrm2sq();
   return loss;
}

template <typename T>
void compute_grad(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout, const int loss) {
   if (loss==0) {
      return compute_grad_logistic(y,x,w,grad,lambda,dropout);
   } else {
      return compute_grad_sqhinge(y,x,w,grad,lambda,dropout);
   }
}

template <typename T>
void compute_grad_logistic(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   grad.copy(x);
   dropout_vec(grad,dropout);
   const T s = T(1.0)/(T(1.0)+exp_alt<T>(y*grad.dot(w)));
   grad.scal(-y*s);
   grad.add(w,lambda);
}


template <typename T>
void compute_grad_sqhinge(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   grad.copy(x);
   dropout_vec(grad,dropout);
   const T s = MAX(0,1-y*grad.dot(w));
   grad.scal(-y*s);
   grad.add(w,lambda);
}

template <typename T>
void compute_fullgrad(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout, const int loss) {
   if (loss==0) {
      return compute_fullgrad_logistic(y,X,w,grad,lambda,dropout);
   } else {
      return compute_fullgrad_sqhinge(y,X,w,grad,lambda,dropout);
   }
}

template <typename T>
void compute_fullgrad_logistic(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   const int n = y.n();
   Vector<T> tmp;
   if (dropout) {
      grad.setZeros();
      grad.resize(w.n());
      for (int kk=0; kk<n; ++kk) {
         X.copyCol(kk,tmp);
         dropout_vec(tmp,dropout);
         const T s = T(1.0)/(T(1.0)+exp_alt<T>(y[kk]*tmp.dot(w)));
         grad.add(tmp,-y[kk]*s);
      }
      grad.scal(T(1.0/n));
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         const T s = T(1.0)/(T(1.0)+exp_alt<T>(y[kk]*tmp[kk]));
         tmp[kk]=-y[kk]*s;
      }
      X.mult(tmp,grad,T(1.0)/n);
   }
   grad.add(w,lambda);
}


template <typename T>
void compute_fullgrad_sqhinge(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   const int n = y.n();
   Vector<T> tmp;
   if (dropout) {
      grad.setZeros();
      grad.resize(w.n());
      for (int kk=0; kk<n; ++kk) {
         X.copyCol(kk,tmp);
         dropout_vec(tmp,dropout);
         const T s=MAX(0,1-y[kk]*tmp.dot(w));
         grad.add(tmp,-y[kk]*s);
      }
      grad.scal(T(1.0/n));
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         tmp[kk]=-y[kk]*MAX(0,1-y[kk]*tmp[kk]);
      }
      X.mult(tmp,grad,T(1.0)/n);
   }
   grad.add(w,lambda);
}

template <typename T>
void smoothify_grad(Vector<T>& grad, const T kappa, const Vector<T>& point, const Vector<T>& y) {
   Vector<T> temp;
   temp.copy(point);
   temp.add(y, T(-1.0));
   grad.add(temp, kappa);
}

// ready for the final run
template <typename T>
void acc_random_svrg(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int budget, const int epochs, const int init_epochs, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const T l1_scale = 0, const int param_loss = 1, const T kappa = 0, const Vector<T>& p_k = {}, const bool verbose=false) {
   const int n = y.n(); const int p = X.m();
   const T eta= MIN(T(1.0)/(3*L), T(1.0)/(5*lambda*n)); // the stepsize
   Vector<T> anchor, grad_anchor, grad, grad2, col, y_k, v_k;
   anchor.copy(w); y_k.copy(w); v_k.copy(w);

   /*
   Loss computation variables:
   */
   int curr_loss_slider = 0;
   int n_oracle_calls = 0;
   // logs[0] = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
   // ticks[0] = 0;

   compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss); 
   if (kappa) smoothify_grad(grad_anchor,kappa,anchor,p_k);

   if (verbose) cout << "Accelerated SVRG " << (decreasing ? "with" : "without") << " decreasing\n";
   for (int ii = 0; ii<min(budget, n*epochs); ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
         const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,12*n/(5*lambda*(T(ii+1)*T(ii+1))))  : eta;
         if (verbose) cout << "Iteration " << ii << "(" << n_oracle_calls << ")" << " - eta: " << etak << " - obj " <<  loss << endl;

         logs[curr_loss_slider]=loss;
         ticks[curr_loss_slider++]=n_oracle_calls;
      }
      const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,12*n/(5*lambda*(T(ii+1)*T(ii+1))))  : eta;
      const T deltak=sqrt(T(5.0)*etak*lambda/(3*n));
      const T thetak=(3*n*deltak-5*lambda*etak)/(3-5*lambda*etak);
      y_k.copy(v_k);
      y_k.scal(thetak);
      y_k.add(anchor,T(1.0-thetak));

      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,y_k,grad,lambda,dropout,param_loss);
      compute_grad(y[ind],col,anchor,grad2,lambda,dropout,param_loss); n_oracle_calls+=2;
      if (kappa) smoothify_grad(grad2,kappa,anchor,p_k);

      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);
      w.copy(y_k);
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);

      v_k.scal(1-deltak);
      v_k.add(y_k,deltak);
      v_k.add(w,(deltak/(lambda*etak)));
      v_k.add(y_k,-(deltak/(lambda*etak)));

      if (random() % n == 0) {
         anchor.copy(w);
         compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss); 
         if (kappa) smoothify_grad(grad_anchor,kappa,anchor,p_k);
      }
   }
}

// ready for the final run
template <typename T>
void sgd(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int budget, const int epochs, const int init_epochs, const bool averaging, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const T l1_scale = 0, const int param_loss = 1, const T kappa = 0, const Vector<T>& y_k = {}, const bool verbose=false) {
   const int n = y.n(); const int p = X.m();
   const T eta= T(1.0)/(L);
   Vector<T> wav, grad, col;
   wav.copy(w);

   /*
   Loss computation variables:
   */
   int curr_loss_slider = 0;
   int n_oracle_calls = 0;
   // logs[0] = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
   // ticks[0] = 0;

   if (verbose) cout << "SGD " << (decreasing ? "with" : "without") << " decreasing\n";
   for (int ii = 0; ii<min(budget, n*epochs); ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
         const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(2.0)/(lambda*(ii+2))) : eta;
         if (verbose) cout << "Iteration " << ii << " -- eta " << etak << " -- obj " <<  loss << endl;
         logs[curr_loss_slider]=loss;
         ticks[curr_loss_slider++]=n_oracle_calls;
      }
      
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss); n_oracle_calls++;
      if (kappa) smoothify_grad(grad,kappa,w,y_k);

      const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(2.0)/(lambda*(ii+2))) : eta;
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);
      if (averaging) {
         const T tau = lambda*etak;
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging)
      w.copy(wav);
}

// ready for the final run
template <typename T>
void acc_sgd(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int budget, const int epochs, const int init_epochs, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const T l1_scale = 0, const int param_loss = 1, const T kappa = 0, const Vector<T>& y_k = {}, const bool verbose=false) {
   const int n = y.n(); const int p = X.m();
   // const T eta= (T(1.0)/(L))*sqrt(lambda/L);
   const T eta= (T(1.0)/(L));
   Vector<T> yk, wold, grad, col;
   yk.copy(w);

   /*
   Loss computation variables:
   */
   int curr_loss_slider = 0;
   int n_oracle_calls = 0;
   // logs[0] = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
   // ticks[0] = 0;

   if (verbose) cout << "Acc SGD " << (decreasing ? "with" : "without") << " decreasing\n";
   for (int ii = 0; ii<min(budget, n*epochs); ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
         const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;
         if (verbose) cout << "Iteration " << ii << "(" << n_oracle_calls << ")" << " -- eta " << etak  << " -- obj " <<  loss << endl;
         logs[curr_loss_slider]=loss;
         ticks[curr_loss_slider++]=n_oracle_calls;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,yk,grad,lambda,dropout,param_loss); n_oracle_calls++;

      const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;

      wold.copy(w);
      w.copy(yk);
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);

      const T etakp1 = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(4.0)/(lambda*T(ii+3)*T(ii+3))) : eta;
      const T deltak=sqrt(lambda*etak);
      const T deltakp1=sqrt(lambda*etakp1);
      const T betak=deltak*(1-deltak)*etakp1/(etak*deltakp1+ etakp1*deltak*deltak);
      yk.copy(w);
      wold.add(w,-T(1.0));
      yk.add(wold,-betak);
   }
}

// ready for the final run
template <typename T>
void acc_sgd_batch(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int budget, const int epochs, const int init_epochs, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const int mb = 1, const T l1_scale = 0, const int param_loss = 1, const T kappa = 0, const Vector<T>& p_k = {}, const bool verbose=false) {
   const int n = y.n(); const int p = X.m();
   //   const T eta= (T(1.0)/(L))*sqrt(lambda/L);
   const T eta= (T(1.0) / L);
   // logs.resize(epochs);
   Vector<T> yk, wold, grad, grad2, col;
   yk.copy(w);

   /*
   Loss computation variables:
   */
   int curr_loss_slider = 0;
   int n_oracle_calls = 0;
   logs[0] = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
   ticks[0] = 0;

   if (verbose) cout << "Acc SGD with mini-batches and " << (decreasing ? "with" : "without") << " decreasing\n";
   const int num_iter= (n*epochs)/mb;
   const int freq_epoch= n/mb;
   for (int ii = 0; ii<min(budget, n*epochs); ++ii) {
      if (ii*mb/(eval_freq*n) >= curr_loss_slider) {
         const T loss = compute_loss(y,X,w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
         const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;
         if (verbose) cout << "Iteration " << ii*mb << " -- eta " << etak << " - over " << T(4.0)/(lambda*T(ii+2)*T(ii+2)) << " -- obj " <<  loss << endl;
         // logs[ii/(eval_freq*n)]=loss;
         logs[curr_loss_slider]=loss;
         ticks[curr_loss_slider++]=n_oracle_calls;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,yk,grad,lambda,dropout,param_loss); n_oracle_calls++;
      // being wrapped by S-Catalyst, gradient computations do not change
      if (mb > 1) {
         for (int ii=2; ii<mb; ++ii) {
            const int ind = random() % n;
            X.refCol(ind,col);
            compute_grad(y[ind],col,yk,grad2,lambda,dropout,param_loss); n_oracle_calls++;
            grad.add(grad2);
         }
         grad.scal(T(1.0)/mb);
      }

      const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;

      wold.copy(w);
      w.copy(yk);
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);
      const T etakp1 = (decreasing && (ii/n >= init_epochs)) ? MIN(eta,T(4.0)/(lambda*T(ii+3)*T(ii+3))) : eta;
      const T deltak=sqrt(lambda*etak);
      const T deltakp1=sqrt(lambda*etakp1);
      const T betak=deltak*(1-deltak)*etakp1/(etak*deltakp1+ etakp1*deltak*deltak);
      yk.copy(w);
      wold.add(w,-T(1.0));
      yk.add(wold,-betak);
   }

}

// ready for the final run
template <typename T>
void saga_miso(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int budget, const int epochs, const int init_epochs, const bool averaging, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const T beta = 0, const T l1_scale = 0, const int param_loss = 1, const T kappa = 0, const Vector<T>& y_k = {}, const bool is_fair=false, const bool verbose=false, int& n_oracle_calls=0, const int ep_residual=0, const T decr_factor=T(1.0)) {
   const int n = y.n(); const int p = X.m();
   const T eta = is_fair ? decr_factor/(3*L) : decr_factor/(12*L);
   if (verbose && kappa==0) cout << (beta ? "MISO " : "SAGA ")  << (decreasing ? "with" : "without") << " decreasing; (";
   if (verbose && kappa==0) cout << (is_fair ? "" : "non-")  << "fair comparison)\n";

   /*  Loss computation variables: */
   int curr_loss_slider = 0;
   
   Vector<T> wav, grad_anchor, grad_entry, grad, grad2, col, miso_term;
   Matrix<T> table_grads;
   table_grads.resize(n,p);
   
   wav.copy(w);
   miso_term.copy(w); 
   miso_term.scal(-beta);

   // initialize gradients table 
   for (int ii = 0; ii<n; ++ii) {
      X.refCol(ii,col);
      compute_grad(y[ii],col,w,grad_entry,lambda-kappa,dropout,param_loss); 
      if (kappa) smoothify_grad(grad_entry,kappa,w,y_k);
      if (beta) grad_entry.add(miso_term);
      table_grads.setRow(ii,grad_entry);
   }
   table_grads.meanRow(grad_anchor);

   for (int ii = 0; ii<min(budget, n*epochs); ++ii) {
      if ((ii % (eval_freq*n)) == ep_residual) {
         const T loss = compute_loss(y,X,averaging ? wav : w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
         const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
         if (verbose) cout << "Iteration " << ii << "(" << n_oracle_calls << ")" << " - stepsize - " << etak << " - obj " <<  loss << endl;
         logs[curr_loss_slider]=loss;
         ticks[curr_loss_slider++]=n_oracle_calls;
      }
      // TODO add miso_term from here 
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad_entry,lambda-kappa,dropout,param_loss); n_oracle_calls++;
      if (kappa) smoothify_grad(grad_entry,kappa,w,y_k);
      table_grads.extractRow(ind, grad2);
      table_grads.setRow(ind, grad_entry);

      grad_entry.add(grad2,-T(1.0));
      grad.copy(grad_entry);
      grad.add(grad_anchor);

      const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);

      // grad_anchor update
      grad_entry.scal(T(1.0) / n);
      grad_anchor.add(grad_entry);
      if (averaging) {
         const T tau = MIN(lambda*etak,T(1.0)/(5*n));
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging) w.copy(wav);
}

// ready for the final run
template <typename T>
void random_svrg(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int budget, const int epochs, const int init_epochs, const bool averaging, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const T l1_scale = 0, const int param_loss = 1, const T kappa = 0, const Vector<T>& y_k = {}, const bool is_fair=false, const bool verbose=false, int& n_oracle_calls=0, const int ep_residual=0, const T decr_factor=T(1.0)) {
   const int n = y.n(); const int p = X.m();
   const T eta = is_fair ? decr_factor/(3*L) : decr_factor/(12*L);
   if (verbose && kappa==0) cout << "SVRG " << (decreasing ? "with" : "without") << " decreasing; (";
   if (verbose && kappa==0) cout << (is_fair ? "" : "non-")  << "fair comparison)\n";

   /* Loss computation variables: */
   int curr_loss_slider = 0;
   Vector<T> wav, anchor, grad_anchor, grad, grad2, col;
   wav.copy(w);
   anchor.copy(w);
   compute_fullgrad(y,X,anchor,grad_anchor,lambda-kappa,dropout,param_loss); 
   // NO NEED TO SMOOTHIFY ANCHOR GRAD, AS THE SMOOTH-TERM WILL DISAPPEAR

   for (int ii = 0; ii<min(budget, n*epochs); ++ii) {
      if ((ii % (eval_freq*n)) == ep_residual) {
         const T loss = compute_loss(y,X,averaging ? wav : w,lambda-kappa,dropout,eval_freq,param_loss,l1_scale);
         const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
         if (verbose) cout << "Iteration " << ii << "(" << n_oracle_calls << ")" << " - eta: " << etak << " - obj " <<  loss << endl;
         logs[curr_loss_slider]=loss;
         ticks[curr_loss_slider++]=n_oracle_calls;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda-kappa,dropout,param_loss);
      compute_grad(y[ind],col,anchor,grad2,lambda-kappa,dropout,param_loss); n_oracle_calls+=2;
      if (kappa) smoothify_grad(grad,kappa,w,y_k);
      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);

      const T etak = (decreasing && (ii/n >= init_epochs)) ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
      
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);
      if (random() % n == 0) {
      // if ((ii % n) == 0) {
         anchor.copy(w);
         compute_fullgrad(y,X,anchor,grad_anchor,lambda-kappa,dropout,param_loss); 
      }
      if (averaging) {
         const T tau = MIN(lambda*etak,T(1.0)/(5*n));
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging) w.copy(wav);
}


// ready for the final run
template <typename T>
void katyusha(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int budget, const int epochs, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const T l1_scale = 0, const int param_loss = 1, const bool verbose=false) {
   const int n = y.n(); const int p = X.m();
   const T eta = T(1.0) / (3*L);
   if (verbose) cout << "KATYUSHA\n";

   /* Loss computation variables: */
   int curr_loss_slider = 0, n_oracle_calls = 0;
   Vector<T> grad_anchor, grad, grad2, col;
   Vector<T> anchor, temp_anchor, y_k, x_kpone, z_k;
   anchor.copy(w); y_k.copy(w); z_k.copy(w); temp_anchor.copy(w);
   x_kpone.copy(w);
   const T tau_2 = T(0.5), tau_1 = min(sqrt(2 * n * lambda / (3 * L)), T(0.5));
   const T alpha = eta / tau_1;
   const T beta = T(1.0) + lambda * alpha;
   const T scal_value = (beta - T(1.0)) / (power(beta, 2 * n) - 1);
   T weight;

   for (int ii = 0; ii<min(budget, n*epochs); ++ii) {
      if ((ii % (2*n)) == 0) {
         anchor.copy(temp_anchor);
         compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss); 
         weight = T(1.0);
         temp_anchor.copy(y_k);
         temp_anchor.scal(weight * scal_value);
      }
      
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         if (verbose) cout << "Iteration " << ii << "(" << n_oracle_calls << ")" << " - obj " << loss << endl;

         logs[curr_loss_slider]=loss;
         ticks[curr_loss_slider++]=n_oracle_calls;
      }
      
      x_kpone.copy(anchor); x_kpone.scal(tau_2);
      x_kpone.add(z_k, tau_1);
      x_kpone.add(y_k, T(1.0) - tau_1 - tau_2);

      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,x_kpone,grad,lambda,dropout,param_loss);
      compute_grad(y[ind],col,anchor,grad2,lambda,dropout,param_loss); n_oracle_calls+=2;
      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);

      y_k.copy(x_kpone);
      // y_k.add(grad,-eta);
      y_k.add(z_k,-tau_1);
      z_k.add(grad,-alpha);
      y_k.add(z_k,tau_1);

      weight *= beta;
      temp_anchor.add( y_k, weight * scal_value );
      w.copy(x_kpone);
      // if (l1_scale) lasso_prox(w, etak * l1_scale);
   }
}

template <typename T>
void catalyst(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const T kappa, const int budget, int local_epochs, const int k0, const int n_stages, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, Vector<T>& ticks, const T l1_scale = 0, const int param_loss = 1, const int workhorse_method = 1, const bool is_fair=false, const bool verbose=false, const bool method_verbose=false, const bool is_warm_yk=true) {
   // catalyst(y,X,w,L,lambda,epochs,decreasing,dropout,eval_freq,logs,l1_scale);
   // workhorse_method:
   //    1: SVRG
   //    2: SAGA
   
   if (verbose) cout << "Catalyst over " << (workhorse_method == 1 ? "SVRG " : "SAGA ") << (decreasing ? "with" : "without") << " decreasing" << endl;
   if (verbose) cout << "epochs per stage: " << local_epochs << endl;
   // initialization
   const int n = y.n();
   int log_position = 0, n_oracle_calls = 0, local_epochs_0 = local_epochs;

   T q = lambda / (lambda + kappa);
   T alpha = sqrt(q), beta = (1 - sqrt(q)) / (1 + sqrt(q)); // let us keep them constant for simplicity (only case lambda > 0) (may look in backup)

   Vector<T> y_k, warm_start, x_kmone;
   y_k.copy(w); x_kmone.copy(w);

   // cycle over stages
   for (int ii = 0; ii < n_stages && budget > n_oracle_calls; ++ii) {
      const T decr_factor = (ii >= k0 && decreasing) ? min(T(1.0), 10 * power(T(1.0)-sqrt(q)/2,ii - k0)) : T(1.0);
      local_epochs = int(local_epochs_0 / decr_factor);

      // this variable is to correctly deal with eval_freq non-multiple to local_epochs (we define an auxiliary residual)
      int excessive = (ii * n * local_epochs) % (n * eval_freq); // excessive <= n * eval_freq
      int ep_residual = (eval_freq * n - excessive) % (eval_freq * n);
      // imitate flow of inner method to calculate number of function calculations (too hard to find a precise formula)
      // (only due to the case eval_freq>1)
      int local_logs_size = 0;
      for (int jj = 0; jj<min(budget-n_oracle_calls, n*local_epochs); ++jj) {
         if ((jj % (eval_freq*n)) == ep_residual) local_logs_size++;
      }

      // now, we can proceed with a launch of our inner method
      Vector<T> sublogs(local_logs_size), subticks(local_logs_size);

      if (is_warm_yk) warm_start.copy(y_k);
      else warm_start.copy(x_kmone);

      if (workhorse_method == 1) {
         random_svrg(y,X,warm_start,L+kappa,lambda+kappa,budget-n_oracle_calls,local_epochs,0,false,false,
                     dropout,eval_freq,sublogs,subticks,l1_scale,param_loss,
                     kappa,y_k,is_fair,method_verbose,n_oracle_calls,ep_residual,decr_factor);
      } else {
         saga_miso(y,X,warm_start,L+kappa,lambda+kappa,budget-n_oracle_calls,local_epochs,0,false,false,
                     dropout,eval_freq,sublogs,subticks,l1_scale,T(0.0),param_loss,
                     kappa,y_k,is_fair,method_verbose,n_oracle_calls,ep_residual,decr_factor);
      }
      w.copy(warm_start); // warm_start contains the final solution to the stage

      for (int jj = 0; jj < local_logs_size; ++jj) {
         if (verbose) cout << "Stage " << ii << " - iter " << jj  << "("  << subticks[jj] << ") - eta - " << decr_factor << " - (iter: " << eval_freq * n * log_position << ") - obj " <<  sublogs[jj] << endl;
         logs[log_position] = sublogs[jj];
         ticks[log_position++] = subticks[jj];
      }

      y_k.copy(w);
      y_k.scal(beta + T(1.0));
      y_k.add(x_kmone, T(-1.0) * beta);

      x_kmone.copy(w);
   }

   return;
}

#endif
