function [] = main(data,dropout,lambda_factor,seed,nepochs_factor,init_epochs_factor,setting,loss,verbose)

if isdeployed || (strcmp(class(data),'char'))
   data=str2num(data);
   dropout=str2num(dropout);
   lambda_factor=str2num(lambda_factor);
   seed=str2num(seed);
   nepochs_factor=str2num(nepochs_factor);
   init_epochs_factor=str2num(init_epochs_factor);
   setting=str2num(setting);
   loss=str2num(loss);
   verbose=str2num(verbose);
end


name=sprintf('exps/exp_data%d_s%d_d%d_l%d_n%d_%d_seed%d_loss%d.mat',data,setting,dropout * 100,lambda_factor,300,5,seed,loss);
name
%{
if exist(name, 'file')==1
    return
end
%}

if data==1
%%%%% Dataset 1 - CKN %%%%%%
   load('/scratch2/clear/mairal/ckn_matrix.mat');
   X=psiTr;
   n=size(X,2);
   y=-ones(n,1);
   y(find(Ytr==0))=1;
elseif data==2
   %%%%% Dataset 2 - gene100 %%%%%%
   load('/scratch/clear/abietti/data/vant.mat');
   X=X';
   mex_normalize(X);
   y=Y(:,2);
elseif data==3
   load('/scratch/clear/mairal/large_datasets/alpha.full_norm.mat');
   y=y(1:250000);
   X=X(:,1:250000);
   mex_normalize(X);
end

n=size(X,2);
param.lambda=1/(lambda_factor*n);  %% This is the regularization parameter
X=double(X);
y=double(y);
param.threads=1;
param.dropout=dropout;

if (dropout==0)
   param.eval_freq=1;
else
   param.eval_freq=5;
end

if loss==0
   L=0.25;
else 
   L=1;
end
param.L = L;
%% epochs_per_stage = ceil(max(log(L * n / param.lambda), 1));

epochs_per_stage = 1;
nepochs = epochs_per_stage * nepochs_factor;
init_epochs = epochs_per_stage * init_epochs_factor;
param.seed=seed;
param.loss=loss;
param.decreasing=false;
param.averaging=false; % seems to always hurt
param.verbose=verbose;
param.method_verbose = false;
param.epochs=nepochs;
param.is_warm_yk=true;
param.is_fair=true;
param.epochs_per_stage = epochs_per_stage;

factor_ = 5;
param.kappa = param.L / (factor_ * n);
fprintf('kappa factor: %.6f\n', param.kappa);

budget_factor = 400;
cat_k0 = 25;
param.budget = n * budget_factor;
fprintf('total budget: %d\n', param.budget);


%% initial approximation
%% w0 = 10 * ones(size(X,1),1);
w0 = zeros(size(X,1),1);
w = w0;

tic
if setting==1
   %%%% Exp 1 - catalyst over SVRG
   param.do_catalyst=true;
   param.workhorse=1;
   param.epochs=epochs_per_stage;
   param.n_stages = nepochs_factor;  %% ceil(nepochs / epochs_per_stage);
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==2
   %%%% Exp 2 - SAGA with 1/3L (inserted to the 'is_fair' flag)
   param.do_saga=true;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==3
   %%%% Exp 3 - accelerated SVRG - constant step size
   param.do_acc_svrg=true;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);
   logs_exp
   ticks
elseif setting==4
   %%%% Exp 4 - SGD with 1/L
   param.do_sgd=true;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==5
   %%%% Exp 5 - SVRG with 1/3L (inserted to the 'is_fair' flag)
   param.do_svrg=true;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==6
   %%%% Exp 6 - catalyst over SAGA
   param.do_catalyst=true;
   param.workhorse=2;
   param.epochs=epochs_per_stage;
   param.n_stages = nepochs_factor;  %% ceil(nepochs / epochs_per_stage);

   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==7
   %%%% Exp 7 - catalyst over SVRG (with decreasing stepsizes)
   param.do_catalyst=true;
   param.workhorse=1;
   param.decreasing=true;
   param.epochs=epochs_per_stage;
   param.k0 = cat_k0;
   param.n_stages = nepochs_factor;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param); % constant step size regime for init_epochs epochs
   
elseif setting==8
   %%%% Exp 8 - SAGA with decreasing stepsizes
   param.do_saga=true;
   param.epochs = nepochs;
   param.init_epochs = init_epochs;
   param.decreasing=true;

   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==9
   %%%% Exp 9 - accelerated SVRG  (with decreasing stepsizes)
   param.do_acc_svrg=true;
   param.epochs = nepochs;
   param.init_epochs = init_epochs;
   param.decreasing=true;

   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);
   logs_exp
   ticks

elseif setting==10
   %%%% Exp 10 - SGD with 1/L (with decreasing stepsizes)
   param.do_sgd=true;
   param.epochs = nepochs;
   param.init_epochs = init_epochs;
   param.decreasing=true;

   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==11
   %%%% Exp 11 - SVRG with 1/3L  (with decreasing stepsizes)
   param.do_svrg=true;
   param.epochs = nepochs;
   param.init_epochs = init_epochs;
   param.decreasing=true;

   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==12
   %%%% Exp 12 - catalyst over SAGA (with decreasing stepsizes)
   param.do_catalyst=true;
   param.workhorse=2;
   param.decreasing=true;
   param.epochs=epochs_per_stage;
   param.k0 = cat_k0;
   param.n_stages = nepochs_factor;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param); % constant step size regime for init_epochs epochs

elseif setting==13
   %%%% Exp 13 - ??? todo
   param.do_svrg=true;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==14
   %%%% Exp 14 - accelerated SGD
   param.do_acc_sgd=true;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

elseif setting==15
   %%%% Exp 10 - accelerated SGD (with decreasing stepsizes)
   param.do_acc_sgd=true;
   param.epochs = init_epochs;
   [w logs_expa ticksa]=mex_svm_svrg(y,X,w0,param); % constant step size regime for init_epochs epochs

   param.decreasing=true;
   param.epochs = nepochs - init_epochs;
   [w logs_expb ticksb]=mex_svm_svrg(y,X,w,param);
   
   logs_exp=[logs_expa' logs_expb']';
   ticks=[ticksa' ticksb']';
   %% logs_exp=logs_expb;
elseif setting == 16
   %%%% Exp 16 - minibatch-acc-SGD (with decreasing stepsizes)
   param.do_acc_sgd=true;
   param.decreasing=true;
   param.minibatch=round(sqrt(param.L/param.lambda));
   param.epochs=nepochs;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);
elseif setting == 17
   %%%% Exp 17 - minibatch-acc-SGD (with decreasing stepsizes)
   param.do_katyusha=true;
   param.epochs=nepochs;
   [w logs_exp ticks]=mex_svm_svrg(y,X,w0,param);

end

toc

save(name,'logs_exp','w','ticks');
