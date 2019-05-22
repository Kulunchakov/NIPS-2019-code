path_mkl='/scratch2/clear/mairal/intel/compilers_and_libraries/linux/mkl/lib/intel64/';
include_mkl='/scratch2/clear/mairal/intel/mkl/include/';
pathlibiomp='/scratch2/clear/mairal/intel/compilers_and_libraries_2017/linux/lib/intel64/';
path_icc='/scratch2/clear/mairal/intel/compilers_and_libraries/linux/';
path_cuda='/scratch2/clear/mairal/cuda-9.0/';
path_matlab='/softs/stow/matlab-2016a/bin/';
path_libstd='/usr/lib/gcc/x86_64-linux-gnu/5/';
debug=false;
cuda8=false; % for cuda 8 and above

%%%%%% list of mex files %%%%
%names_mklst={'mex_svm_miso','mex_permutation','mex_create_dataset','mex_centering','mex_normalize','mex_encode_cpu'};
%names_mklmt={'mex_exp','mex_eig','mex_kmeans'};
names_mklmt={}
names_mklst={'mex_svm_svrg','mex_normalize'};
%names_mklst={'mex_encode_cpu','mex_create_dataset'}
%names_cuda={'mex_encode_cudnn','mex_train_ckn_cudnn'};
names_cuda={};

%%%%%% various flags %%%%%
format compact;
compiler_icc=[path_icc '/bin/intel64/icpc'];
lib_mkl_sequential=sprintf('-Wl,--start-group %slibmkl_intel_ilp64.a %slibmkl_sequential.a %slibmkl_core.a -Wl,--end-group',path_mkl,path_mkl,path_mkl);
lib_mkl_mt=sprintf('-Wl,--start-group %slibmkl_intel_ilp64.a %slibmkl_intel_thread.a %slibmkl_core.a -Wl,--end-group -L%s -liomp5 -ldl',path_mkl,path_mkl,path_mkl,pathlibiomp);
lib_openmp='-liomp5';
defines='-DTIMINGS -DNDEBUG -DHAVE_MKL -DINT_64BITS -DAXPBY';
defines='-DNDEBUG -DHAVE_MKL -DINT_64BITS -DAXPBY';
if cuda8
   defines=[defines ' -DCUDA_8'];
end
flags='-O3 -fopenmp -static-intel -fno-alias -align -falign-functions';
lflags='';
includes=sprintf('-I./utils/ -I%s',include_mkl);

fid=fopen('run_matlab.sh','w+');
fprintf(fid,'#!/bin/sh\n');
fprintf(fid,sprintf('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s:%s:%s\n',[path_icc 'lib/intel64/'],path_mkl,[path_cuda 'lib64/']));
fprintf(fid,sprintf('export LIB_INTEL=%s\n',[path_icc 'lib/intel64/']));
fprintf(fid,'export KMP_AFFINITY=verbose,granularity=fine,compact,1,0\n');fprintf(fid,'export LD_PRELOAD=$LIB_INTEL/libimf.so:$LIB_INTEL/libintlc.so.5:$LIB_INTEL/libiomp5.so:$LIB_INTEL/libsvml.so:%s/libstdc++.so\n',path_libstd);
fprintf(fid,[path_matlab 'matlab -nodisplay -singleCompThread -r \"addpath(''./mex/''); "\n']); 
fclose(fid);
!chmod +x run_matlab.sh

for ii=1:length(names_mklmt)
   name=names_mklmt{ii};
   name
   str=sprintf(' -v -largeArrayDims CXX="%s" DEFINES="\\$DEFINES %s" CXXFLAGS="\\$CXXFLAGS %s" INCLUDE="\\$INCLUDE %s" LDFLAGS="\\$LDFLAGS " LINKLIBS="\\$LINKLIBS -L"%s" %s %s" mex/%s.cpp -output mex/%s.mexa64',compiler_icc,defines,flags,includes,path_mkl,lib_mkl_mt,lib_openmp,name,name);
   args = regexp(str, '\s+', 'split');
   args = args(find(~cellfun(@isempty, args)));
   mex(args{:});
end

for ii=1:length(names_mklst)
   name=names_mklst{ii};
   name
   str=sprintf(' -v -largeArrayDims CXX="%s" DEFINES="\\$DEFINES %s" CXXFLAGS="\\$CXXFLAGS %s" LDFLAGS="\\$LDFLAGS " INCLUDE="\\$INCLUDE %s" LINKLIBS="\\$LINKLIBS -L"%s" %s %s" mex/%s.cpp -output mex/%s.mexa64',compiler_icc,defines,flags,includes,path_mkl,lib_mkl_mt,lib_openmp,name,name);
   args = regexp(str, '\s+', 'split');
   args = args(find(~cellfun(@isempty, args)));
   mex(args{:});
end

%% creates mex_optimize_sgd_cuda
includes=[includes ' -I' path_cuda '/include/'];
system(sprintf('%s/bin/nvcc -c utils/cuda_kernels.cu -Xcompiler -fpic',path_cuda));
system('mv cuda_kernels.o mex/');
for ii=1:length(names_cuda)
   name=names_cuda{ii};
   name
   str=sprintf(' -v -largeArrayDims CXX="%s" DEFINES="\\$DEFINES %s -DCUDA -DCUDNN" CXXFLAGS="\\$CXXFLAGS %s" INCLUDE="\\$INCLUDE %s" LDFLAGS="\\$LDFLAGS %s" LINKLIBS="\\$LINKLIBS -L"%s" -L"%s" %s -lcudart -lcublas -lcusolver -lcudnn %s mex/cuda_kernels.o" mex/%s.cpp -output mex/%s.mexa64',compiler_icc,defines,flags,includes,lflags,path_mkl,[path_cuda '/lib64/'],lib_mkl_mt,lib_openmp,name,name);
   args = regexp(str, '\s+', 'split');
   args = args(find(~cellfun(@isempty, args)));
   mex(args{:});
end
