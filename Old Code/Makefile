#makefile
a.out: ex_particle.c
	gcc -lm ex_particle.c

opt: ex_particle.c
	gcc -lm -O3 -ffast-math ex_particle.c -o opt.out

time: ex_particle_OPENMP_seq.c
	gcc -lm -O3 -ffast-math ex_particle_OPENMP_seq.c -o c_time.out

openmp: ex_particle_OPENMP_seq.c
	gcc -lm -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o openmp.out

icc_time: ex_particle_OPENMP_seq.c
	icc -O3 -fast ex_particle_OPENMP_seq.c -o c_time_icc.out

icc_openmp: ex_particle_OPENMP_seq.c
	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o openmp_icc.out
	
debug: ex_particle.c
	gcc -lm -g -Wall ex_particle.c -o debug.out
	
cuda: ex_particle_CUDA_seq.cu
	/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -g -lm -O3 -use_fast_math -arch sm_13 ex_particle_CUDA_seq.cu -o cuda.out

debug_cuda: ex_particle_CUDA_seq.cu
	/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -g -deviceemu -lm -arch sm_13 ex_particle_CUDA_seq.cu -o cuda.out

naive: ex_particle_CUDA_naive_seq.cu
	/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -g -lm -O3 -use_fast_math -arch sm_13 ex_particle_CUDA_naive_seq.cu -o naive.out
	
float: ex_particle_CUDA_float_seq.cu
	/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -g -lm -O3 -use_fast_math -arch sm_13 ex_particle_CUDA_float_seq.cu -o float.out

mex: ex_particle_OPENMP.c
	 gcc  -c -I/net/af12/mjt5v/Thesis -L/common/matlab/7.8/bin/glnxa64 -O3 -ffast-math ex_particle_OPENMP.c 
	 
thrust: ex_particle_THRUST.cu
	/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -g -lm -O3 -use_fast_math -arch sm_13 ex_particle_THRUST.cu -o thrust.out

clean:
	rm *.out
