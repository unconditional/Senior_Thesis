#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>

#include "mex.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "driver_types.h"

#define BLOCK_X 16
#define BLOCK_Y 16
//3D matrix consisting the picture and the frames
unsigned char * I;
//dimension X of the picture in pixels
int IszX = 128;
//dimension Y of the picture in pixels
int IszY = 128;
//number of frames
int Nfr = 10;
//define number of particles
int Nparticles = 10;


texture <float> tex_CDF;


const int threads_per_block = 512;

/*****************************
*GET_TIME
*returns a long int representing the time
*****************************/
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
        return (float) (end_time - start_time) / (1000 * 1000);
}
/*****************************
* CHECK_ERROR
* Checks for CUDA errors and prints them to the screen to help with
* debugging of CUDA related programming
*****************************/
void check_error(cudaError e) {
     if (e != cudaSuccess) {
     	printf("\nCUDA error: %s\n", cudaGetErrorString(e));
	    exit(1);
     }
}
/********************************
* CALC LIKELIHOOD SUM
* DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* param 1 I 3D matrix
* param 2 current ind array
* param 3 length of ind array
* returns a float representing the sum
********************************/
__device__ float calcLikelihoodSum(unsigned char * I, int * ind, int numOnes, int index){
	float likelihoodSum = 0.0;
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((float)(I[ind[index*numOnes + x]] - 100),2) - pow((float)(I[ind[index*numOnes + x]]-228),2))/50.0;
	return likelihoodSum;
}
/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
*****************************/
__device__ void cdfCalc(float * CDF, float * weights, int Nparticles){
	int x;
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
}
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a float representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
__device__ float d_randu(int * seed, int index)
{
	
	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	num = seed[index];
	return fabs(num/((float) M));
}
float randu()
{
	float max = (float)RAND_MAX;
	int num = rand();
	return num/max;
}
/******************************
* RANDN
* GENERATES A NORMAL DISTRIBUTION
* returns a float representing random number generated using Irwin-Hall distribution method
* see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
******************************/
float randn(){
	//Box-Muller algortihm
	float u1, u2, v1, v2;
	float s = 2;
	while(s >= 1){
		u1 = randu();
		u2 = randu();
		v1 = 2.0*u1 - 1.0;
		v2 = 2.0*u2 - 1.0;
		s = pow(v1, 2)+pow(v2, 2);
	}
	float x1 = v1*sqrt((-2.0*log(s))/s);
	return x1;
}
__device__ float d_randn(int * seed, int index){
	//Box-Muller algortihm
	float pi = 3.14159265358979323846;
	float u = d_randu(seed, index);
	float v = d_randu(seed, index);
	float cosine = cos(2*pi*v);
	float rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparcitles
****************************/
__device__ float updateWeights(float * weights, float * likelihood, int Nparticles){
	int x;
	float sum = 0;
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x] * exp(likelihood[x]);
		sum += weights[x];
	}		
	return sum;
}
__device__ int findIndexBin(float * CDF, int beginIndex, int endIndex, float value)
{
	if(endIndex < beginIndex)
		return -1;
	int middleIndex;
	while(endIndex > beginIndex)
	{
		middleIndex = beginIndex + ((endIndex-beginIndex)/2);
		if(CDF[middleIndex] >= value)
		{
			if(middleIndex == 0)
				return middleIndex;
			else if(CDF[middleIndex-1] < value)
				return middleIndex;
			else if(CDF[middleIndex-1] == value)
			{
				while(CDF[middleIndex] == value && middleIndex >= 0)
					middleIndex--;
				middleIndex++;
				return middleIndex;
			}
		}
		if(CDF[middleIndex] > value)
			endIndex = middleIndex-1;
		else
			beginIndex = middleIndex+1;
	}
	return -1;
}
/*****************************
* CUDA Find Index Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: weights
* param8: Nparticles
*****************************/
__global__ void find_index_kernel(float * arrayX, float * arrayY, float * CDF, float * u, float * xj, float * yj, float * weights, int Nparticles, float * x_partial_sums, float * y_partial_sums, int k, float * x_loc, float * y_loc){
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;
	x_loc[k] = x_partial_sums[0];
	y_loc[k] = y_partial_sums[0];
	if(i < Nparticles){
	
		int index = -1;
		int x;

		for(x = 0; x < Nparticles; x++){
			if(tex1Dfetch(tex_CDF, x) >= u[i]){
				index = x;
				break;
			}
		}
		if(index == -1){
			index = Nparticles-1;
		}
		
		xj[i] = arrayX[index];
		yj[i] = arrayY[index];
		
		weights[i] = 1/((float)(Nparticles));
		
	}
	__syncthreads();
}
__global__ void normalize_weights_kernel(float * weights, int Nparticles, float * partial_sums, float * CDF, float * u, int * seed, float * x_partial_sums, float * y_partial_sums, float * arrayX, float * arrayY)
{
	int block_id = blockIdx.x;
	int i = blockDim.x*block_id + threadIdx.x;
	__shared__ float u1, sumWeights;
	__shared__ float xbuffer[512];
	__shared__ float ybuffer[512];
	sumWeights = partial_sums[0];
	if(i < Nparticles)
	{
		weights[i] = weights[i]/sumWeights;
	}
	if(i == 0)
	{
		cdfCalc(CDF, weights, Nparticles);
		u1 = (1/((float)(Nparticles)))*d_randu(seed, i);
	}
	if(i < Nparticles)
	{
		__syncthreads();
		u[i] = u1 + i/((float)(Nparticles));
		xbuffer[threadIdx.x] = weights[i]*arrayX[i];
		ybuffer[threadIdx.x] = weights[i]*arrayY[i];
		__syncthreads();
		for(unsigned int s=blockDim.x/2; s>0; s>>=1)
		{
			if(threadIdx.x < s)
			{
				xbuffer[threadIdx.x] += xbuffer[threadIdx.x + s];
				ybuffer[threadIdx.x] += ybuffer[threadIdx.x + s];
			}
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			x_partial_sums[blockIdx.x] = xbuffer[0];
			y_partial_sums[blockIdx.x] = ybuffer[0];
		}
	}
}
__global__ void sum_kernel(float* partial_sums, int Nparticles)
{
	int block_id = blockIdx.x;
	int i = blockDim.x*block_id + threadIdx.x;
	
	if(i == 0)
	{
		int x;
		float sum = 0;
		for(x = 0; x < Nparticles/512; x++)
		{
			sum += partial_sums[x];
		}
		partial_sums[0] = sum;
	}
}
__global__ void sum_xy_kernel(float * x_partial_sums, float * y_partial_sums, int Nparticles)
{
	int block_id = blockIdx.x;
	int i = blockDim.x*block_id + threadIdx.x;
	if(i == 0)
	{
		int x;
		float x_sum = 0;
		float y_sum = 0;
		for(x = 0; x < Nparticles/512; x++)
		{
			x_sum += x_partial_sums[x];
			y_sum += y_partial_sums[x];
		}
		x_partial_sums[0] = x_sum;
		y_partial_sums[0] = y_sum;
	}
}
/*****************************
* CUDA Likelihood Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param2.5: CDF
* param3: ind
* param4: objxy
* param5: likelihood
* param6: I
* param6.5: u
* param6.75: weights
* param7: Nparticles
* param8: countOnes
* param9: max_size
* param10: k
* param11: IszY
* param12: Nfr
*****************************/
__global__ void likelihood_kernel(float * arrayX, float * arrayY, float * CDF, int * ind, int * objxy, float * likelihood, unsigned char * I, float * u, float * weights, int Nparticles, int countOnes, int max_size, int k, int IszX, int IszY, int Nfr, int *seed, float * partial_sums){
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;
	int y;
	float indX, indY;
	__shared__ float buffer[512];
	if(i < Nparticles){
		arrayX[i] = arrayX[i] + 1.0 + 5.0*d_randn(seed, i);
		arrayY[i] = arrayY[i] - 2.0 + 2.0*d_randn(seed, i);
		__syncthreads();
	}
	if(i < Nparticles)
	{
		for(y = 0; y < countOnes; y++){
			indX = round(arrayX[i]) + objxy[y*2 + 1];
			indY = round(arrayY[i]) + objxy[y*2];
			ind[i*countOnes + y] = fabs(k*IszY*IszX + indX*IszX + indY);
			if(ind[i*countOnes + y] >= max_size)
				ind[i*countOnes + y] = 0;
		}
		likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);
		likelihood[i] = likelihood[i]/countOnes;
		
		__syncthreads();
	}
	
	if(i < Nparticles)
	{
		weights[i] = weights[i]*likelihood[i];
		__syncthreads();
		buffer[threadIdx.x] = weights[i];
		__syncthreads();
		
		for(unsigned int s=blockDim.x/2; s>0; s>>=1)
		{
			if(threadIdx.x < s)
			{
				buffer[threadIdx.x] += buffer[threadIdx.x + s];
			}
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			partial_sums[blockIdx.x] = buffer[0];
		}
		__syncthreads();
	}	
}
__global__ void likelihood_kernel1F(float * arrayX, float * arrayY, float * CDF, int * ind, int * objxy, float * likelihood, unsigned char * I, float * u, float * weights, int Nparticles, int countOnes, int max_size, int IszX, int IszY, int *seed, float * partial_sums){
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;
	int y;
	float indX, indY;
	__shared__ float buffer[512];
	if(i < Nparticles){
		arrayX[i] = arrayX[i] + 1.0 + 5.0*d_randn(seed, i);
		arrayY[i] = arrayY[i] - 2.0 + 2.0*d_randn(seed, i);
		__syncthreads();
	}
	if(i < Nparticles)
	{
		for(y = 0; y < countOnes; y++){
			indX = round(arrayX[i]) + objxy[y*2 + 1];
			indY = round(arrayY[i]) + objxy[y*2];
			ind[i*countOnes + y] = fabs(indX*IszX + indY);
			if(ind[i*countOnes + y] >= max_size)
				ind[i*countOnes + y] = 0;
		}
		likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);
		likelihood[i] = likelihood[i]/countOnes;
		
		__syncthreads();
	}
	
	if(i < Nparticles)
	{
		weights[i] = weights[i]*likelihood[i];
		__syncthreads();
		buffer[threadIdx.x] = weights[i];
		__syncthreads();
		
		for(unsigned int s=blockDim.x/2; s>0; s>>=1)
		{
			if(threadIdx.x < s)
			{
				buffer[threadIdx.x] += buffer[threadIdx.x + s];
			}
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			partial_sums[blockIdx.x] = buffer[0];
		}
		__syncthreads();
	}	
}
float roundDouble(float value){
	int newValue = (int)(value);
	if(value - newValue < .5)
		return newValue;
	else
		return newValue++;
}

/******************************
* SETIF
* set values of the 3D array to a newValue if that value is equal to the testValue
* param1: value to test
* param2: 3D array
* param3: dim X
* param4: dim Y
* param5: dim Z
******************************/
void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ){
	int x, y, z;
	for(x = 0; x < *dimX; x++){
		for(y = 0; y < *dimY; y++){
			for(z = 0; z < *dimZ; z++){
				if(array3D[x * *dimY * *dimZ+y * *dimZ + z] == testValue)
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
			}
		}
	}
}
/******************************
* ADDNOISE
* sets values of 3D matrix using randomly generated numbers from a normal distribution
* param matrix
******************************/
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ){
	int x, y, z;
	for(x = 0; x < *dimX; x++){
		for(y = 0; y < *dimY; y++){
			for(z = 0; z < *dimZ; z++){
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int)(5*randn());
			}
		}
	}
}
/******************************
* STRELDISK
* param: pointer to the disk to be made 
* creates a 9x9 matrix representing the disk
******************************/
void strelDisk(int * disk, int radius)
{
	int diameter = radius*2 - 1;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			double distance = sqrt(pow((double)(x-radius+1),2) + pow((double)(y-radius+1),2));
			if(distance < radius)
			disk[x*diameter + y] = 1;
		}
	}
}
/******************************
* DILATE_MATRIX
* param1: matrix to be dilated
* param2: current x position
* param3: current y position
* param4: current z position
* param5: x length
* param6: y length
* param7: z length
* param8: error radius
*******************************/
void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
{
	int startX = posX - error;
	while(startX < 0)
	startX++;
	int startY = posY - error;
	while(startY < 0)
	startY++;
	/*int startZ = posZ - error;
	while(startZ < 0)
		startZ++;*/
	int endX = posX + error;
	while(endX > dimX)
	endX--;
	int endY = posY + error;
	while(endY > dimY)
	endY--;
	/*int endZ = posZ + error;
	while(endZ > dimZ)
		endZ--;*/
	int x,y;
	for(x = startX; x < endX; x++){
		for(y = startY; y < endY; y++){
			double distance = sqrt( pow((double)(x-posX),2) + pow((double)(y-posY),2) );
			if(distance < error)
			matrix[x*dimY*dimZ + y*dimZ + posZ] = 1;
		}
	}
}

/******************************
* IMDILATE_DISK
* param1: target 3d matrix
* param2: dimX
* param3: dimY
* param4: dimZ
* param5: radius
* param6: error
* returns the dilated matrix
* dilates the target matrix using the radius as a guide
******************************/
void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix)
{
	int x, y, z;
	for(z = 0; z < dimZ; z++){
		for(x = 0; x < dimX; x++){
			for(y = 0; y < dimY; y++){
				if(matrix[x*dimY*dimZ + y*dimZ + z] == 1){
					dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
				}
			}
		}
	}
}
/*****************************
* GET NEIGHBORS
* returns a 2D array describing the offets
* param 1 strel object
* param 2 dimX of object
* param 3 dimY of object
*******************************/
void getneighbors(int * se, int numOnes, int * neighbors, int radius){
	int x, y;
	int neighY = 0;
	int center = radius - 1;
	int diameter = radius*2 -1;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(se[x*diameter + y]){
				neighbors[neighY*2] = (int)(y - center);
				neighbors[neighY*2 + 1] = (int)(x - center);
				neighY++;
			}
		}
	}
}
/******************************
* VIDEO SEQUENCE
* the synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
*******************************/
void videoSequence(int * I, int IszX, int IszY, int Nfr){
	int k;
	int max_size = IszX*IszY*Nfr;
	/*get object centers*/
	int x0 = (int)roundDouble(IszY/2.0);
	int y0 = (int)roundDouble(IszX/2.0);
	I[x0 *IszY *Nfr + y0 * Nfr  + 0] = 1;
	
	/*move point*/
	int xk, yk, pos;
	for(k = 1; k < Nfr; k++){
		xk = abs(x0 + (k-1));
		yk = abs(y0 - 2*(k-1));
		pos = yk * IszY * Nfr + xk *Nfr + k;
		if(pos >= max_size)
		pos = 0;
		I[pos] = 1;
	}
	
	/*dilate matrix*/
	int * newMatrix = (int *)mxCalloc(IszX*IszY*Nfr, sizeof(int));
	imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
	int x, y;
	for(x = 0; x < IszX; x++){
		for(y = 0; y < IszY; y++){
			for(k = 0; k < Nfr; k++){
				I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
			}
		}
	}
	mxFree(newMatrix);
	
	/*define background, add noise*/
	setIf(0, 100, I, &IszX, &IszY, &Nfr);
	setIf(1, 228, I, &IszX, &IszY, &Nfr);
	/*add noise*/
	addNoise(I, &IszX, &IszY, &Nfr);
}

void particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, float * x_loc, float * y_loc){
	int max_size = IszX*IszY*Nfr;
	//original particle centroid
	float xe = roundDouble(IszY/2.0);
	float ye = roundDouble(IszX/2.0);
	x_loc[0] = xe;
	y_loc[0] = ye;
	/*expected object locations, compared to center*/
	int radius = 5;
	int diameter = radius*2 -1;
	int * disk = (int *)mxCalloc(diameter*diameter, sizeof(int));
	strelDisk(disk, radius);
	int countOnes = 0;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(disk[x*diameter + y] == 1)
			countOnes++;
		}
	}
	int * objxy = (int *)mxCalloc(countOnes*2, sizeof(int));
	getneighbors(disk, countOnes, objxy, radius);
	
	
	//initial weights are all equal (1/Nparticles)
	float * weights = (float *)mxCalloc(Nparticles,sizeof(float));
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((float)(Nparticles));
	}
	
	//initial likelihood to 0.0
	float * likelihood = (float *)mxCalloc(Nparticles, sizeof(float));
	float * arrayX = (float *)mxCalloc(Nparticles, sizeof(float));
	float * arrayY = (float *)mxCalloc(Nparticles, sizeof(float));
	float * xj = (float *)mxCalloc(Nparticles, sizeof(float));
	float * yj = (float *)mxCalloc(Nparticles, sizeof(float));
	float * CDF = (float *)mxCalloc(Nparticles, sizeof(float));
	
	//GPU copies of arrays
	float * arrayX_GPU;
	float * arrayY_GPU;
	float * xj_GPU;
	float * yj_GPU;
	float * CDF_GPU;
	float * likelihood_GPU;
	unsigned char * I_GPU;
	float * weights_GPU;
	int * objxy_GPU;
	float * xloc_GPU;
	float * yloc_GPU;
	
	//int * ind = (int*)malloc(sizeof(int)*countOnes);
	int * ind_GPU;
	//float * u = (float *)malloc(sizeof(float)*Nparticles);
	float * u_GPU;
	int * seed_GPU;
	float * partial_sums;
	float * x_partial_sums;
	float * y_partial_sums;
	
	//CUDA memory allocation
	check_error(cudaMalloc((void **) &arrayX_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &arrayY_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &xj_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &yj_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &CDF_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &u_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &likelihood_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &weights_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &I_GPU, sizeof(unsigned char)*IszX*IszY*Nfr));
	check_error(cudaMalloc((void **) &objxy_GPU, sizeof(int)*countOnes));
	check_error(cudaMalloc((void **) &ind_GPU, sizeof(int)*countOnes*Nparticles));
	check_error(cudaMalloc((void **) &seed_GPU, sizeof(int)*Nparticles));
	check_error(cudaMalloc((void **) &partial_sums, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &xloc_GPU, sizeof(float)*Nfr));
	check_error(cudaMalloc((void **) &yloc_GPU, sizeof(float)*Nfr));
	check_error(cudaMalloc((void **) &x_partial_sums, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &y_partial_sums, sizeof(float)*Nparticles));
	
	
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;
	//float * Ik = (float *)malloc(sizeof(float)*IszX*IszY);

	//start send
	long long send_start = get_time();
	cudaMemcpy(I_GPU, I, sizeof(unsigned char)*IszX*IszY*Nfr, cudaMemcpyHostToDevice);
	cudaMemcpy(objxy_GPU, objxy, sizeof(int)*countOnes, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_GPU, weights, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayX_GPU, arrayX, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayY_GPU, arrayY, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(seed_GPU, seed, sizeof(int)*Nparticles, cudaMemcpyHostToDevice);
	long long send_end = get_time();
	printf("TIME TO SEND TO GPU: %f\n", elapsed_time(send_start, send_end));
	int num_blocks = ceil((float) Nparticles/(float) threads_per_block);
	
	for(k = 1; k < Nfr; k++){
		
		
		likelihood_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, ind_GPU, objxy_GPU, likelihood_GPU, I_GPU, u_GPU, weights_GPU, Nparticles, countOnes, max_size, k, IszX, IszY, Nfr, seed_GPU, partial_sums);
		sum_kernel <<< num_blocks, threads_per_block >>> (partial_sums, Nparticles);
		
		normalize_weights_kernel <<< num_blocks, threads_per_block >>> (weights_GPU, Nparticles, partial_sums, CDF_GPU, u_GPU, seed_GPU, x_partial_sums, y_partial_sums, arrayX_GPU, arrayY_GPU);
		sum_xy_kernel <<< num_blocks, threads_per_block >>> (x_partial_sums, y_partial_sums, Nparticles);
		cudaBindTexture(0, tex_CDF, CDF_GPU, Nparticles);
		//KERNEL FUNCTION CALL
		find_index_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, weights_GPU, Nparticles, x_partial_sums, y_partial_sums, k, xloc_GPU, yloc_GPU);
		cudaUnbindTexture(tex_CDF);
		
	}
	
	long long back_time = get_time();
	
	cudaFree(xj_GPU);
	cudaFree(yj_GPU);
	cudaFree(CDF_GPU);
	cudaFree(u_GPU);
	cudaFree(likelihood_GPU);
	cudaFree(I_GPU);
	cudaFree(objxy_GPU);
	cudaFree(ind_GPU);
	cudaFree(seed_GPU);
	cudaFree(partial_sums);
	//CUDA freeing of memory
	cudaFree(weights_GPU);
	cudaFree(arrayY_GPU);
	cudaFree(arrayX_GPU);
	cudaFree(x_partial_sums);
	cudaFree(y_partial_sums);
	
	long long free_time = get_time();
	
	cudaMemcpy(x_loc, xloc_GPU, sizeof(float)*Nfr, cudaMemcpyDeviceToHost);
	long long x_loc_time = get_time();
	cudaMemcpy(y_loc, yloc_GPU, sizeof(float)*Nfr, cudaMemcpyDeviceToHost);
	long long y_loc_time = get_time();
	printf("GPU Execution: %lf\n", elapsed_time(send_end, back_time));
	printf("FREE TIME: %lf\n", elapsed_time(back_time, free_time));
	printf("SEND TO SEND BACK: %lf\n", elapsed_time(back_time, y_loc_time));
	printf("SEND ARRAY X BACK: %lf\n", elapsed_time(free_time, x_loc_time));
	printf("SEND ARRAY Y BACK: %lf\n", elapsed_time(x_loc_time, y_loc_time));
	
	cudaFree(xloc_GPU);
	cudaFree(yloc_GPU);
	x_loc[0] = xe;
	y_loc[0] = ye;
		
}
void particleFilter1F(unsigned char * I, int IszX, int IszY, int * seed, int Nparticles, float * x_loc, float * y_loc, float prevX, float prevY){
	
	int max_size = IszX*IszY;
	long long start = get_time();
	/*expected object locations, compared to center*/
	int radius = 5;
	int diameter = radius*2 -1;
	int * disk = (int *)mxCalloc(diameter*diameter, sizeof(int));
	strelDisk(disk, radius);
	int countOnes = 0;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(disk[x*diameter + y] == 1)
			countOnes++;
		}
	}
	int * objxy = (int *)mxCalloc(countOnes*2, sizeof(int));
	getneighbors(disk, countOnes, objxy, radius);
	
	
	//initial weights are all equal (1/Nparticles)
	float * weights = (float *)mxCalloc(Nparticles,sizeof(float));
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((float)(Nparticles));
	}
	
	//initial likelihood to 0.0
	float * likelihood = (float *)mxCalloc(Nparticles, sizeof(float));
	float * arrayX = (float *)mxCalloc(Nparticles, sizeof(float));
	float * arrayY = (float *)mxCalloc(Nparticles, sizeof(float));
	float * xj = (float *)mxCalloc(Nparticles, sizeof(float));
	float * yj = (float *)mxCalloc(Nparticles, sizeof(float));
	float * CDF = (float *)mxCalloc(Nparticles, sizeof(float));
	
	//GPU copies of arrays
	float * arrayX_GPU;
	float * arrayY_GPU;
	float * xj_GPU;
	float * yj_GPU;
	float * CDF_GPU;
	float * likelihood_GPU;
	unsigned char * I_GPU;
	float * weights_GPU;
	int * objxy_GPU;
	float * xloc_GPU;
	float * yloc_GPU;
	
	//int * ind = (int*)malloc(sizeof(int)*countOnes);
	int * ind_GPU;
	//float * u = (float *)malloc(sizeof(float)*Nparticles);
	float * u_GPU;
	int * seed_GPU;
	float * partial_sums;
	float * x_partial_sums;
	float * y_partial_sums;
	
	//CUDA memory allocation
	check_error(cudaMalloc((void **) &arrayX_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &arrayY_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &xj_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &yj_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &CDF_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &u_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &likelihood_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &weights_GPU, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &I_GPU, sizeof(unsigned char)*IszX*IszY*Nfr));
	check_error(cudaMalloc((void **) &objxy_GPU, sizeof(int)*countOnes));
	check_error(cudaMalloc((void **) &ind_GPU, sizeof(int)*countOnes*Nparticles));
	check_error(cudaMalloc((void **) &seed_GPU, sizeof(int)*Nparticles));
	check_error(cudaMalloc((void **) &partial_sums, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &xloc_GPU, sizeof(float)));
	check_error(cudaMalloc((void **) &yloc_GPU, sizeof(float)));
	check_error(cudaMalloc((void **) &x_partial_sums, sizeof(float)*Nparticles));
	check_error(cudaMalloc((void **) &y_partial_sums, sizeof(float)*Nparticles));
	
	
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = prevX;
		arrayY[x] = prevY;
	}
	
	//float * Ik = (float *)malloc(sizeof(float)*IszX*IszY);

	//start send
	long long send_start = get_time();
	cudaMemcpy(I_GPU, I, sizeof(unsigned char)*IszX*IszY*Nfr, cudaMemcpyHostToDevice);
	cudaMemcpy(objxy_GPU, objxy, sizeof(int)*countOnes, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_GPU, weights, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayX_GPU, arrayX, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayY_GPU, arrayY, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(seed_GPU, seed, sizeof(int)*Nparticles, cudaMemcpyHostToDevice);
	long long send_end = get_time();
	printf("TIME TO SEND TO GPU: %f\n", elapsed_time(send_start, send_end));
	int num_blocks = ceil((float) Nparticles/(float) threads_per_block);
	
	//for(k = 1; k < Nfr; k++){
		
		
		likelihood_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, ind_GPU, objxy_GPU, likelihood_GPU, I_GPU, u_GPU, weights_GPU, Nparticles, countOnes, max_size, 0, IszX, IszY, 0, seed_GPU, partial_sums);
		sum_kernel <<< num_blocks, threads_per_block >>> (partial_sums, Nparticles);
		
		normalize_weights_kernel <<< num_blocks, threads_per_block >>> (weights_GPU, Nparticles, partial_sums, CDF_GPU, u_GPU, seed_GPU, x_partial_sums, y_partial_sums, arrayX_GPU, arrayY_GPU);
		sum_xy_kernel <<< num_blocks, threads_per_block >>> (x_partial_sums, y_partial_sums, Nparticles);
		cudaBindTexture(0, tex_CDF, CDF_GPU, Nparticles);
		//KERNEL FUNCTION CALL
		find_index_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, weights_GPU, Nparticles, x_partial_sums, y_partial_sums, 0, xloc_GPU, yloc_GPU);
		cudaUnbindTexture(tex_CDF);
		
	
	
	long long back_time = get_time();
	
	cudaFree(xj_GPU);
	cudaFree(yj_GPU);
	cudaFree(CDF_GPU);
	cudaFree(u_GPU);
	cudaFree(likelihood_GPU);
	cudaFree(I_GPU);
	cudaFree(objxy_GPU);
	cudaFree(ind_GPU);
	cudaFree(seed_GPU);
	cudaFree(partial_sums);
	//CUDA freeing of memory
	cudaFree(weights_GPU);
	cudaFree(arrayY_GPU);
	cudaFree(arrayX_GPU);
	cudaFree(x_partial_sums);
	cudaFree(y_partial_sums);
	
	long long free_time = get_time();
	
	cudaMemcpy(x_loc, xloc_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	long long x_loc_time = get_time();
	cudaMemcpy(y_loc, yloc_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	long long y_loc_time = get_time();
	printf("GPU Execution: %lf\n", elapsed_time(send_end, back_time));
	printf("FREE TIME: %lf\n", elapsed_time(back_time, free_time));
	printf("SEND TO SEND BACK: %lf\n", elapsed_time(back_time, y_loc_time));
	printf("SEND ARRAY X BACK: %lf\n", elapsed_time(free_time, x_loc_time));
	printf("SEND ARRAY Y BACK: %lf\n", elapsed_time(x_loc_time, y_loc_time));
	
	cudaFree(xloc_GPU);
	cudaFree(yloc_GPU);

	
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	unsigned char * I;
	int IszX, IszY, Nfr, Nparticles;
	srand(time(0));
	if(nrhs < 5)
	{
		printf("ERROR: TOO FEW ARGS HAVE BEEN ENTERED\n");
		printf("EXITING\n");
		exit(0);
	}
	else if(nrhs == 5)
	{
		IszX = (int)(mxGetScalar(prhs[1]));
		IszY = (int)(mxGetScalar(prhs[2]));
		Nfr = (int)(mxGetScalar(prhs[3]));
		Nparticles = (int)(mxGetScalar(prhs[4]));
		printf("ISZX: %d\n", IszX);
		printf("ISZY: %d\n", IszY);
		printf("Nfr: %d\n", Nfr);
		printf("Nparticles: %d\n", Nparticles);
		unsigned char * cI = (unsigned char *)mxGetData(prhs[0]);
		I = (unsigned char *)mxCalloc(IszX*IszY*Nfr, sizeof(unsigned char));
		int x, y, z;
		for(x = 0; x < IszX; x++){
			for(y = 0; y < IszY; y++){
				for(z = 0; z < Nfr; z++){
					I[x*IszY*Nfr + y*Nfr + z] = (unsigned char)cI[x*IszY*Nfr + y*Nfr + z];
				}
			}
		}
		
		int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
		int i;
		for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
		float * posX = (float *)mxCalloc(Nfr, sizeof(float));
		float * posY = (float *)mxCalloc(Nfr, sizeof(float));
		long long start = get_time();
		particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, posX, posY);
		long long end = get_time();
		mxFree(I);
		mxFree(seed);
		
		printf("PARTICLE FILTER TOOK %f\n", elapsed_time(start, end));
		plhs[0] = mxCreateDoubleMatrix(Nfr, 1, mxREAL);
		plhs[1] = mxCreateDoubleMatrix(Nfr, 1, mxREAL);
		double * bufferX = mxGetPr(plhs[0]);
		double * bufferY = mxGetPr(plhs[1]);
		for(i = 0; i < Nfr; i++)
		{
			bufferX[i] = (double)posX[i];
			bufferY[i] = (double)posY[i];
		}
		mxFree(posX);
		mxFree(posY);
	}
	else if(nrhs == 6)
	{
		IszX = (int)(mxGetScalar(prhs[1]));
		IszY = (int)(mxGetScalar(prhs[2]));
		Nparticles = (int)(mxGetScalar(prhs[3]));
		printf("ISZX: %d\n", IszX);
		printf("ISZY: %d\n", IszY);
		printf("Nparticles: %d\n", Nparticles);
		double startX = (double)mxGetScalar(prhs[4]);
		double startY = (double)mxGetScalar(prhs[5]);
		printf("Starting PosX: %lf\n", startX);
		printf("Starting PosY: %lf\n", startY);
		unsigned char * cI = (unsigned char *)mxGetData(prhs[0]);
		I = (unsigned char *)mxCalloc(IszX*IszY, sizeof(unsigned char));
		int x, y;
		for(x = 0; x < IszX; x++){
			for(y = 0; y < IszY; y++){
				I[x*IszX + y] = (unsigned char)cI[x*IszX + y];
			}
		}
		
		int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
		int i;
		for(i = 0; i < Nparticles; i++)
			seed[i] = time(0)*i;
		float posX[1];
		float posY[1];
		long long start = get_time();
		particleFilter1F(I, IszX, IszY, seed, Nparticles, posX, posY, (float)startX, (float)startY);
		long long end = get_time();
		mxFree(I);
		mxFree(seed);
		
		printf("PARTICLE FILTER TOOK %f\n", elapsed_time(start, end));
		plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
		plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
		double * bufferX = mxGetPr(plhs[0]);
		double * bufferY = mxGetPr(plhs[1]);
		bufferX[0] = posX[0];
		bufferY[0] = posY[0];
		
	}
	else
	{
		printf("ERROR: TOO MANY ARGS\n");
		printf("EXITING\n");
		exit(0);
	}
}
int main(){
	//establish seed
	int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
	int i;
	int IszX = 128;
	int IszY = 128;
	int Nfr = 10;
	int Nparticles = 10000;
	for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
	//malloc matrix
	unsigned char * I = (unsigned char *)malloc(sizeof(unsigned char)*IszX*IszY*Nfr);
	long long start = get_time();
	//call video sequence
	//videoSequence();
	long long endVideoSequence = get_time();
	printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
	//call particle filter
	//particleFilter();
	long long endParticleFilter = get_time();
	printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
	printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));
	return 0;
}
