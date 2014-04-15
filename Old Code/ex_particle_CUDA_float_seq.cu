#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
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
int Nparticles = 100;
int * seed;

texture <float> tex_CDF;
texture <float> tex_sums;

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
__global__ void find_index_kernel(float * arrayX, float * arrayY, float * CDF, float * u, float * xj, float * yj, float * weights, int Nparticles){
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;
	
	if(i < Nparticles){
	
		int index = -1;
		int x;

		for(x = 0; x < Nparticles; x++){
			if(tex1Dfetch(tex_CDF, x) >= u[i] && index == -1){
				index = x;
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
__global__ void normalize_weights_kernel(float * weights, int Nparticles, float * partial_sums, float * CDF, float * u, int * seed)
{
	int block_id = blockIdx.x;
	int i = blockDim.x*block_id + threadIdx.x;
	__shared__ float u1, sumWeights;
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
			sum += tex1Dfetch(tex_sums, x);
		}
		partial_sums[0] = sum;
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
__global__ void likelihood_kernel(float * arrayX, float * arrayY, float * CDF, int * ind, int * objxy, float * likelihood, unsigned char * I, float * u, float * weights, int Nparticles, int countOnes, int max_size, int k, int IszY, int Nfr, int *seed, float * partial_sums){
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
			ind[i*countOnes + y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
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

/*****************************
* SETIF
* set values of the 3D array to a newValue if that value is equal to the testValue
* param1: value to test
* param2: 3D array
* param3: dim X
* param4: dim Y
* param5: dim Z
******************************/
void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ){
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
void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ){
	int x, y, z;
	//
	for(x = 0; x < *dimX; x++){
		//
		for(y = 0; y < *dimY; y++){
			//
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
int * strelDisk()
{
	int * disk = (int *)malloc(sizeof(int)*9*9);
	int x, y;
	for(x = 0; x < 9; x++){
		for(y = 0; y < 9; y++){
			float distance = sqrt(pow((float)(x-4),2) + pow((float)(y-4),2));
			if(distance < 5.0)
				disk[x*9 + y] = 1;
		}
	}
	return disk;
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
void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
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
			float distance = sqrt( pow((float)(x-posX),2) + pow((float)(y-posY),2) );
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
unsigned char* imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error)
{
	unsigned char * newMatrix = (unsigned char *)malloc(sizeof(unsigned char)*dimX*dimY*dimZ);
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
	free(matrix);
	return newMatrix;
}
/*****************************
* GET NEIGHBORS
* returns a 2D array describing the offets
* param 1 strel object
* param 2 dimX of object
* param 3 dimY of object
*******************************/
int * getneighbors(int * se, int numOnes){
	int * neighbors = (int *)malloc(sizeof(int)*numOnes*2);
	int x, y;
	int neighY = 0;
	int center = 4;
	for(x = 0; x < 9; x++){
		for(y = 0; y < 9; y++){
			if(se[x*9 + y]){
				neighbors[neighY*2] = (int)(y - center);
				neighbors[neighY*2 + 1] = (int)(x - center);
				neighY++;
			}
		}
	}
	return neighbors;
}
/******************************
* VIDEO SEQUENCE
* the synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
*******************************/
void videoSequence(){
	int k;
	int max_size = IszX*IszY*Nfr;
	//get object centers
	int x0 = (int)roundDouble(IszY/2.0);
	int y0 = (int)roundDouble(IszX/2.0);
	I[x0 *IszY *Nfr + y0 * Nfr  + 0] = 1;
	
	//move point
	int xk, yk, pos;
	for(k = 1; k < Nfr; k++){
		xk = abs(x0 + (k));
		yk = abs(y0 - 2*(k));
		pos = yk * IszY * Nfr + xk *Nfr + k;
		if(pos >= max_size)
			pos = 0;
		I[pos] = 1;
	}	
	int x, y;
	int count = 0;
	/*for(x = 0; x < IszX; x++)
		for(y = 0; y < IszY; y++)
			for(k = 0; k < Nfr; k++)
				if(I[x*IszY*Nfr + y*Nfr + k]){
					printf("ARRAY [%d][%d][%d]: %d\n", x, y, k, I[x*IszY*Nfr + y*Nfr + k]);
					count++;
					}
	printf("COUNT: %d\n", count);*/
	//dilate matrix
	I = imdilate_disk(I, IszX, IszY, Nfr, 5);
	count = 0;
	/*printf("SECOND TIME\n");
	for(k = 0; k< Nfr; k++)
		for(x = 0; x < IszX; x++)
			for(y = 0; y < IszY; y++)
				if(I[x*IszY*Nfr + y*Nfr + k]){
					printf("ARRAY [%d][%d][%d]: %d\n", x, y, k, I[x*IszY*Nfr + y*Nfr + k]);
					count++;
					}
	printf("COUNT: %d", count);*/
	//define background, add noise
	setIf(0, 100, I, &IszX, &IszY, &Nfr);
	setIf(1, 228, I, &IszX, &IszY, &Nfr);
	//add noise
	addNoise(I, &IszX, &IszY, &Nfr);
}
/******************************
* FIND INDEX
* FINDS THE FIRST OCCURRENCE OF AN ELEMENT IN CDF GREATER THAN THE PROVIDED VALUE AND RETURNS THAT INDEX
* param1 CDF
* param2 length of CDF
* param3 value
*******************************/
int findIndex(float * CDF, int lengthCDF, float value){
	int index = -1;
	int x;
	for(x = 0; x < lengthCDF; x++){
		if(CDF[x] >= value){
			index = x;
			break;
		}
	}
	if(index == -1){
		return lengthCDF-1;
	}
	return index;
}
void particleFilter(){
	int max_size = IszX*IszY*Nfr;
	//original particle centroid
	float xe = roundDouble(IszY/2.0);
	float ye = roundDouble(IszX/2.0);
	
	//expected object locations, compared to center
	int radius = 5;
	int * disk = strelDisk();
	int countOnes = 0;
	int x, y;
	for(x = 0; x < 9; x++){
		for(y = 0; y < 9; y++){
			if(disk[x*9 + y] == 1)
				countOnes++;
			//printf("%d ", disk[x*9+y]);
		}
		//printf("\n");
	}
	int * objxy = getneighbors(disk, countOnes);
	/*for(x = 0; x < countOnes; x++){
		printf("%d %d\n", objxy[x*2], objxy[x*2 + 1]);
	}
	printf("NUM ONES: %d\n", countOnes);*/
	
	//initial weights are all equal (1/Nparticles)
	float * weights = (float *)malloc(sizeof(float)*Nparticles);
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((float)(Nparticles));
	}
	
	//initial likelihood to 0.0
	float * likelihood = (float *)malloc(sizeof(float)*Nparticles);
	float * arrayX = (float *)malloc(sizeof(float)*Nparticles);
	float * arrayY = (float *)malloc(sizeof(float)*Nparticles);
	float * xj = (float *)malloc(sizeof(float)*Nparticles);
	float * yj = (float *)malloc(sizeof(float)*Nparticles);
	float * CDF = (float *)malloc(sizeof(float)*Nparticles);
	
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
	
	int * ind = (int*)malloc(sizeof(int)*countOnes);
	int * ind_GPU;
	float * u = (float *)malloc(sizeof(float)*Nparticles);
	float * u_GPU;
	int * seed_GPU;
	float * partial_sums;
	
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
	
	
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;
	//float * Ik = (float *)malloc(sizeof(float)*IszX*IszY);
	int indX, indY;
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
		
		//apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction
		
		/*for(x = 0; x < Nparticles; x++){
			arrayX[x] = arrayX[x] + 1.0 + 5.0*randn();
			arrayY[x] = arrayY[x] - 2.0 + 2.0*randn();
		}
		//particle filter likelihood
		
		for(x = 0; x < Nparticles; x++){
		
			//compute the likelihood: remember our assumption is that you know
			// foreground and the background image intensity distribution.
			// Notice that we consider here a likelihood ratio, instead of
			// p(z|x). It is possible in this case. why? a hometask for you.		
			//calc ind
			for(y = 0; y < countOnes; y++){
				indX = roundDouble(arrayX[x]) + objxy[y*2 + 1];
				indY = roundDouble(arrayY[x]) + objxy[y*2];
				ind[y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
				if(ind[y] >= max_size)
					ind[y] = 0;
			}
			likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
			likelihood[x] = likelihood[x]/countOnes;
		}
		// update & normalize weights
		// using equation (63) of Arulampalam Tutorial	

		float sumWeights = updateWeights(weights, likelihood, Nparticles);
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x] * exp(likelihood[x]);
		}
		
		float sumWeights = 0;	
		for(x = 0; x < Nparticles; x++){
			sumWeights += weights[x];
		}
		for(x = 0; x < Nparticles; x++){
				weights[x] = weights[x]/sumWeights;
		}
		xe = 0;
		ye = 0;
		// estimate the object location by expected values
		for(x = 0; x < Nparticles; x++){
			xe += arrayX[x] * weights[x];
			ye += arrayY[x] * weights[x];
		}
		printf("XE: %lf\n", xe);
		printf("YE: %lf\n", ye);
		//float distance = sqrt( pow((float)(xe-(int)roundDouble(IszY/2.0)),2) + pow((float)(ye-(int)roundDouble(IszX/2.0)),2) );
		//printf("%lf\n", distance);
		//display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling
		
		
		cdfCalc(CDF, weights, Nparticles);
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}
		
		float u1 = (1/((float)(Nparticles)))*randu();
		for(x = 0; x < Nparticles; x++){
			u[x] = u1 + x/((float)(Nparticles));
		}*/
		
		likelihood_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, ind_GPU, objxy_GPU, likelihood_GPU, I_GPU, u_GPU, weights_GPU, Nparticles, countOnes, max_size, k, IszY, Nfr, seed_GPU, partial_sums);
		sum_kernel <<< num_blocks, threads_per_block >>> (partial_sums, Nparticles);
		
		cudaBindTexture(0, tex_sums, partial_sums, Nparticles);
		normalize_weights_kernel <<< num_blocks, threads_per_block >>> (weights_GPU, Nparticles, partial_sums, CDF_GPU, u_GPU, seed_GPU);
		cudaUnbindTexture(tex_sums);
		//long long start_copy = get_time();
		//CUDA memory copying from CPU memory to GPU memory
		
		//long long end_copy = get_time();
		//Set number of threads
		cudaBindTexture(0, tex_CDF, CDF_GPU, Nparticles);
		//KERNEL FUNCTION CALL
		find_index_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, weights_GPU, Nparticles);
		cudaUnbindTexture(tex_CDF);
		
		//long long start_copy_back = get_time();
		//CUDA memory copying back from GPU to CPU memory
		//cudaMemcpy(arrayY_GPU, yj_GPU, sizeof(float)*Nparticles, cudaMemcpyDeviceToDevice);
		//cudaMemcpy(arrayX_GPU, xj_GPU, sizeof(float)*Nparticles, cudaMemcpyDeviceToDevice);
		//long long end_copy_back = get_time();
		//printf("SENDING TO GPU TOOK: %lf\n", elapsed_time(start_copy, end_copy));
		//printf("CUDA EXEC TOOK: %lf\n", elapsed_time(end_copy, start_copy_back));
		//printf("SENDING BACK FROM GPU TOOK: %lf\n", elapsed_time(start_copy_back, end_copy_back));
		/**
		int j, i;
		for(j = 0; j < Nparticles; j++){
			i = findIndex(CDF, Nparticles, u[j]);
			xj[j] = arrayX[i];
			yj[j] = arrayY[i];
			
		}
		**/

		//reassign arrayX and arrayY
		//arrayX = xj;
		//arrayY = yj;
		
		/*for(x = 0; x < Nparticles; x++){
			weights[x] = 1/((float)(Nparticles));
		}*/
	}
	
	/*check_error(cudaMalloc((void **) &arrayX_GPU, sizeof(float)*Nparticles));
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
	check_error(cudaMalloc((void **) &partial_sums, sizeof(float)*Nparticles));*/
	
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
	
	long long free_time = get_time();
	
	cudaMemcpy(arrayX, arrayX_GPU, sizeof(float)*Nparticles, cudaMemcpyDeviceToHost);
	long long arrayX_time = get_time();
	cudaMemcpy(arrayY, arrayY_GPU, sizeof(float)*Nparticles, cudaMemcpyDeviceToHost);
	long long arrayY_time = get_time();
	cudaMemcpy(weights, weights_GPU, sizeof(float)*Nparticles, cudaMemcpyDeviceToHost);
	long long back_end_time = get_time();
	printf("GPU Execution: %lf\n", elapsed_time(send_end, back_time));
	printf("FREE TIME: %lf\n", elapsed_time(back_time, free_time));
	printf("SEND TO SEND BACK: %lf\n", elapsed_time(back_time, back_end_time));
	printf("SEND ARRAY X BACK: %lf\n", elapsed_time(free_time, arrayX_time));
	printf("SEND ARRAY Y BACK: %lf\n", elapsed_time(arrayX_time, arrayY_time));
	printf("SEND WEIGHTS BACK: %lf\n", elapsed_time(arrayY_time, back_end_time));
	
	xe = 0;
	ye = 0;
	// estimate the object location by expected values
	for(x = 0; x < Nparticles; x++){
		xe += arrayX[x] * weights[x];
		ye += arrayY[x] * weights[x];
	}
	printf("XE: %lf\n", xe);
	printf("YE: %lf\n", ye);
	float distance = sqrt( pow((float)(xe-(int)roundDouble(IszY/2.0)),2) + pow((float)(ye-(int)roundDouble(IszX/2.0)),2) );
	printf("%lf\n", distance);
	
	//CUDA freeing of memory
	cudaFree(weights_GPU);
	cudaFree(arrayY_GPU);
	cudaFree(arrayX_GPU);

}
int main(){
	//establish seed
	seed = (int *)malloc(sizeof(int)*Nparticles);
	int i;
	for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
	//malloc matrix
	I = (unsigned char *)malloc(sizeof(unsigned char)*IszX*IszY*Nfr);
	long long start = get_time();
	//call video sequence
	videoSequence();
	long long endVideoSequence = get_time();
	printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
	//call particle filter
	particleFilter();
	long long endParticleFilter = get_time();
	printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
	printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));
	return 0;
}