/**
 * @file ex_particle_CUDA_float.cu
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation completely in CUDA
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>

#include "mex.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "driver_types.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI acos(-1)

/**
* @var tex_CDF The CDF texture array
*/
texture <float> tex_CDF;

/**
* @var threads_per_block The number of threads per block used on the GPU
*/
const int threads_per_block = 512;
/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is NOT thread-safe
* @return a double representing a Gaussian random number
*/
double randn(){
	/* Box-Muller algorithm */
	double u = (double)rand();
	u = u/RAND_MAX;
	double v = (double)rand();
	v = v/RAND_MAX;
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/**
* Checks for CUDA errors and prints them to the screen to help with
* debugging of CUDA related programming
* @param e Cuda error code
*/
void check_error(cudaError e) {
     if (e != cudaSuccess) {
     	printf("\nCUDA error: %s\n", cudaGetErrorString(e));
	    exit(1);
     }
}
/**
* Device funciton that determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
__device__ float calcLikelihoodSum(unsigned char * I, int * ind, int numOnes, int index){
	float likelihoodSum = 0.0;
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((float)(I[ind[index*numOnes + x]] - 100),2) - pow((float)(I[ind[index*numOnes + x]]-228),2))/50.0;
	return likelihoodSum;
}
/**
* Device function used to calculated the CDF using the previously calculated weights
* @param CDF The CDF array
* @param weights The weights array
* @param Nparticles The length of CDF + weights array
*/
__device__ void cdfCalc(float * CDF, float * weights, int Nparticles){
	int x;
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
}
/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe for use on the GPU
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
__device__ float d_randu(int * seed, int index)
{
	//use GCC's M, A and C value for the LCG
	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	num = seed[index];
	return fabs(num/((float) M));
}

/**
* Generates a normally distributed random number using the Box-Muller transformation on the GPU
* @note This function is thread-safe for use on the GPU
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
__device__ float d_randn(int * seed, int index){
	//Box-Muller algortihm
	float pi = 3.14159265358979323846;
	float u = d_randu(seed, index);
	float v = d_randu(seed, index);
	float cosine = cos(2*pi*v);
	float rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/**
* @deprecated device function for calculating the weights; replaced by reduction function for weights
* @param weights The weights array
* @param likelihood The likelihood array
* @param Nparticles The length of the weights and likelihood arrays
* @return The sum of the weights
*/
__device__ float updateWeights(float * weights, float * likelihood, int Nparticles){
	int x;
	float sum = 0;
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x] * exp(likelihood[x]);
		sum += weights[x];
	}		
	return sum;
}
/**
* Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
* @note This function uses binary search before switching to sequential search
* @param CDF The CDF
* @param beginIndex The index to start searching from
* @param endIndex The index to stop searching
* @param value The value to find
* @return The index of value in the CDF; if value is never found, returns the last index
* @warning Use at your own risk; not fully tested
*/
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
/**
* Finds the first element in the CDF that is greater than or equal to the provided value and updates arrayX and arrayY using those values
* @note This function uses sequential search
* @param arrayX The array containing the guesses in the x direction
* @param arrayY The array containing the guesses in the y direction
* @param CDF The CDF array
* @param u The array containing the updated values
* @param xj The temp array for arrayX
* @param yj The temp array for arrayY
* @param weights The weights array
* @param Nparticles The number of particles used
* @param x_partial_sums The array containing the parital sum of arrayX; final sum is at index 0
* @param y_partial_sums The array containing the partial sum of arrayY; final sum is at index 0
* @param k The current frame
* @param x_loc The array containing the x location of the object for frame k
* @param y_loc The array containing the y location of the object for frame k
*/
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
/**
* Normalizes the weights using the partial sums previously calculated; sets up the partial sums for the x + y positions
* @param weights The weights array
* @param Nparticles The length of the weights array, arrayX and arrayY
* @param partial_sums The array containing the result of the partial sums in its initial index
* @param CDF The CDF array
* @param u The array for calculating the indices used for resampling
* @param seed The seed array used for random number generation
* @param x_partial_sums The array used for storing the partial sums of arrayX
* @param y_partial_sums The array used for storing the partial sums of arrayY
* @param arrayX The array storing the guesses for the x position of the particle
* @param arrayY The array storing the guesses for the y position of the particle
*/
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
/**
* Calculates the ultimate sum using the partial sums array & stores it at index 0
* @param partial_sums The array containing the partial sums
* @param Nparticles The length of the array
*/
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
/**
* Calculates the ultimate sum using the partial sums arrays & store them at index 0 of the respective arrays
* @param x_partial_sums The array containing the partial sums of arrayX
* @param y_partial_sums The array containing the partial sums of arrayY
* @param Nparticles The length of the arrays
*/
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
/**
* Calculates the likelihoods of an object going to the positions guessed using arrayX and arrayY
* @param arrayX The array containing the guesses in the x direction
* @param arrayY The array containing the guesses in the y direction
* @param ind The translated position in the video
* @param objxy The representation of the object
* @param likelihood The likelihood array
* @param I The video data to be analyzed
* @param u The array containing the update data
* @param weights The weights array
* @param Nparticles The number of particles to be used
* @param countOnes The length objxy
* @param max_size The maximum index in I
* @param k The current frame
* @param IszX The x dimension
* @param IszY The y dimension
* @param Nfr The number of frames
* @param seed The seed array
* @param partial_sums The partial sums array
*/
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
/**
* Calculates the likelihood for a single frame
* @param arrayX The array containing the guesses in the x direction
* @param arrayY The array containing the guesses in the y direction
* @param CDF The CDF array
* @param ind The array containing the translated addresses for I
* @param objxy The representation of the object to be tracked
* @param likelihood The likelihood array
* @param I The image to be analyzed
* @param u The array containing the information for updating arrayX and arrayY
* @param weights The weights array
* @param Nparticles The number of particles
* @param countOnes The length of the objxy array
* @param max_size The maximum length of objxy
* @param IszX The x dimension of the image
* @param IszY The y dimension of the image
* @param seed The seed array
* @param partial_sums The partial sums array
*/
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
/** 
* Takes in a double and returns an integer that approximates to that double
* @if the mantissa < .5 => return value < input value
* @else return value > input value
* @endif
*/
float roundDouble(float value){
	int newValue = (int)(value);
	if(value - newValue < .5)
		return newValue;
	else
		return newValue++;
}

/**
* Set values of the 3D array to a newValue if that value is equal to the testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
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
/**
* Sets values of 3D matrix using randomly generated numbers from a normal distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
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
/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
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

/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
{
	int startX = posX - error;
	while(startX < 0)
	startX++;
	int startY = posY - error;
	while(startY < 0)
	startY++;
	int endX = posX + error;
	while(endX > dimX)
	endX--;
	int endY = posY + error;
	while(endY > dimY)
	endY--;
	int x,y;
	for(x = startX; x < endX; x++){
		for(y = startY; y < endY; y++){
			double distance = sqrt( pow((double)(x-posX),2) + pow((double)(y-posY),2) );
			if(distance < error)
			matrix[x*dimY*dimZ + y*dimZ + posZ] = 1;
		}
	}
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
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
/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
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
/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
*/
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

/**
* The implementation of the particle filter using CUDA for many frames
* @see http://www.nvidia.com/object/cuda_home_new.html
* @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
* @param I The video to be run
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames
* @param seed The seed array used for random number generation
* @param Nparticles The number of particles to be used
* @param x_loc The array that will store the x locations of the desired object
* @param y_loc The array that will store the y locations of the desired object
* @param xe The starting x position of the object
* @param ye The starting y position of the object
*/
void particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, float * x_loc, float * y_loc, float xe, float ye){
	int max_size = IszX*IszY*Nfr;
	//original particle centroid
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
	cudaMemcpy(I_GPU, I, sizeof(unsigned char)*IszX*IszY*Nfr, cudaMemcpyHostToDevice);
	cudaMemcpy(objxy_GPU, objxy, sizeof(int)*countOnes, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_GPU, weights, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayX_GPU, arrayX, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayY_GPU, arrayY, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(seed_GPU, seed, sizeof(int)*Nparticles, cudaMemcpyHostToDevice);
	
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
	

	//CUDA freeing of memory
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
	cudaFree(weights_GPU);
	cudaFree(arrayY_GPU);
	cudaFree(arrayX_GPU);
	cudaFree(x_partial_sums);
	cudaFree(y_partial_sums);
	

	cudaMemcpy(x_loc, xloc_GPU, sizeof(float)*Nfr, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_loc, yloc_GPU, sizeof(float)*Nfr, cudaMemcpyDeviceToHost);
	
	cudaFree(xloc_GPU);
	cudaFree(yloc_GPU);
	x_loc[0] = xe;
	y_loc[0] = ye;
		
}
/**
* The implementation of the particle filter using CUDA for a single image
* @see http://www.nvidia.com/object/cuda_home_new.html
* @note This function is designed to work with a single image. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
* @warning Use the other particle filter function for videos; the accuracy of this function decreases significantly as it is called repeatedly while processing video
* @param I The image to be run
* @param IszX The x dimension of the image
* @param IszY The y dimension of the image
* @param seed The seed array used for random number generation
* @param Nparticles The number of particles to be used
* @param x_loc The array that will store the x locations of the desired object
* @param y_loc The array that will store the y locations of the desired object
* @param prevX The starting x position of the object
* @param prevY The starting y position of the object
*/
void particleFilter1F(unsigned char * I, int IszX, int IszY, int * seed, int Nparticles, float * x_loc, float * y_loc, float prevX, float prevY){
	
	int max_size = IszX*IszY;
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
	
	int * ind_GPU;
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
	check_error(cudaMalloc((void **) &I_GPU, sizeof(unsigned char)*IszX*IszY));
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
	

	//start send
	cudaMemcpy(I_GPU, I, sizeof(unsigned char)*IszX*IszY, cudaMemcpyHostToDevice);
	cudaMemcpy(objxy_GPU, objxy, sizeof(int)*countOnes, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_GPU, weights, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayX_GPU, arrayX, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayY_GPU, arrayY, sizeof(float)*Nparticles, cudaMemcpyHostToDevice);
	cudaMemcpy(seed_GPU, seed, sizeof(int)*Nparticles, cudaMemcpyHostToDevice);
	
	int num_blocks = ceil((float) Nparticles/(float) threads_per_block);
	

		
	likelihood_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, ind_GPU, objxy_GPU, likelihood_GPU, I_GPU, u_GPU, weights_GPU, Nparticles, countOnes, max_size, 0, IszX, IszY, 0, seed_GPU, partial_sums);
	sum_kernel <<< num_blocks, threads_per_block >>> (partial_sums, Nparticles);
	
	normalize_weights_kernel <<< num_blocks, threads_per_block >>> (weights_GPU, Nparticles, partial_sums, CDF_GPU, u_GPU, seed_GPU, x_partial_sums, y_partial_sums, arrayX_GPU, arrayY_GPU);
	sum_xy_kernel <<< num_blocks, threads_per_block >>> (x_partial_sums, y_partial_sums, Nparticles);
	cudaBindTexture(0, tex_CDF, CDF_GPU, Nparticles);
	//KERNEL FUNCTION CALL
	find_index_kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, weights_GPU, Nparticles, x_partial_sums, y_partial_sums, 0, xloc_GPU, yloc_GPU);
	cudaUnbindTexture(tex_CDF);
		
	//CUDA freeing of memory
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
	cudaFree(weights_GPU);
	cudaFree(arrayY_GPU);
	cudaFree(arrayX_GPU);
	cudaFree(x_partial_sums);
	cudaFree(y_partial_sums);
	

	cudaMemcpy(x_loc, xloc_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y_loc, yloc_GPU, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(xloc_GPU);
	cudaFree(yloc_GPU);

	
}
/**
* Function that allows the 2 particle filter implementations to be run
* @details The number of arguments provided to this function determines which function will be called. 7 args will call the video processing version. 6 (leaving out the number of frames) will call the image processing version.
* @param nlhs (Number on the Left Hand Side) The number of items to return (2 will be in this case; the x and y arrays)
* @param plhs (Parameters on the Left Hand Side) A pointer to the arrays containing the x and y arrays
* @param nrhs (Number on the Right Hand Side) The number of arguments to take in (7 are needed for video processing (The image as an unsigned char, the x dimension, the y dimension, the number of frames, the number of particles, the x starting position, the y starting position)
* 6 are needed for the image processing (same as before but leave out the number of frames)
* @param prhs (Parameters on the Right Hand Side) A pointer to the arrays containing the parameters
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	unsigned char * I;
	int IszX, IszY, Nfr, Nparticles;
	srand(time(0));
	if(nrhs < 6)
	{
		printf("ERROR: TOO FEW ARGS HAVE BEEN ENTERED\n");
		printf("EXITING\n");
		exit(0);
	}
	else if(nrhs == 7)
	{
		IszX = (int)(mxGetScalar(prhs[1]));
		IszY = (int)(mxGetScalar(prhs[2]));
		Nfr = (int)(mxGetScalar(prhs[3]));
		Nparticles = (int)(mxGetScalar(prhs[4]));

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
		
		float xe = (float)mxGetScalar(prhs[5]);
		float ye = (float)mxGetScalar(prhs[6]);
		int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
		int i;
		for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
		float * posX = (float *)mxCalloc(Nfr, sizeof(float));
		float * posY = (float *)mxCalloc(Nfr, sizeof(float));

		particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, posX, posY, xe, ye);

		mxFree(I);
		mxFree(seed);
		
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
		
		double startX = (double)mxGetScalar(prhs[4]);
		double startY = (double)mxGetScalar(prhs[5]);
		
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

		particleFilter1F(I, IszX, IszY, seed, Nparticles, posX, posY, (float)startX, (float)startY);

		mxFree(I);
		mxFree(seed);
		

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
/**
* Unused
*/
int main(){
	
	
	return 0;
}
