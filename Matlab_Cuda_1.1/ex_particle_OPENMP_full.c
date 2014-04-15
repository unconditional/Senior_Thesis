/**
 * @file ex_particle_OPENMP_full.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP 
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include "mex.h"
#include "mat.h"
#define PI acos(-1)
/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;

/** 
* Takes in a double and returns an integer that approximates to that double
* @if the mantissa < .5 => return value < input value
* @else return value > input value
* @endif
*/
double roundDouble(double value){
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
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int * seed, int index)
{
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}
/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double randn(int * seed, int index){
	/*Box-Muller algorithm*/
	double u = randu(seed, index);
	double v = randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/**
* Sets values of 3D matrix using randomly generated numbers from a normal distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed){
	int x, y, z;
	for(x = 0; x < *dimX; x++){
		for(y = 0; y < *dimY; y++){
			for(z = 0; z < *dimZ; z++){
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int)(5*randn(seed, 0));
			}
		}
	}
}
/**
* Creates a 9x9 matrix representing the disk
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
void getneighbors(int * se, int numOnes, double * neighbors, int radius){
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
* @param seed The seed array used for number generation
*/
void videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed){
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
	int * newMatrix = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
	imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
	int x, y;
	for(x = 0; x < IszX; x++){
		for(y = 0; y < IszY; y++){
			for(k = 0; k < Nfr; k++){
				I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
			}
		}
	}
	free(newMatrix);
	
	/*define background, add noise*/
	setIf(0, 100, I, &IszX, &IszY, &Nfr);
	setIf(1, 228, I, &IszX, &IszY, &Nfr);
	/*add noise*/
	addNoise(I, &IszX, &IszY, &Nfr, seed);
}
/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double calcLikelihoodSum(int * I, int * ind, int numOnes){
	double likelihoodSum = 0.0;
	int y;
	for(y = 0; y < numOnes; y++)
	likelihoodSum += (pow((I[ind[y]] - 100),2) - pow((I[ind[y]]-228),2))/50.0;
	return likelihoodSum;
}
/**
* Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
* @note This function uses sequential search
* @param CDF The CDF
* @param lengthCDF The length of CDF
* @param value The value to be found
* @return The index of value in the CDF; if value is never found, returns the last index
*/
int findIndex(double * CDF, int lengthCDF, double value){
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
int findIndexBin(double * CDF, int beginIndex, int endIndex, double value){
	if(endIndex < beginIndex)
	return -1;
	int middleIndex = beginIndex + ((endIndex - beginIndex)/2);
	/*check the value*/
	if(CDF[middleIndex] >= value)
	{
		/*check that it's good*/
		if(middleIndex == 0)
		return middleIndex;
		else if(CDF[middleIndex-1] < value)
		return middleIndex;
		else if(CDF[middleIndex-1] == value)
		{
			while(middleIndex > 0 && CDF[middleIndex-1] == value)
			middleIndex--;
			return middleIndex;
		}
	}
	if(CDF[middleIndex] > value)
	return findIndexBin(CDF, beginIndex, middleIndex+1, value);
	return findIndexBin(CDF, middleIndex-1, endIndex, value);
}
/**
* The implementation of the particle filter using OpenMP for many frames
* @see http://openmp.org/wp/
* @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the target matrix and the x and y arrays as arguments and returns the likelihoods
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
* @param target The matrix containing the offsets to be used in the likelihood function
*/
void particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, double * x_loc, double * y_loc, double xe, double ye, mxArray * target){
	int max_size = IszX*IszY*Nfr;
	
	/*original particle centroid*/
	x_loc[0] = xe;
	y_loc[0] = ye;
	int x, y;

	/*initial weights are all equal (1/Nparticles)*/
	double * weights = (double *)malloc(sizeof(double)*Nparticles);
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
	}
	
	/*initial likelihood to 0.0*/
	double * likelihood = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayX = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayY = (double *)mxCalloc(Nparticles, sizeof(double));
	double * xj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * yj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * CDF = (double *)mxCalloc(Nparticles, sizeof(double));
	double * u = (double *)mxCalloc(Nparticles, sizeof(double));
	mxArray * arguments[4];
	mxArray * mxIK = mxCreateDoubleMatrix(IszX, IszY, mxREAL);
	mxArray * mxX = mxCreateDoubleMatrix(1, Nparticles, mxREAL);
	mxArray * mxY = mxCreateDoubleMatrix(1, Nparticles, mxREAL);
	double * Ik = (double *)mxCalloc(IszX*IszY, sizeof(double));
	mxArray * result = mxCreateDoubleMatrix(1, Nparticles, mxREAL);

	#pragma omp parallel for shared(arrayX, arrayY, xe, ye) private(x)
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;	
	int indX, indY;
	//speed
	double speed = .01;
	
	for(k = 1; k < Nfr; k++){
		/*apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction*/
		#pragma omp parallel for shared(arrayX, arrayY, Nparticles, seed) private(x)
		for(x = 0; x < Nparticles; x++){
			arrayX[x] += speed + 2*randn(seed, x);
			arrayY[x] += 2*randn(seed, x);
			if(arrayX[x] > IszX)
				arrayX = IszX;
			if(arrayY[x] > IszY)
				arrayY = IszY;
			if(arrayX[x] < 1)
				arrayX[x] = 1;
			if(arrayY[x] < 1)
				arrayY[x] = 1;
		}	

		//get the current image
		for(x = 0; x < IszX; x++)
		{
			for(y = 0; y < IszY; y++)
			{
				Ik[x*IszX + y] = (double)I[k*IszX*IszY + x*IszY + y];
			}
		}
		//copy arguments
		memcpy(mxGetPr(mxIK), Ik, sizeof(double)*IszX*IszY);
		memcpy(mxGetPr(mxX), arrayX, sizeof(double)*Nparticles);
		memcpy(mxGetPr(mxY), arrayY, sizeof(double)*Nparticles);
		arguments[0] = mxIK;
		arguments[1] = target;
		arguments[2] = mxX;
		arguments[3] = mxY;
		mexCallMATLAB(1, &result, 4, arguments, "GetLikelihood");
		memcpy(likelihood, result, sizeof(double)*Nparticles);

		/* update & normalize weights
		// using equation (63) of Arulampalam Tutorial*/
		#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x] * exp(likelihood[x]);
		}

		double sumWeights = 0;
		#pragma omp parallel for shared(weights) private(x) reduction(+:sumWeights)
		for(x = 0; x < Nparticles; x++){
			sumWeights += weights[x];
		}
		
		#pragma omp parallel for shared(sumWeights, weights) private(x)
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x]/sumWeights;
		}

		xe = 0;
		ye = 0;
		/* estimate the object location by expected values*/
		#pragma omp parallel for private(x) reduction(+:xe, ye)
		for(x = 0; x < Nparticles; x++){
			xe += arrayX[x] * weights[x];
			ye += arrayY[x] * weights[x];
		}
		x_loc[k] = xe;
		y_loc[k] = ye;
		/*display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling*/
		
		
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}

		double u1 = (1/((double)(Nparticles)))*randu(seed, 0);
		
		#pragma omp parallel for shared(u, u1, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			u[x] = u1 + x/((double)(Nparticles));
		}

		int j, i;
		#pragma omp parallel for shared(CDF, Nparticles, xj, yj, u, arrayX, arrayY) private(i, j)
		for(j = 0; j < Nparticles; j++){
			i = findIndex(CDF, Nparticles, u[j]);
			/*i = findIndexBin(CDF, 0, Nparticles, u[j]);*/
			if(i == -1)
			i = Nparticles-1;
			xj[j] = arrayX[i];
			yj[j] = arrayY[i];
			
		}

		/*reassign arrayX and arrayY*/
		#pragma omp parallel for shared(weights, arrayX, arrayY, xj, yj, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			weights[x] = 1/((double)(Nparticles));
			arrayX[x] = xj[x];
			arrayY[x] = yj[x];
		}
	}
	mxFree(weights);	
	mxFree(likelihood);
	mxFree(arrayX);
	mxFree(arrayY);
	mxFree(CDF);
	mxFree(u);
	mxFree(xj);
	mxFree(yj);
	mxFree(Ik);
}
/**
* The implementation of the particle filter using OpenMP for a single image
* @see http://openmp.org/wp/
* @note This function is designed to work with a single image. In addition, it references a provided MATLAB function which takes the video, the target matrix and the x and y arrays as arguments and returns the likelihoods
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
* @param target The matrix containing the offsets to be used in the likelihood function
*/
void particleFilter1F(int * I, int IszX, int IszY, int * seed, int Nparticles, double * x_loc, double * y_loc, double prevX, double prevY, mxArray * target){
	int max_size = IszX*IszY;
	
	/*original particle centroid*/
	double xe = prevX;
	double ye = prevY;

	/*initial weights are all equal (1/Nparticles)*/
	double * weights = (double *)mxCalloc(Nparticles, sizeof(double));
	#pragma omp parallel for shared(weights, Nparticles) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
	}

	/*initial likelihood to 0.0*/
	double * likelihood = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayX = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayY = (double *)mxCalloc(Nparticles, sizeof(double));
	double * xj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * yj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * CDF = (double *)mxCalloc(Nparticles, sizeof(double));
	double * u = (double *)mxCalloc(Nparticles, sizeof(double));

	mxArray * arguments[4];
	mxArray * mxIK = mxCreateDoubleMatrix(IszX, IszY, mxREAL);
	mxArray * mxX = mxCreateDoubleMatrix(1, Nparticles, mxREAL);
	mxArray * mxY = mxCreateDoubleMatrix(1, Nparticles, mxREAL);
	mxArray * result = mxCreateDoubleMatrix(1, Nparticles, mxREAL);
	
	double * Ik = (double *)mxCalloc(IszX*IszY, sizeof(double));
	
	//speed
	double speed = .01;
	
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;
	int indX, indY;

	/*apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction*/
	#pragma omp parallel for shared(arrayX, arrayY, xe, ye) private(x)
	for(x = 0; x < Nparticles; x++){
		arrayX[x] += speed + 2*randn(seed, x);
		arrayY[x] += 2*randn(seed, x);
		if(arrayX[x] > IszX)
			arrayX = IszX;
		if(arrayY[x] > IszY)
			arrayY = IszY;
		if(arrayX[x] < 1)
			arrayX[x] = 1;
		if(arrayY[x] < 1)
			arrayY[x] = 1;
	}	

	/*particle filter likelihood*/
	//get the current image
	for(x = 0; x < IszX; x++)
	{
		for(y = 0; y < IszY; y++)
		{
			Ik[x*IszX + y] = (double)I[x*IszY + y];
		}
	}
	//copy arguments
	memcpy(mxGetPr(mxIK), Ik, sizeof(double)*IszX*IszY);
	memcpy(mxGetPr(mxX), arrayX, sizeof(double)*Nparticles);
	memcpy(mxGetPr(mxY), arrayY, sizeof(double)*Nparticles);
	arguments[0] = mxIK;
	arguments[1] = target;
	arguments[2] = mxX;
	arguments[3] = mxY;
	mexCallMATLAB(1, &result, 4, arguments, "GetLikelihood");
	memcpy(likelihood, result, sizeof(double)*Nparticles);

	/* update & normalize weights
		// using equation (63) of Arulampalam Tutorial*/
	#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)	
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x] * exp(likelihood[x]);
	}

	double sumWeights = 0;
	#private omp parallel for shared(weights) private(x) reduction(+:sumWeights)
	for(x = 0; x < Nparticles; x++){
		sumWeights += weights[x];
	}

	#pragma omp parallel for shared(weights) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x]/sumWeights;
	}

	xe = 0;
	ye = 0;
	/* estimate the object location by expected values*/
	#pragma omp parallel for shared(arrayX, arrayY, weights) private(x) reduction(+:xe, ye)
	for(x = 0; x < Nparticles; x++){
		xe += arrayX[x] * weights[x];
		ye += arrayY[x] * weights[x];
	}

	x_loc[0] = xe;
	y_loc[0] = ye;

	/*display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling*/
	
	
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}

	double u1 = (1/((double)(Nparticles)))*randu(seed, 0);

	#pragma omp parallel for shared(u, u1, Nparticles) private(x)
	for(x = 0; x < Nparticles; x++){
		u[x] = u1 + x/((double)(Nparticles));
	}

	int j, i;
	
	#pragma omp parallel for shared(CDF, Nparticles, xj, yj, u, arrayX, arrayY) private(i, j)
	for(j = 0; j < Nparticles; j++){
		i = findIndex(CDF, Nparticles, u[j]);
		/*i = findIndexBin(CDF, 0, Nparticles, u[j]);*/
		if(i == -1)
		i = Nparticles-1;
		xj[j] = arrayX[i];
		yj[j] = arrayY[i];
		
	}

	/*reassign arrayX and arrayY*/
	#pragma omp parallel for shared(weights, arrayX, arrayY, xj, yj, Nparticles) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
		arrayX[x] = xj[x];
		arrayY[x] = yj[x];
	}

	
	mxFree(weights);
	mxFree(likelihood);
	mxFree(arrayX);
	mxFree(arrayY);
	mxFree(CDF);
	mxFree(u);
	mxFree(xj);
	mxFree(yj);
	mxFree(Ik);
}
/**
* Function that allows the 2 particle filter implementations to be run
* @details The number of arguments provided to this function determines which function will be called. 7 args will call the video processing version. 6 (leaving out the number of frames) will call the image processing version.
* @param nlhs (Number on the Left Hand Side) The number of items to return (2 will be in this case; the x and y arrays)
* @param plhs (Parameters on the Left Hand Side) A pointer to the arrays containing the x and y arrays
* @param nrhs (Number on the Right Hand Side) The number of arguments to take in (8 are needed for video processing (The image as an unsigned char, the x dimension, the y dimension, the number of frames, the number of particles, the x starting position, the y starting position, the target matrix)
* 7 are needed for the image processing (same as before but leave out the number of frames)
* @param prhs (Parameters on the Right Hand Side) A pointer to the arrays containing the parameters
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	int * I;
	int IszX, IszY, Nfr, Nparticles;
	if(nrhs < 7)
	{
		printf("ERROR: TOO FEW ARGS HAVE BEEN ENTERED\n");
		printf("EXITING\n");
		exit(0);
	}
	else if(nrhs == 8)
	{
		IszX = (int)(mxGetScalar(prhs[1]));
		IszY = (int)(mxGetScalar(prhs[2]));
		Nfr = (int)(mxGetScalar(prhs[3]));
		Nparticles = (int)(mxGetScalar(prhs[4]));

		unsigned char * cI = (unsigned char *)mxGetData(prhs[0]);
		I = (int *)mxCalloc(IszX*IszY*Nfr, sizeof(int));
		int x, y, z;
		for(x = 0; x < IszX; x++){
			for(y = 0; y < IszY; y++){
				for(z = 0; z < Nfr; z++){
					I[x*IszY*Nfr + y*Nfr + z] = (int)cI[x*IszY*Nfr + y*Nfr + z];
				}
			}
		}
		double xe = (double)mxGetScalar(prhs[5]);
		double ye = (double)mxGetScalar(prhs[6]);
		mxArray * target = prhs[7];
		int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
		int i;
		for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
		double * posX = (double *)mxCalloc(Nfr, sizeof(double));
		double * posY = (double *)mxCalloc(Nfr, sizeof(double));

		particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, posX, posY, xe, ye, target);

		mxFree(I);
		mxFree(seed);
		
		plhs[0] = mxCreateDoubleMatrix(Nfr, 1, mxREAL);
		plhs[1] = mxCreateDoubleMatrix(Nfr, 1, mxREAL);
		double * bufferX = mxGetPr(plhs[0]);
		double * bufferY = mxGetPr(plhs[1]);
		for(i = 0; i < Nfr; i++)
		{
			bufferX[i] = posX[i];
			bufferY[i] = posY[i];
		}
		mxFree(posX);
		mxFree(posY);
	}
	else if(nrhs == 7)
	{
		IszX = (int)(mxGetScalar(prhs[1]));
		IszY = (int)(mxGetScalar(prhs[2]));
		Nparticles = (int)(mxGetScalar(prhs[3]));

		double startX = (double)mxGetScalar(prhs[4]);
		double startY = (double)mxGetScalar(prhs[5]);

		unsigned char * cI = (unsigned char *)mxGetData(prhs[0]);
		I = (int *)mxCalloc(IszX*IszY, sizeof(int));
		int x, y, z;
		for(x = 0; x < IszX; x++){
			for(y = 0; y < IszY; y++){
				I[x*IszX + y] = (int)cI[x*IszX + y];
			}
		}
		mxArray * target = prhs[6];
		int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
		int i;
		for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
		double posX[1];
		double posY[1];
		
		particleFilter1F(I, IszX, IszY, seed, Nparticles, posX, posY, startX, startY, target);
		
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
