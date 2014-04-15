#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <limits.h>
#include "mex.h"
#include "mat.h"
#define PI acos(-1)
/*M value for Linear Congruential Generator (LCG); use GCC's value*/
long M = INT_MAX;
/*A value for LCG*/
int A = 1103515245;
/*C value for LCG*/
int C = 12345;
/*****************************
*GET_TIME
*returns a long int representing the time
*****************************/
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
/* Returns the number of seconds elapsed between the two specified times */
float elapsed_time(long long start_time, long long end_time) {
	return (float) (end_time - start_time) / (1000 * 1000);
}
/*****************************
*ROUND
*takes in a double and returns an integer that approximates to that double
*if the mantissa < .5 => return value < input value
*else return value > input value
*****************************/
double roundDouble(double value){
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
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
double randu(int * seed, int index)
{
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}
/******************************
* RANDN
* GENERATES A NORMAL DISTRIBUTION
* returns a double representing random number generated using Irwin-Hall distribution method
* see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
******************************/
double randn(int * seed, int index){
	/*Box-Muller algorithm*/
	double u = randu(seed, index);
	double v = randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/******************************
* ADDNOISE
* sets values of 3D matrix using randomly generated numbers from a normal distribution
* param matrix
******************************/
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
/******************************
* VIDEO SEQUENCE
* the synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
*******************************/
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
/********************************
* CALC LIKELIHOOD SUM
* DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* param 1 I 3D matrix
* param 2 current ind array
* param 3 length of ind array
* returns a double representing the sum
********************************/
double calcLikelihoodSum(int * I, int * ind, int numOnes){
	double likelihoodSum = 0.0;
	int y;
	for(y = 0; y < numOnes; y++)
	likelihoodSum += (pow((I[ind[y]] - 100),2) - pow((I[ind[y]]-228),2))/50.0;
	return likelihoodSum;
}
/******************************
* FIND INDEX
* FINDS THE FIRST OCCURRENCE OF AN ELEMENT IN CDF GREATER THAN THE PROVIDED VALUE AND RETURNS THAT INDEX
* param1 CDF
* param2 length of CDF
* param3 value
*******************************/
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
void particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, double * x_loc, double * y_loc, double xe, double ye){
	int max_size = IszX*IszY*Nfr;
	long long start = get_time();
	/*original particle centroid*/
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
	double * objxy = (double *)mxCalloc(countOnes*2, sizeof(double));
	getneighbors(disk, countOnes, objxy, radius);
	
	long long get_neighbors = get_time();
	printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
	/*initial weights are all equal (1/Nparticles)*/
	double * weights = (double *)malloc(sizeof(double)*Nparticles);
	
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
	}
	long long get_weights = get_time();
	printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
	/*initial likelihood to 0.0*/
	double * likelihood = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayX = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayY = (double *)mxCalloc(Nparticles, sizeof(double));
	double * xj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * yj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * CDF = (double *)mxCalloc(Nparticles, sizeof(double));
	double * u = (double *)mxCalloc(Nparticles, sizeof(double));
	//int * ind = (int*)mxCalloc(countOnes*Nparticles, sizeof(int));
	mxArray * arguments[4];
	mxArray * mxIK = mxCreateDoubleMatrix(IszX, IszY, mxREAL);
	mxArray * mxObj = mxCreateDoubleMatrix(countOnes, 2, mxREAL);
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
	
	printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, get_time()));
	int indX, indY;
	for(k = 1; k < Nfr; k++){
		long long set_arrays = get_time();
		/*apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction*/
		#pragma omp parallel for shared(arrayX, arrayY, Nparticles, seed) private(x)
		for(x = 0; x < Nparticles; x++){
			arrayX[x] += 1 + 5*randn(seed, x);
			arrayY[x] += -2 + 2*randn(seed, x);
		}
		long long error = get_time();
		printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
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
		memcpy(mxGetPr(mxObj), objxy, sizeof(double)*countOnes);
		memcpy(mxGetPr(mxX), arrayX, sizeof(double)*Nparticles);
		memcpy(mxGetPr(mxY), arrayY, sizeof(double)*Nparticles);
		arguments[0] = mxIK;
		arguments[1] = mxObj;
		arguments[2] = mxX;
		arguments[3] = mxY;
		mexCallMATLAB(1, &result, 4, arguments, "GetSimpleLikelihood");
		memcpy(likelihood, result, sizeof(double)*Nparticles);
		long long likelihood_time = get_time();
		printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
		/* update & normalize weights
		// using equation (63) of Arulampalam Tutorial*/
		#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x] * exp(likelihood[x]);
		}
		long long exponential = get_time();
		printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
		double sumWeights = 0;
		#pragma omp parallel for private(x) reduction(+:sumWeights)
		for(x = 0; x < Nparticles; x++){
			sumWeights += weights[x];
		}
		long long sum_time = get_time();
		printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential, sum_time));
		#pragma omp parallel for shared(sumWeights, weights) private(x)
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x]/sumWeights;
		}
		long long normalize = get_time();
		printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time, normalize));
		xe = 0;
		ye = 0;
		/* estimate the object location by expected values*/
		#pragma omp parallel for private(x) reduction(+:xe, ye)
		for(x = 0; x < Nparticles; x++){
			xe += arrayX[x] * weights[x];
			ye += arrayY[x] * weights[x];
		}
		long long move_time = get_time();
		printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize, move_time));
		printf("XE: %lf\n", xe);
		printf("YE: %lf\n", ye);
		x_loc[k] = xe;
		y_loc[k] = ye;
		double distance = sqrt( pow((double)(xe-(int)roundDouble(IszY/2.0)),2) + pow((double)(ye-(int)roundDouble(IszX/2.0)),2) );
		printf("%lf\n", distance);
		/*display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling*/
		
		
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}
		long long cum_sum = get_time();
		printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
		double u1 = (1/((double)(Nparticles)))*randu(seed, 0);
		#pragma omp parallel for shared(u, u1, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			u[x] = u1 + x/((double)(Nparticles));
		}
		long long u_time = get_time();
		printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
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
		long long xyj_time = get_time();
		printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time, xyj_time));
		/*reassign arrayX and arrayY*/
		#pragma omp parallel for shared(weights, arrayX, arrayY, xj, yj, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			weights[x] = 1/((double)(Nparticles));
			arrayX[x] = xj[x];
			arrayY[x] = yj[x];
		}
		long long reset = get_time();
		printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
	}
	mxFree(disk);
	mxFree(weights);
	mxFree(objxy);	
	mxFree(likelihood);
	mxFree(arrayX);
	mxFree(arrayY);
	mxFree(CDF);
	mxFree(u);
	//mxFree(ind);
	mxFree(xj);
	mxFree(yj);
	mxFree(Ik);
}
void particleFilter1F(int * I, int IszX, int IszY, int * seed, int Nparticles, double * x_loc, double * y_loc, double prevX, double prevY){
	int max_size = IszX*IszY;
	long long start = get_time();
	/*original particle centroid*/
	double xe = prevX;
	double ye = prevY;
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
	double * objxy = (double *)mxCalloc(countOnes*2, sizeof(double));
	getneighbors(disk, countOnes, objxy, radius);
	
	long long get_neighbors = get_time();
	printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
	/*initial weights are all equal (1/Nparticles)*/
	double * weights = (double *)malloc(sizeof(double)*Nparticles);
	#pragma omp parallel for shared(weights, Nparticles) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
	}
	long long get_weights = get_time();
	printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
	/*initial likelihood to 0.0*/
	double * likelihood = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayX = (double *)mxCalloc(Nparticles, sizeof(double));
	double * arrayY = (double *)mxCalloc(Nparticles, sizeof(double));
	double * xj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * yj = (double *)mxCalloc(Nparticles, sizeof(double));
	double * CDF = (double *)mxCalloc(Nparticles, sizeof(double));
	double * u = (double *)mxCalloc(Nparticles, sizeof(double));
	//int * ind = (int*)mxCalloc(countOnes*Nparticles, sizeof(int));
	mxArray * arguments[4];
	mxArray * mxIK = mxCreateDoubleMatrix(IszX, IszY, mxREAL);
	mxArray * mxObj = mxCreateDoubleMatrix(countOnes, 2, mxREAL);
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
	
	printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, get_time()));
	int indX, indY;
	
	long long set_arrays = get_time();
	/*apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction*/
	#pragma omp parallel for shared(arrayX, arrayY, Nparticles, seed) private(x)
	for(x = 0; x < Nparticles; x++){
		arrayX[x] += 1 + 5*randn(seed, x);
		arrayY[x] += -2 + 2*randn(seed, x);
	}
	long long error = get_time();
	printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
	/*particle filter likelihood*/
	#pragma omp parallel for shared(likelihood, I, arrayX, arrayY, objxy, ind) private(x, y, indX, indY)
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
	memcpy(mxGetPr(mxObj), objxy, sizeof(double)*countOnes);
	memcpy(mxGetPr(mxX), arrayX, sizeof(double)*Nparticles);
	memcpy(mxGetPr(mxY), arrayY, sizeof(double)*Nparticles);
	arguments[0] = mxIK;
	arguments[1] = mxObj;
	arguments[2] = mxX;
	arguments[3] = mxY;
	mexCallMATLAB(1, &result, 4, arguments, "GetSimpleLikelihood");
	memcpy(likelihood, result, sizeof(double)*Nparticles);
	long long likelihood_time = get_time();
	printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
	/* update & normalize weights
		// using equation (63) of Arulampalam Tutorial*/
	#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x] * exp(likelihood[x]);
	}
	long long exponential = get_time();
	printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
	double sumWeights = 0;
	#pragma omp parallel for private(x) reduction(+:sumWeights)
	for(x = 0; x < Nparticles; x++){
		sumWeights += weights[x];
	}
	long long sum_time = get_time();
	printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential, sum_time));
	#pragma omp parallel for shared(sumWeights, weights) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = weights[x]/sumWeights;
	}
	long long normalize = get_time();
	printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time, normalize));
	xe = 0;
	ye = 0;
	/* estimate the object location by expected values*/
	#pragma omp parallel for private(x) reduction(+:xe, ye)
	for(x = 0; x < Nparticles; x++){
		xe += arrayX[x] * weights[x];
		ye += arrayY[x] * weights[x];
		/*printf("POSX[%d]: %lf \t WGT[%d]: %lf\n", x, arrayX[x], x, weights[x]);
		printf("POSY[%d]: %lf \t WGT[%d]: %lf\n", x, arrayY[x], x, weights[x]);*/
	}
	long long move_time = get_time();
	printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize, move_time));
	printf("XE: %lf\n", xe);
	printf("YE: %lf\n", ye);
	x_loc[0] = xe+.5;
	y_loc[0] = ye+.5;
	double distance = sqrt( pow((double)(xe-(int)roundDouble(IszY/2.0)),2) + pow((double)(ye-(int)roundDouble(IszX/2.0)),2) );
	printf("%lf\n", distance);
	/*display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling*/
	
	
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
	long long cum_sum = get_time();
	printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
	double u1 = (1/((double)(Nparticles)))*randu(seed, 0);
	#pragma omp parallel for shared(u, u1, Nparticles) private(x)
	for(x = 0; x < Nparticles; x++){
		u[x] = u1 + x/((double)(Nparticles));
	}
	long long u_time = get_time();
	printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
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
	long long xyj_time = get_time();
	printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time, xyj_time));
	/*reassign arrayX and arrayY*/
	#pragma omp parallel for shared(weights, arrayX, arrayY, xj, yj, Nparticles) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
		arrayX[x] = xj[x];
		arrayY[x] = yj[x];
	}
	long long reset = get_time();
	printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
	
	mxFree(disk);
	mxFree(weights);
	mxFree(objxy);	
	mxFree(likelihood);
	mxFree(arrayX);
	mxFree(arrayY);
	mxFree(CDF);
	mxFree(u);
	//mxFree(ind);
	mxFree(xj);
	mxFree(yj);
	mxFree(Ik);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	int * I;
	int IszX, IszY, Nfr, Nparticles;
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
		printf("ISZX: %d\n", IszX);
		printf("ISZY: %d\n", IszY);
		printf("Nfr: %d\n", Nfr);
		printf("Nparticles: %d\n", Nparticles);
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
		int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
		int i;
		for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
		double * posX = (double *)mxCalloc(Nfr, sizeof(double));
		double * posY = (double *)mxCalloc(Nfr, sizeof(double));
		long long start = get_time();
		particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, posX, posY, xe, ye);
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
			bufferX[i] = posX[i];
			bufferY[i] = posY[i];
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
		I = (int *)mxCalloc(IszX*IszY, sizeof(int));
		int x, y, z;
		for(x = 0; x < IszX; x++){
			for(y = 0; y < IszY; y++){
				I[x*IszX + y] = (int)cI[x*IszX + y];
			}
		}
		
		int * seed = (int *)mxCalloc(Nparticles, sizeof(int));
		int i;
		for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
		double posX[1];
		double posY[1];
		long long start = get_time();
		particleFilter1F(I, IszX, IszY, seed, Nparticles, posX, posY, startX, startY);
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
	/*3D matrix consisting the picture and the frames*/
	int * I;
	/*dimension X of the picture in pixels*/
	int IszX = 128;
	/*dimension Y of the picture in pixels*/
	int IszY = 128;
	/*number of frames*/
	int Nfr = 10;
	/*define number of particles*/
	int Nparticles = 100000;
	/*establish seed*/
	int * seed = (int *)malloc(sizeof(int)*Nparticles);
	int i;
	for(i = 0; i < Nparticles; i++)
	seed[i] = time(0)*i;
	/*malloc matrix*/
	I = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
	long long start = get_time();
	/*call video sequence*/
	videoSequence(I, IszX, IszY, Nfr, seed);
	long long endVideoSequence = get_time();
	printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
	double * posX = (double *)mxCalloc(Nfr, sizeof(double));
	double * posY = (double *)mxCalloc(Nfr, sizeof(double));
	double xe = IszX/2.0;
	double ye = IszY/2.0;
	/*call particle filter*/
	particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, posX, posY, xe, ye);
	free(I);
	free(seed);
	long long endParticleFilter = get_time();
	printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
	printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));
	
	return 0;
}
