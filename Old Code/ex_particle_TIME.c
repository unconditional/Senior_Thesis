#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#define PI acos(-1)
//3D matrix consisting the picture and the frames
int * I;
//dimension X of the picture in pixels
int IszX = 128;
//dimension Y of the picture in pixels
int IszY = 128;
//number of frames
int Nfr = 10;
//define number of particles
int Nparticles = 10000;
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
	//#pragma omp parallel for
	for(x = 0; x < *dimX; x++){
		//#pragma omp parallel for
		for(y = 0; y < *dimY; y++){
			//#pragma omp parallel for
			for(z = 0; z < *dimZ; z++){
				if(array3D[x * *dimY * *dimZ+y * *dimZ + z] == testValue)
					array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
			}
		}
	}
}
/******************************
* RANDN
* GENERATES A NORMAL DISTRIBUTION
* returns a double representing random number generated using Irwin-Hall distribution method
* see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
******************************/
double randn(){
	//Box-Muller algortihm
	double u = (double)rand()/(RAND_MAX);
	double v = (double)rand()/(RAND_MAX);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
double randu()
{
	double max = (double)RAND_MAX;
	int num = rand();
	return num/max;
}
/******************************
* ADDNOISE
* sets values of 3D matrix using randomly generated numbers from a normal distribution
* param matrix
******************************/
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ){
	int x, y, z;
	//#pragma omp parallel for
	for(x = 0; x < *dimX; x++){
		//#pragma omp parallel for
		for(y = 0; y < *dimY; y++){
			//#pragma omp parallel for
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
			double distance = sqrt(pow((double)(x-4),2) + pow((double)(y-4),2));
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
int* imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error)
{
	int * newMatrix = (int *)malloc(sizeof(int)*dimX*dimY*dimZ);
	int x, y, z;
	//#pragma omp parallel for
	for(z = 0; z < dimZ; z++){
		//#pragma omp parallel for
		for(x = 0; x < dimX; x++){
			//#pragma omp parallel for
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
		xk = abs(x0 + (k-1));
		yk = abs(y0 - 2*(k-1));
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
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((I[ind[x]] - 100),2) - pow((I[ind[x]]-228),2))/50.0;
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
		if(CDF[x] >= value && index == -1){
			index = x;
		}
	}
	if(index == -1){
		return lengthCDF-1;
	}
	return index;
}
void particleFilter(){
	int max_size = IszX*IszY*Nfr;
	long long start = get_time();
	//original particle centroid
	double xe = roundDouble(IszY/2.0);
	double ye = roundDouble(IszX/2.0);
	
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
	long long get_neighbors = get_time();
	printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
	//initial weights are all equal (1/Nparticles)
	double * weights = (double *)malloc(sizeof(double)*Nparticles);
	//#pragma omp parallel for shared(weights, Nparticles) private(x)
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
	}
	long long get_weights = get_time();
	printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
	//initial likelihood to 0.0
	double * likelihood = (double *)malloc(sizeof(double)*Nparticles);
	double * arrayX = (double *)malloc(sizeof(double)*Nparticles);
	double * arrayY = (double *)malloc(sizeof(double)*Nparticles);
	double * xj = (double *)malloc(sizeof(double)*Nparticles);
	double * yj = (double *)malloc(sizeof(double)*Nparticles);
	double * CDF = (double *)malloc(sizeof(int)*Nparticles);
	int * ind = (int*)malloc(sizeof(int)*countOnes);
	double * u = (double *)malloc(sizeof(double)*Nparticles);
	//#pragma omp parallel for shared(arrayX, arrayY, xe, ye) private(x)
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;
	//double * Ik = (double *)malloc(sizeof(double)*IszX*IszY);
	int indX, indY;
	for(k = 1; k < Nfr; k++){
		long long set_arrays = get_time();
		printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, set_arrays));
		//apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction
		//TODO #pragma omp parallel for shared(arrayX, arrayY, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			arrayX[x] += 1 + 5*randn();
			arrayY[x] += -2 + 2*randn();
		}
		long long error = get_time();
		printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
		//particle filter likelihood
		//TODO parallelize HERE
		//#pragma omp parallel for shared(likelihood, I, ind, countOnes) private (x)
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
		long long likelihood_time = get_time();
		printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
		// update & normalize weights
		// using equation (63) of Arulampalam Tutorial
		
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x] * exp(likelihood[x]);
		}
		long long exponential = get_time();
		printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
		double sumWeights = 0;
		
		for(x = 0; x < Nparticles; x++){
			sumWeights += weights[x];
		}
		long long sum_time = get_time();
		printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential, sum_time));
		
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x]/sumWeights;
		}
		long long normalize = get_time();
		printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time, normalize));
		xe = 0;
		ye = 0;
		// estimate the object location by expected values
		
		for(x = 0; x < Nparticles; x++){
			xe += arrayX[x] * weights[x];
			ye += arrayY[x] * weights[x];
		}
		long long move_time = get_time();
		printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize, move_time));
		printf("XE: %lf\n", xe);
		printf("YE: %lf\n", ye);
		double distance = sqrt( pow((double)(xe-(int)roundDouble(IszY/2.0)),2) + pow((double)(ye-(int)roundDouble(IszX/2.0)),2) );
		printf("%lf\n", distance);
		//display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling
		
		
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}
		long long cum_sum = get_time();
		printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
		double u1 = (1/((double)(Nparticles)))*randu();
		//#pragma omp parallel for shared(u, u1, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			u[x] = u1 + x/((double)(Nparticles));
		}
		long long u_time = get_time();
		printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
		int j, i;
		
		for(j = 0; j < Nparticles; j++){
			i = findIndex(CDF, Nparticles, u[j]);
			xj[j] = arrayX[i];
			yj[j] = arrayY[i];
			
		}
		long long xyj_time = get_time();
		printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time, xyj_time));
		//reassign arrayX and arrayY
		arrayX = xj;
		arrayY = yj;
		//#pragma omp parallel for shared(weights, Nparticles) private(x)
		for(x = 0; x < Nparticles; x++){
			weights[x] = 1/((double)(Nparticles));
		}
		long long reset = get_time();
		printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
	}
}
int main(){
	//establish seed
	srand(time(0));
	//malloc matrix
	I = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
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
