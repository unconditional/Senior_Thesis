#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

#define PI acos(-1)
/*M value for Linear Congruential Generator (LCG); use GCC's value*/
long M = INT_MAX;
/*A value for LCG*/
int A = 1103515245;
/*C value for LCG*/
int C = 12345;

using namespace std;

/**********************************
* GET_TIME
* returns a long int representing the time
**********************************/
long long get_time(){
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
void setIf(int testValue, int newValue, thrust::host_vector<int> array3D, int dimX, int dimY, int dimZ)
{
	int x, y, z;
	for(x = 0; x < dimX; x++){
		for(y = 0; y < dimY; y++){
			for(z = 0; z < dimZ; z++){
				if(array3D[x*dimY*dimZ + y*dimZ + z] == testValue)
					array3D[x*dimY*dimZ + y*dimZ + z] = newValue;
			}
		}
	}
}
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
double host_randu(thrust::host_vector<int> seed, int index)
{
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}
/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
double device_randu(thrust::device_vector<int> seed, int index)
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
double host_randn(thrust::host_vector<int> seed, int index){
	/*Box-Muller algorithm*/
	double u = host_randu(seed, index);
	double v = host_randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/******************************
* RANDN
* GENERATES A NORMAL DISTRIBUTION
* returns a double representing random number generated using Irwin-Hall distribution method
* see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
******************************/
double device_randn(thrust::device_vector<int> seed, int index){
	/*Box-Muller algorithm*/
	double u = device_randu(seed, index);
	double v = device_randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/******************************
* ADDNOISE
* sets values of 3D matrix using randomly generated numbers from a normal distribution
* param matrix
******************************/
void addNoise(thrust::host_vector<int> array3D, int dimX, int dimY, int dimZ, thrust::host_vector<int> seed){
	int x, y, z;
	for(x = 0; x < dimX; x++){
		for(y = 0; y < dimY; y++){
			for(z = 0; z < dimZ; z++){
				array3D[x * dimY * dimZ + y * dimZ + z] = array3D[x * dimY * dimZ + y * dimZ + z] + (int)(5*host_randn(seed, 0));
			}
		}
	}
}
/******************************
* STRELDISK
* param: pointer to the disk to be made 
* creates a 9x9 matrix representing the disk
******************************/
void strelDisk(thrust::host_vector<int> disk, int radius)
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
void dilate_matrix(thrust::host_vector<int> matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
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
void imdilate_disk(thrust::host_vector<int> matrix, int dimX, int dimY, int dimZ, int error, thrust::host_vector<int> newMatrix)
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
void getneighbors(thrust::host_vector<int> se, int numOnes, thrust::host_vector<int> neighbors, int radius){
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
void videoSequence(thrust::host_vector<int> I, int IszX, int IszY, int Nfr, thrust::host_vector<int> seed){
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
	
	thrust::host_vector<int> newMatrix(IszX*IszY*Nfr);
	imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
	int x, y;
	for(x = 0; x < IszX; x++){
		for(y = 0; y < IszY; y++){
			for(k = 0; k < Nfr; k++){
				I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
			}
		}
	}
	
	
	/*define background, add noise*/
	setIf(0, 100, I, IszX, IszY, Nfr);
	setIf(1, 228, I, IszX, IszY, Nfr);
	/*add noise*/
	addNoise(I, IszX, IszY, Nfr, seed);

}
/********************************
* CALC LIKELIHOOD SUM
* DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* param 1 I 3D matrix
* param 2 current ind array
* param 3 length of ind array
* returns a double representing the sum
********************************/
double calcLikelihoodSum(thrust::device_vector<int> I, thrust::device_vector<int> ind, int numOnes){
	double likelihoodSum = 0.0;
	int y;
	for(y = 0; y < numOnes; y++)
		likelihoodSum += (pow((double)(I[ind[y]] - 100),2) - pow((double)(I[ind[y]]-228),2))/50.0;
	return likelihoodSum;
}
/******************************
* FIND INDEX
* FINDS THE FIRST OCCURRENCE OF AN ELEMENT IN CDF GREATER THAN THE PROVIDED VALUE AND RETURNS THAT INDEX
* param1 CDF
* param2 length of CDF
* param3 value
*******************************/
int findIndex(thrust::device_vector<double> CDF, int lengthCDF, double value){
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
void particleFilter(thrust::device_vector<int> I, int IszX, int IszY, int Nfr, thrust::device_vector<int> seed, int Nparticles)
{
	int max_size = IszX*IszY*Nfr;
	long long start = get_time();
	/*original particle centroid*/
	double xe = roundDouble(IszY/2.0);
	double ye = roundDouble(IszX/2.0);
	/*expected object locations, compared to center*/
	int radius = 5;
	int diameter = radius*2 -1;
	thrust::host_vector<int> host_disk(diameter*diameter);
	strelDisk(host_disk, radius);
	int countOnes = 0;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(host_disk[x*diameter + y] == 1)
				countOnes++;
		}
	}
	thrust::host_vector<int> objxy_host(countOnes*2);
	getneighbors(host_disk, countOnes, objxy_host, radius);
	thrust::device_vector<int> objxy = objxy_host;
	long long get_neighbors = get_time();
	//printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
	/*initial weights are all equal (1/Nparticles)*/
	thrust::device_vector<double> weights(Nparticles, (1/((double)Nparticles)));
	long long get_weights = get_time();
	printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
	thrust::device_vector<double> likelihood(Nparticles);
	thrust::device_vector<double> arrayX(Nparticles, xe);
	thrust::device_vector<double> arrayY(Nparticles, ye);
	thrust::device_vector<double> xj(Nparticles);
	thrust::device_vector<double> yj(Nparticles);
	thrust::device_vector<double> CDF(Nparticles);
	thrust::device_vector<double> u(Nparticles);
	thrust::device_vector<int> ind(Nparticles*countOnes);
	int k;
	printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, get_time()));
	int indX, indY;
	for(k = 1; k < Nfr; k++){
		long long set_arrays = get_time();
		//apply motion model
		for(x = 0; x < Nparticles; x++){
			arrayX[x] += 1 + 5*device_randn(seed, x);
			arrayY[x] += -2 + 2*device_randn(seed, x);
		}
	
		long long error = get_time();
		printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
		for(x = 0; x < Nparticles; x++){
			/*compute the likelihood: remember our assumption is that you know
			// foreground and the background image intensity distribution.
			// Notice that we consider here a likelihood ratio, instead of
			// p(z|x). It is possible in this case. why? a hometask for you.		
			//calc ind*/
			for(y = 0; y < countOnes; y++){
				indX = round(arrayX[x]) + objxy[y*2 + 1];
				indY = round(arrayY[x]) + objxy[y*2];
				ind[x*countOnes + y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
				if(ind[x*countOnes + y] >= max_size)
					ind[x*countOnes + y] = 0;
			}
			likelihood[x] = 0;
			for(y = 0; y < countOnes; y++)
				likelihood[x] += (pow((double)(I[ind[x*countOnes + y]] - 100),2) - pow((double)(I[ind[x*countOnes + y]]-228),2))/50.0;
			likelihood[x] = likelihood[x]/((double) countOnes);
		}
		long long likelihood_time = get_time();
		printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
		/* update & normalize weights
		// using equation (63) of Arulampalam Tutorial*/
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x] * exp(likelihood[x]);
		}
		long long exponential = get_time();
		printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
		double sumWeights = thrust::reduce(weights.begin(), weights.end());
		long long normalize = get_time();
		
		
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}
		
		long long cum_sum = get_time();
		printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(normalize, cum_sum));
		double u1 = (1/((double)(Nparticles)))*device_randu(seed, 0);
		for(x = 0; x < Nparticles; x++){
			u[x] = u1 + x/((double)(Nparticles));
		}
		long long u_time = get_time();
		printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
		int j, i;
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
		for(x = 0; x < Nparticles; x++){
			weights[x] = 1/((double)(Nparticles));
			arrayX[x] = xj[x];
			arrayY[x] = yj[x];
		}
		long long reset = get_time();
		printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
	}
	
	thrust::host_vector<int> host_arrayX = arrayX;
	thrust::host_vector<int> host_arrayY = arrayY;
	thrust::host_vector<int> host_weights = weights;
	
	xe = 0;
	ye = 0;
	for(x = 0; x < Nparticles; x++){
		xe += host_arrayX[x] * host_weights[x];
		ye += host_arrayY[x] * host_weights[x];
	}
	printf("XE: %lf\n", xe);
	printf("YE: %lf\n", ye);
	float distance = sqrt( pow((float)(xe-(int)roundDouble(IszY/2.0)),2) + pow((float)(ye-(int)roundDouble(IszX/2.0)),2) );
	printf("%lf\n", distance);
}
int main(){
	/*dimension X of the picture in pixels*/
	int IszX = 128;
	/*dimension Y of the picture in pixels*/
	int IszY = 128;
	/*number of frames*/
	int Nfr = 10;
	/*define number of particles*/
	int Nparticles = 1000;
	
	/*establish seed*/
	thrust::host_vector<int> seed(Nparticles);
	int i;
	for(i = 0; i < Nparticles; i++)
		seed[i] = time(0)*i;
	/*3D matrix consisting the picture and the frames*/
	thrust::host_vector<int> I(IszX*IszY*Nfr);
	long long start = get_time();
	/*call video sequence*/
	videoSequence(I, IszX, IszY, Nfr, seed);
	long long endVideoSequence = get_time();
	printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
	thrust::device_vector<int> device_I = I;
	thrust::device_vector<int> device_seed = seed;
	/*call particle filter*/
	particleFilter(device_I, IszX, IszY, Nfr, device_seed, Nparticles);
	
	
	long long endParticleFilter = get_time();
	printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
	printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));
	return 0;
}
