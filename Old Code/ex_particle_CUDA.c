#include <stdlib.h>
#include <math.h>
#define PI acos(-1)
#define BLOCK_X 16
#define BLOCK_Y 16
//3D matrix consisting the picture and the frames
int * I;
//dimension X of the picture in pixels
int IszX = 128;
//dimension Y of the picture in pixels
int IszY = 128;
//number of frames
int Nfr = 20;

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
/*****************************
* CUDA Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: Nparticles
*****************************/
/*__gloabl__ void kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles){
	int i = blockDim.y * blockIdx.y + threadId.x;
	int index = -1;
	int x;

	for(x = 0; x < Nparticles; x++){
		if(CDF[x] >= u[i]){
			index = x;
			break;
		}
	}
	if(index == -1){
		index = Nparticles-1;
	}
	
	xj[i] = arrayX[index];
	yj[i] = arrayY[index];
}*/
/*****************************
*ROUND
*takes in a double and returns an integer that approximates to that double
*if the mantissa < .5 => return value < input value
*else return value > input value
*****************************/
double round(double value){
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
	for(z = 0; z < *dimZ; z++){
		for(y = 0; y < *dimY; y++){
			for(x = 0; x < *dimX; x++){
				if(array3D[z * *dimY * *dimX+ y * *dimX + x] == testValue)
					array3D[z * *dimY * *dimX + y * *dimX + x] = newValue;
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
/******************************
* ADDNOISE
* sets values of 3D matrix using randomly generated numbers from a normal distribution
* param matrix
******************************/
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ){
	int x, y, z;
	for(z= 0; z < *dimZ; z++){
		for(y = 0; y < *dimY; y++){
			for(x = 0; x < *dimX; x++){
				array3D[z * *dimY * *dimX + y * *dimX + x] = array3D[z * *dimY * *dimX + y * *dimX + x] + (int)(5*randn());
			}
		}
	}
}
/******************************
* STRELDISK
* creates a 9x9 matrix representing the disk
******************************/
int * strelDisk(){
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
void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error){
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
	for(y = startY; y < endY; y++){
		for(x = startX; x < endX; x++){
			double distance = sqrt( pow((double)(x-posX),2) + pow((double)(y-posY),2) );
			if(distance < error)
				matrix[posZ*dimX*dimY + y*dimX + x] = 1;
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
int* imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error){
	int * newMatrix = (int *)malloc(sizeof(int)*dimX*dimY*dimZ);
	int x, y, z;
	for(z = 0; z < dimZ; z++){
		for(y = 0; y < dimY; y++){
			for(x = 0; x < dimX; x++){
				if(matrix[z*dimY*dimX + y*dimX + x] == 1){
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
	//get object centers
	int x0 = (int)round(IszY/2.0);
	int y0 = (int)round(IszX/2.0);
	I[y0 * IszX + x0] = 1;
	
	//move point
	for(k = 1; k < Nfr; k++){
		int xk = x0 + (k-1);
		int yk = x0 - 2*(k-1);
		I[k*IszY*IszX + yk*IszX + xk] = 1;
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
		likelihoodSum += (pow((I[ind[x]] - 100),2) - pow((I[ind[x]]-228),2))/100.0;
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
void particleFilter(){
	//define number of particles
	int Nparticles = 100;
	
	//original particle centroid
	double xe = round(IszY/2.0);
	double ye = round(IszX/2.0);
	
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
	double * weights = (double *)malloc(sizeof(double)*Nparticles);
	for(x = 0; x < Nparticles; x++){
		weights[x] = 1/((double)(Nparticles));
	}
	//initial likelihood to 0.0
	double * likelihood = (double *)malloc(sizeof(double)*Nparticles);
	
	double * arrayX = (double *)malloc(sizeof(double)*Nparticles);
	double * arrayY = (double *)malloc(sizeof(double)*Nparticles);
	double * xj = (double *)malloc(sizeof(double)*Nparticles);
	double * yj = (double *)malloc(sizeof(double)*Nparticles);
	double * CDF = (double *)malloc(sizeof(int)*Nparticles);
	
	//GPU copies of arrays
	double * arrayX_GPU;
	double * arrayY_GPU;
	double * xj_GPU;
	double * yj_GPU;
	double * CDF_GPU;
	
	int * ind = malloc(sizeof(int)*countOnes);
	for(x = 0; x < Nparticles; x++){
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;
	//double * Ik = (double *)malloc(sizeof(double)*IszX*IszY);
	int indX, indY;
	for(k = 1; k < Nfr; k++){
		
		//apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction
		for(x = 0; x < Nparticles; x++){
			arrayX[x] = arrayX[x] + 1 + 5*randn();
			arrayY[x] = arrayY[x] - 2 + 2*randn();
		}
		//particle filter likelihood
		for(x = 0; x < Nparticles; x++){
		
			//compute the likelihood: remember our assumption is that you know
			// foreground and the background image intensity distribution.
			// Notice that we consider here a likelihood ratio, instead of
			// p(z|x). It is possible in this case. why? a hometask for you.
			
			//calc ind
			for(y = 0; y < countOnes; y++){
				indX = round(arrayX[x]) + objxy[y*2 + 1];
				indY = round(arrayY[x]) + objxy[y*2];
				ind[y] = fabs(k*IszY*IszX + indY*IszX + indX);
			}
			likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
			likelihood[x] = likelihood[x]/countOnes;
		}
		// update & normalize weights
		// using equation (63) of Arulampalam Tutorial
		for(x = 0; x < Nparticles; x++){
			weights[x] = weights[x] * exp(likelihood[x]);
			if(weights[x] == 0)
				printf("WEIGHT[%d] = 0\n", x);
		}
		double sumWeights = 0;
		for(x = 0; x < Nparticles; x++){
			sumWeights += weights[x];
		}
		for(x = 0; x < Nparticles; x++){
			if(sumWeights != 0)
				weights[x] = weights[x]/sumWeights;
			else
				weights[x] = 0;
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
		double distance = sqrt( pow((double)(xe-(int)round(IszY/2.0)),2) + pow((double)(ye-(int)round(IszX/2.0)),2) );
		printf("%lf\n", distance);
		//display(hold off for now)
		
		//pause(hold off for now)
		
		//resampling
		
		
		CDF[0] = weights[0];
		for(x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}
		
		double u1 = (1/((double)(Nparticles)))*randn();
		double * u = (double *)malloc(sizeof(double)*Nparticles);
		double * u_GPU;
		
		for(x = 0; x < Nparticles; x++){
			u[x] = u1 + x/((double)(Nparticles));
		}
		
/*		
		//CUDA memory allocation
		check_error(cudaMalloc((void **) &arrayX_GPU, sizeof(double)*Nparticles));
		check_error(cudaMalloc((void **) &arrayY_GPU, sizeof(double)*Nparticles));
		check_error(cudaMalloc((void **) &xj_GPU, sizeof(double)*Nparticles));
		check_error(cudaMalloc((void **) &yj_GPU, sizeof(double)*Nparticles));
		check_error(cudaMalloc((void **) &CDF_GPU, sizeof(int)*Nparticles));
		check_error(cudaMalloc((void **) &u_GPU, sizeof(double)*Nparticles));
		
		//CUDA memory copying from CPU memory to GPU memory
		cudaMemcpy(arrayX_GPU, arrayX, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(arrayY_GPU, arrayY, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(xj_GPU, xj, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(yj_GPU, yj, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(CDF_GPU, CDF, sizeof(int)*Nparticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(u_GPU, u, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
		
		//Set number of blocks in grid
		dim3 block(16, 16);
		int num_blocks_x = (10 + block.x - 1) / block.x;
		int num_blocks_y = (10 + block.y - 1) / block.y;
		dim3 grid(num_blocks_x, num_blocks_y);
		
		//KERNEL FUNCTION CALL
		kernel << grid, block >> (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, Nparticles);
		
		//CUDA memory copying back from GPU to CPU memory
		cudaMemcpy(arrayY, yj_GPU, sizeof(double)*Nparticles), cudaMemcpyHostToDevice);
		cudaMemcpy(arrayX, xj_GPU, sizeof(double)*Nparticles), cudaMemcpyHostToDevice);
		
		//CUDA freeing of memory
		cudaFree(u_GPU);
		cudaFree(CDF_GPU);
		cudaFree(xj_GPU);
		cudaFree(xi_GPU);
		cudaFree(arrayY_GPU);
		cudaFree(arrayX_GPU);
*/
		/**
		int j, i;
		
		for(j = 0; j < Nparticles; j++){
			i = findIndex(CDF, Nparticles, u[j]);
			xj[j] = arrayX[i];
			yj[j] = arrayY[i];
			
		}

		//reassign arrayX and arrayY
		arrayX = xj;
		arrayY = yj;
		**/
		
		for(x = 0; x < Nparticles; x++){
			weights[x] = 1/((double)(Nparticles));
		}
	}
}
int main(){
	//establish seed
	srand(time(0));
	//malloc matrix
	I = (int *)malloc(sizeof(int)*Nfr*IszY*IszX);
	//call video sequence
	videoSequence();
	//call particle filter
	particleFilter();
	return 0;
}
