#ifndef CUDA_MODULE
#define CUDA_MODULE

#include <iostream>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

//glm
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

// cuda 
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "CalculationModule.h"

#include "VertexBuffer.h"
#include "VertexBufferLayout.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"


// error handling
#define CudaCall(x) do {\
		cudaError_t err = x;\
		if (err != cudaSuccess) {\
			const char* errname = cudaGetErrorName(err);\
			const char* errdesc = cudaGetErrorString(err);\
			printf("ERROR, file: %s, line: %d: Cuda call failed: %s: %s \n", __FILE__, __LINE__, errname, errdesc);\
			exit(-1);\
		}\
	}\
	while(0);\


class CudaModule : public CalculationModule
{
private:
	unsigned int blocksCount;
	unsigned int threadsCount;
	unsigned int boidsCount;

	// mode
	int mode;

	//borders
	float borderL;
	float borderR;
	float borderT;
	float borderB;

	// boids' cooridnates (size = 6 * number of boids) 
	// 0 - x, 1 - y, 2 - vx, 3 - vy, 4 - cell, 5 - boid type
	float* h_boids;
	float* d_boids;

	// indexes for boids index buffer  (size = number of boids)
	unsigned int* boidsIndices;

	// boids calculation arrays (size = 2* number of boids)
	// 0 - x, 1 - y
	float* h_separation;
	float* d_separation;
	float* h_alignment;
	float* d_alignment;
	float* h_cohesion;
	float* d_cohesion;

	// for grid & sorting purposes
	unsigned int gridRows;
	unsigned int gridCols;
	int* d_boidsIndices;
	int* d_cells;
	int* d_cellsStarts; // indexes of table (d_boidsIndice, cell)[] (sorted by cells) where the ith cell starts
	int* d_cellsEnds;

	// cuda GraphicsResouces
	cudaGraphicsResource_t cudaResourceBufBoids;

	//openGL
	VertexArray* va;
	VertexBuffer* vb;
	VertexBufferLayout layout;
	IndexBuffer* ib;
	Shader* shader;
	Renderer* renderer;

public:
	CudaModule(unsigned int blocksCount, unsigned int threadsCount, unsigned int gridRows, unsigned int gridCols, float borderL, float borderR, float borderT, float borderB, int mode);
	~CudaModule();
	virtual void Draw() override;
	void CalculateAndUpdate();

private:
	void InitBoids();
	void PrintTable(float* tab, int size);
	void PrintTableI(int* tab, int size);
};

#endif 
