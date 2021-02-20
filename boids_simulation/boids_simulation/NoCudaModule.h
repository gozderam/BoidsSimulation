#ifndef NO_CUDA_MODULE
#define NO_CUDA_MODULE

#include <iostream>
#include <chrono>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

//glm
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "CalculationModule.h"

#include "VertexBuffer.h"
#include "VertexBufferLayout.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"



class NoCudaModule :  public CalculationModule
{
private:
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

	// indexes for boids index buffer  (size = number of boids)
	unsigned int* boidsIndices;

	// boids calculation arrays (size = 2* number of boids)
	// 0 - x, 1 - y
	float* h_separation;
	float* h_alignment;
	float* h_cohesion;

	// for grid & sorting purposes
	unsigned int gridRows;
	unsigned int gridCols;
	int* h_boidsIndices;
	int* h_cells;
	int* h_cellsStarts; // indexes of table (d_boidsIndice, cell)[] (sorted by cells) where the ith cell starts
	int* h_cellsEnds;

	//openGL
	VertexArray* va;
	VertexBuffer* vb;
	VertexBufferLayout layout;
	IndexBuffer* ib;
	Shader* shader;
	Renderer* renderer;

public:
	NoCudaModule( unsigned int boidsCount, unsigned int gridRows, unsigned int gridCols, float borderL, float borderR, float borderT, float borderB, int mode);
	~NoCudaModule();
	virtual void Draw() override;
	void CalculateAndUpdate();

private:
	void InitBoids();
};

#endif

