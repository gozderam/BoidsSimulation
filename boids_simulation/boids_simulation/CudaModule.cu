#include "cudaModule.h"
#include <time.h>
#define TM

// cuda kernels

// i - index of boid we make calculations for, cellIdx - index of cell we make calculations in
__device__ void calculateWithinCell_kernel(
	int i, int cellIdx,
	float* d_boids, int* d_boidsIndices, int* d_cellsStarts, int* d_cellsEnds,
	int* sepa_count, float sepa_f, float* sepa_x, float* sepa_y,
	int* alig_count, float alig_f, float* align_x, float* align_y,
	int* cohe_count, float cohe_f, float* cohe_x, float* cohe_y)
{
	// from index in (boidsIices, cell)[] where cells beigns to index where it ends
	for (int jj = d_cellsStarts[cellIdx]; jj <= d_cellsEnds[cellIdx]; jj++)
	{
		int j = d_boidsIndices[jj];
		if (i == j)
			continue;

		float dist_square = ((d_boids[6 * i] - d_boids[6 * j])*(d_boids[6 * i] - d_boids[6 * j])) 
		+((d_boids[6 * i + 1] - d_boids[6 * j + 1])*(d_boids[6 * i + 1] - d_boids[6 * j + 1]));

		// sep += i -j / dist
		if (dist_square < sepa_f*sepa_f && dist_square >0)
		{
			float dist = sqrt(dist_square);
			(*sepa_count)++;
			(*sepa_x) += (d_boids[i * 6] - d_boids[6 * j]) / dist;// x
			(*sepa_y) += (d_boids[i * 6 + 1] - d_boids[6 * j + 1]) / dist;// y
		}

		// align += j.v
		if (dist_square < alig_f*alig_f && dist_square >0 )
		{
			(*alig_count)++;
			(*align_x) += d_boids[6 * j + 2]; // vx
			(*align_y) += d_boids[6 * j + 3]; // vy
		}

		//coh += j
		if (dist_square < cohe_f * cohe_f && dist_square >0 )
		{
			(*cohe_count)++;
			(*cohe_x) += d_boids[6 * j]; // x
			(*cohe_y) += d_boids[6 * j + 1]; // y
		}
	}
}

__global__ void calculatePositions_kernel(float* d_boids, int boidsCount, int* d_boidsIndices, int* d_cells, int*d_cellsStarts, int* d_cellsEnds, int gridRows, int gridCols, float* d_separation, float* d_alignment, float* d_cohesion, int mode, float borderL, float borderR)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; //index of boid 
	if (i >= boidsCount)
		return;

	int sepa_count = 0;
	int alig_count = 0;
	int cohe_count = 0;

	float cohe_f = (borderR-borderL) / float(gridCols);
	float alig_f = 0.7*cohe_f;
	float sepa_f = 0.4*cohe_f;

	float sepa_x = 0.0;
	float sepa_y = 0.0;
	float align_x = 0.0;
	float align_y = 0.0;
	float cohe_x = 0.0;
	float cohe_y = 0.0;

	// cellIdx (the same cell)
	int cellIdx = d_boids[6 * i + 4];
	calculateWithinCell_kernel(
		i, cellIdx,
		d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
		&sepa_count, sepa_f, &sepa_x,  &sepa_y,
		&alig_count, alig_f, &align_x, &align_y,
		&cohe_count, cohe_f, &cohe_x,  &cohe_y);

	if (mode == 1)
	{
		//x-1, y-1
		cellIdx = d_boids[6 * i + 4] - gridCols - 1;
		if (cellIdx >= 0)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);

		//x, y-1
		cellIdx = d_boids[6 * i + 4] - gridCols;
		if (cellIdx >= 0)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);

		//x+1, y+1
		cellIdx = d_boids[6 * i + 4] - gridCols + 1;
		if (cellIdx >= 0)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);

		//x-1, y
		cellIdx = d_boids[6 * i + 4] - 1;
		if (cellIdx >= 0)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);

		//x+1, y
		cellIdx = d_boids[6 * i + 4] + 1;
		if (cellIdx < gridCols*gridRows)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);

		//x-1, y+1
		cellIdx = d_boids[6 * i + 4] + gridCols - 1;
		if (cellIdx < gridCols*gridRows)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);

		//x, y+1
		cellIdx = d_boids[6 * i + 4] + gridCols;
		if (cellIdx < gridCols*gridRows)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);

		//x+1, y+1
		cellIdx = d_boids[6 * i + 4] + gridCols + 1;
		if (cellIdx < gridCols*gridRows)
			calculateWithinCell_kernel(
				i, cellIdx,
				d_boids, d_boidsIndices, d_cellsStarts, d_cellsEnds,
				&sepa_count, sepa_f, &sepa_x, &sepa_y,
				&alig_count, alig_f, &align_x, &align_y,
				&cohe_count, cohe_f, &cohe_x, &cohe_y);
	}

	// separation
	if (sepa_count > 0)
	{
		float mod_s = sqrt(sepa_x*sepa_x + sepa_y * sepa_y);
		if (mod_s == 0)
		{
			d_separation[2 * i] = 0;
			d_separation[2 * i + 1] = 0;
		}
		else
		{
			d_separation[2 * i] = sepa_x / mod_s - d_boids[6 * i + 2];// sep_x = sep_x/mod + i.vx 
			d_separation[2 * i + 1] = sepa_y / mod_s - d_boids[6 * i + 3];// sep_y = sep_y/mod + i.vy 
		}
	}
	

	// alignment
	if (alig_count > 0)
	{
		float mod_a = sqrt(align_x*align_x + align_y*align_y);
		if (mod_a == 0)
		{
			d_alignment[2 * i] = 0;
			d_alignment[2 * i + 1] = 0;
		}
		else
		{
			d_alignment[2 * i] = align_x / mod_a - d_boids[6 * i + 2]; //alig_x = alig_x.mod + i.vx
			d_alignment[2 * i + 1] = align_y / mod_a - d_boids[6 * i + 3]; //alig_y = alig_y.mod + i.vy
		}
	}
	

	//cohesion
	if (cohe_count > 0)
	{
		cohe_x = cohe_x / cohe_count - d_boids[6 * i];
		cohe_y = cohe_y / cohe_count - d_boids[6 * i+1];


		float mod_c = sqrt(cohe_x*cohe_x + cohe_y*cohe_y);
		if (mod_c == 0)
		{
			d_cohesion[2 * i] = 0;
			d_cohesion[2 * i + 1] = 0;
		}
		else
		{
			d_cohesion[2 * i] = cohe_x / mod_c; // coh_x = (coh_x / n  - i.x)/mod
			d_cohesion[2 * i + 1] = cohe_y / mod_c; // coh_y = (coh_y / n  - i.y)/mod
		}
	}
}

__global__ void updatePositions_kernel(float* d_boids, int boidsCount, float* d_separation, float* d_alignment, float* d_cohesion, float borderL, float borderR, float borderT, float borderB)
{

	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; //index of boid 
	if (i >= boidsCount)
		return;
	
	float dtime = 0.002;
	if (d_boids[6 * i + 5] == 1)
		dtime *= 7;
	else if (d_boids[6 * i + 5] == 2)
		dtime /= 5;
	float A = 10.0, B = 10.0, C = 10.0;

	float dvx = A * d_separation[2 * i] + B * d_alignment[2 * i] + C * d_cohesion[2 * i];
	float dvy = A * d_separation[2 * i + 1 ] + B * d_alignment[2 * i + 1] + C * d_cohesion[2 * i + 1];

	// update velocities
	d_boids[6 * i + 2] += (dvx * dtime);
	d_boids[6 * i + 3] += (dvy * dtime);


	// update pos
	 //-1 1
	if (d_boids[6 * i] <= borderL && d_boids[6 * i + 2] < 0)
	{
		d_boids[6 * i] = 1.0;
		d_boids[6 * i + 1] = d_boids[6 * i + 1] + (d_boids[6 * i + 3] * dtime);
	}
	else if (d_boids[6 * i] >= borderR && d_boids[6 * i + 2] > 0)
	{
		d_boids[6 * i] = -1.0;
		d_boids[6 * i + 1] = d_boids[6 * i + 1] + (d_boids[6 * i + 3] * dtime);
	}
	else if (d_boids[6 * i + 1] <= borderT && d_boids[6 * i + 3] < 0)
	{
		d_boids[6 * i] = d_boids[6 * i] + (d_boids[6 * i + 2] * dtime);
		d_boids[6 * i + 1] = 1.0;
	}
	else if (d_boids[6 * i + 1] >= borderB && d_boids[6 * i + 3] > 0)
	{
		d_boids[6 * i] = d_boids[6 * i] + (d_boids[6 * i + 2] * dtime);
		d_boids[6 * i + 1] = -1.0;
	}
	else
	{
		d_boids[6 * i] = d_boids[6 * i] + (d_boids[6 * i + 2] * dtime);
		d_boids[6 * i + 1] = d_boids[6 * i + 1] + (d_boids[6 * i + 3] * dtime);
	}

	// reset calculation arrays
	d_separation[2*i] = 0.0;
	d_alignment[2*i] = 0.0;
	d_cohesion[2*i] = 0.0;

	d_separation[2 * i + 1] = 0.0;
	d_alignment[2 * i + 1] = 0.0;
	d_cohesion[2 * i + 1] = 0.0;
}

__global__ void calculateCells_kernel(int* d_boidsIndices, int* d_cells, float* d_boids, unsigned int boidsCount, unsigned int gridRows, unsigned int gridCols, float borderL, float borderR, float borderT, float borderB)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; //index of boid 
	if (i >= boidsCount)
		return;

	// scale to [0, 1]
	float scalledBoids_x = (d_boids[6 * i] + (-borderL)) / (borderR - borderL);
	float scalledBoids_y = (d_boids[6 * i + 1] + (-borderB)) / (borderT - borderB);

	int cell_j = int(scalledBoids_x*(float)gridCols); // number of column
	int cell_i = int(scalledBoids_y*(float)gridRows); // number of row
	if (cell_j == gridCols)
		cell_j--;
	if (cell_i == gridRows)
		cell_i--;

	int cell = gridCols * cell_i + cell_j;

	// save cell value
	d_boids[6 * i + 4] = cell;
	d_boidsIndices[i] = i;
	d_cells[i] = cell;
}

__global__ void setCellsStartAndEnds_kernel(int* d_cells, int boidsCount, int* d_cellsStarts, int* d_cellsEnds)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= boidsCount)
		return;

	if (idx == 0 || d_cells[idx] != d_cells[idx - 1])
	{
		d_cellsStarts[d_cells[idx]] = idx;
	}
	if (idx == boidsCount - 1 || d_cells[idx] != d_cells[idx + 1])
	{
		d_cellsEnds[d_cells[idx]] = idx;
	}
}




CudaModule::CudaModule(unsigned int blocksCount, unsigned int threadsCount, unsigned int gridRows, unsigned int gridCols, float borderL, float borderR, float borderT, float borderB, int mode)
{
	this->blocksCount = blocksCount;
	this->threadsCount = threadsCount;
	this->boidsCount = blocksCount * threadsCount;
	this->gridRows = gridRows;
	this->gridCols = gridCols;
	this->borderL = borderL;
	this->borderR = borderR;
	this->borderT = borderT;
	this->borderB = borderB;

	this->mode = mode;

	InitBoids();

	//vertex array
	this->va = new VertexArray();

	// vertex buffer
	this->vb = new VertexBuffer(h_boids, boidsCount * 6 * sizeof(float));
	CudaCall(cudaGraphicsGLRegisterBuffer(&this->cudaResourceBufBoids, this->vb->m_RendererID, cudaGraphicsRegisterFlagsNone));

	// vertex buffer layout
	// add vertex buffer layout attributes
	// 0 - positon
	layout.Push<float>(2);
	// 1 - velocity
	layout.Push<float>(2);
	// cell idx
	layout.Push<float>(1);
	// boid type
	layout.Push<float>(1);
	// add buffer with layout to vertex array
	va->AddBuffer(vb, layout);

	// index buffer
	this->ib = new IndexBuffer(boidsIndices, boidsCount);

	// shader
	this->shader = new Shader("Basic2.shader");
	shader->Bind();

	// unbind 
	va->UnBind();
	vb->UnBind();
	ib->UnBind();
	shader->UnBind();

	this->renderer = new Renderer();

}

CudaModule::~CudaModule()
{
	if (boidsIndices != nullptr)
		delete this->boidsIndices;

	if (h_separation != nullptr)
		delete this->h_separation;

	CudaCall(cudaFree(this->d_separation));

	if (h_alignment != nullptr)
		delete this->h_alignment;

	CudaCall(cudaFree(this->d_alignment));

	if (h_cohesion != nullptr)
		delete this->h_cohesion;

	CudaCall(cudaFree(this->d_cohesion));

	if (this->va != nullptr)
		delete va;

	if (this->va != nullptr)
		delete vb;

	if (this->ib != nullptr)
		delete ib;

	if (this->va != nullptr)
		delete shader;

	if (this->va != nullptr)
		delete renderer;

	CudaCall(cudaFree(this->d_boidsIndices));
	CudaCall(cudaFree(this->d_cells));
	CudaCall(cudaFree(this->d_cellsStarts));
	CudaCall(cudaFree(this->d_cellsEnds));

	if (h_boids != nullptr)
		delete this->h_boids;

	CudaCall(cudaGraphicsUnregisterResource(this->cudaResourceBufBoids));
}

void CudaModule::Draw()
{
	renderer->Clear();

	shader->Bind();

	CalculateAndUpdate();

	renderer->Draw(va, ib, shader);
}

void CudaModule::CalculateAndUpdate()
{
#ifdef TM
	cudaEvent_t start, stop;
	CudaCall(cudaEventCreate(&start));
	CudaCall(cudaEventCreate(&stop));
	float milliseconds = 0;
#endif

#ifdef TM
	CudaCall(cudaEventRecord(start));
#endif
	/////////////////////////////////////////////////////////// boids ///////////////////////////////////////////////////////////
	CudaCall(cudaGraphicsMapResources(1, &this->cudaResourceBufBoids, 0));
	size_t sizeBoids = sizeof(float) * 6 * boidsCount;
	CudaCall(cudaGraphicsResourceGetMappedPointer((void **)&this->d_boids, &sizeBoids, this->cudaResourceBufBoids));
#ifdef TM
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	milliseconds = 0;
	CudaCall(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "1. Mapping graphics resources: " << milliseconds << " ms" << endl;
#endif

	/////////////////////////////////////////////////////////// calculation ///////////////////////////////////////////////////////////
#ifdef TM
	CudaCall(cudaEventRecord(start));
#endif
	/////////////////////////////////////////////////////////// set in which cell the boid is ///////////////////////////////////////////////////////////	
	calculateCells_kernel << <blocksCount, threadsCount >> > (this->d_boidsIndices, this->d_cells, this->d_boids, this->boidsCount, this->gridRows, this->gridCols, this->borderL, this->borderR, this->borderT, this->borderB);
#ifdef TM
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	milliseconds = 0;
	CudaCall(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "2. calculateCells_kernel (for each boid calculate cell): " << milliseconds << " ms" << endl;
#endif

#ifdef TM
	CudaCall(cudaEventRecord(start));
#endif
	/////////////////////////////////////////////////////////// sort boids by cells ///////////////////////////////////////////////////////////
	thrust::device_ptr<int> dev_ptr_cells(this->d_cells);
	thrust::device_ptr<int> dev_ptr_boidsIndices(this->d_boidsIndices);
	thrust::sort_by_key(dev_ptr_cells, dev_ptr_cells + boidsCount, dev_ptr_boidsIndices);
#ifdef TM
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	milliseconds = 0;
	CudaCall(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "3. thrust::sort_by_key (sort boids by cells): " << milliseconds << " ms" << endl;
#endif

#ifdef TM
	CudaCall(cudaEventRecord(start));
#endif
	/////////////////////////////////////////////////////////// set cells starts and ends ///////////////////////////////////////////////////////////
	setCellsStartAndEnds_kernel << <blocksCount, threadsCount >> > (this->d_cells, this->boidsCount, this->d_cellsStarts, this->d_cellsEnds);
#ifdef TM
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	milliseconds = 0;
	CudaCall(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "4. setCellsStartAndEnds_kernel (set start index and end index for each cell): " << milliseconds << " ms" << endl;
#endif

#ifdef TM
	CudaCall(cudaEventRecord(start));
#endif
	/////////////////////////////////////////////////////////// boids model algorithm calculation ///////////////////////////////////////////////////////////
	calculatePositions_kernel << <blocksCount, threadsCount >> > (this->d_boids, this->boidsCount, this->d_boidsIndices, this->d_cells, this->d_cellsStarts, this->d_cellsEnds, this->gridRows, this->gridCols, this->d_separation, this->d_alignment, this->d_cohesion, this->mode, this->borderL, this->borderR);
#ifdef TM
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	milliseconds = 0;
	CudaCall(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "5. calculatePositions_kernel (main part of the boids model algorithm): " << milliseconds << " ms" << endl;
#endif

#ifdef TM
	CudaCall(cudaEventRecord(start));
#endif
	/////////////////////////////////////////////////////////// update position based on previous calculation ///////////////////////////////////////////////////////////
	updatePositions_kernel << <blocksCount, threadsCount >> > (this->d_boids, this->boidsCount, this->d_separation, this->d_alignment, this->d_cohesion, this->borderL, this->borderR, this->borderT, this->borderB);
#ifdef TM
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	milliseconds = 0;
	CudaCall(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "6. updatePositions_kernel (update position of each boid according to previous calculation): " << milliseconds << " ms" << endl;
#endif

#ifdef TM
	CudaCall(cudaEventRecord(start));
#endif
	CudaCall(cudaGraphicsUnmapResources(1, &this->cudaResourceBufBoids, 0));
#ifdef TM
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	milliseconds = 0;
	CudaCall(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "7. unmapping graphics resources: " << milliseconds << " ms" << endl;
#endif

}

void CudaModule::InitBoids()
{
	srand(time(NULL));

#ifdef TM
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif

#ifdef TM
	begin = std::chrono::steady_clock::now();
#endif
	this->h_boids = new float[6*boidsCount];
	this->boidsIndices = new unsigned int[boidsCount];
	this->h_separation = new float[2 * boidsCount];
	this->h_alignment = new float[2 * boidsCount];
	this->h_cohesion = new float[2 * boidsCount];

	for (int i = 0; i < 6*boidsCount; i+=6)
	{
		this->h_boids[i] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i+1] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i+2] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i+3] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i+4] = 0;
		this->h_boids[i + 5] = 0;
		if (boidsCount > 100)
		{
			if (i >= 0 && i <= 40)
				this->h_boids[i + 5] = 1;
			else if (i > 40 && i <= 60)
				this->h_boids[i + 5] = 2;
		}
	}

	for (int i = 0; i < boidsCount; i++)
	{
		this->boidsIndices[i] = i;
	}

	for (int i = 0; i < 2* boidsCount; i++)
	{
		this->h_separation[i] = 0.0;
		this->h_alignment[i]  = 0.0;
		this->h_cohesion[i]   = 0.0;
	}

	CudaCall(cudaMalloc((void**)&(this->d_separation), 2 * sizeof(float)*this->boidsCount));
	CudaCall(cudaMalloc((void**)&(this->d_alignment), 2 * sizeof(float)*this->boidsCount));
	CudaCall(cudaMalloc((void**)&(this->d_cohesion), 2 * sizeof(float)*this->boidsCount));
	CudaCall(cudaMalloc((void**)&(this->d_boidsIndices), sizeof(int)*this->boidsCount));
	CudaCall(cudaMalloc((void**)&(this->d_cells), sizeof(int)*this->boidsCount));
	CudaCall(cudaMalloc((void**)&(this->d_cellsStarts), sizeof(int)*this->gridRows*this->gridCols));
	CudaCall(cudaMalloc((void**)&(this->d_cellsEnds), sizeof(int)*this->gridRows*this->gridCols));
#ifdef TM
	end = std::chrono::steady_clock::now();
	std::cout << "0. Preparing data : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " ms" << std::endl;
#endif

}

void CudaModule::PrintTable(float* tab, int size)
{
	for (int i = 0; i < size; i++)
		std::cout << i << ": " << tab[i] << std::endl;
}

void CudaModule::PrintTableI(int* tab, int size)
{
	for (int i = 0; i < size; i++)
		std::cout << i << ": " << tab[i] << std::endl;
}
