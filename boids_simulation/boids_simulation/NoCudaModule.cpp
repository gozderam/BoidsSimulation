#include "NoCudaModule.h"
#include <time.h>
#include <algorithm>

#define TM

// i - index of boid we make calculations for, cellIdx - index of cell we make calculations in
void calculateWithinCell(
	int i, int cellIdx,
	float* h_boids, int* h_boidsIndices, int* h_cellsStarts, int* h_cellsEnds,
	int* sepa_count, float sepa_f, float* sepa_x, float* sepa_y,
	int* alig_count, float alig_f, float* align_x, float* align_y,
	int* cohe_count, float cohe_f, float* cohe_x, float* cohe_y)
{
	// from index in (boidsIices, cell)[] where cells beigns to index where it ends
	for (int jj = h_cellsStarts[cellIdx]; jj <= h_cellsEnds[cellIdx]; jj++)
	{
		int j = h_boidsIndices[jj];
		if (i == j)
			continue;

		float dist_square = ((h_boids[6 * i] - h_boids[6 * j])*(h_boids[6 * i] - h_boids[6 * j]))
			+ ((h_boids[6 * i + 1] - h_boids[6 * j + 1])*(h_boids[6 * i + 1] - h_boids[6 * j + 1]));

		// sep += i -j / dist
		if (dist_square < sepa_f*sepa_f && dist_square > 0)
		{
			float dist = sqrt(dist_square);
			(*sepa_count)++;
			(*sepa_x) += (h_boids[i * 6] - h_boids[6 * j]) / dist;// x
			(*sepa_y) += (h_boids[i * 6 + 1] - h_boids[6 * j + 1]) / dist;// y
		}

		// align += j.v
		if (dist_square < alig_f*alig_f && dist_square>0)
		{
			(*alig_count)++;
			(*align_x) += h_boids[6 * j + 2]; // vx
			(*align_y) += h_boids[6 * j + 3]; // vy
		}

		//coh += j
		if (dist_square < cohe_f * cohe_f && dist_square > 0)
		{
			(*cohe_count)++;
			(*cohe_x) += h_boids[6 * j]; // x
			(*cohe_y) += h_boids[6 * j + 1]; // y
		}
	}
}

void calculatePositions(float* h_boids, int boidsCount, int* h_boidsIndices, int* h_cells, int* h_cellsStarts, int* h_cellsEnds, int gridRows, int gridCols, float* h_separation, float* h_alignment, float* h_cohesion, int mode, float borderL, float borderR)
{
	for (int i = 0; i < boidsCount; i++)
	{

		int sepa_count = 0;
		int alig_count = 0;
		int cohe_count = 0;

		float cohe_f = (borderR - borderL) / float(gridCols);
		float alig_f = 0.7*cohe_f;
		float sepa_f = 0.4*cohe_f;

		float sepa_x = 0.0;
		float sepa_y = 0.0;
		float align_x = 0.0;
		float align_y = 0.0;
		float cohe_x = 0.0;
		float cohe_y = 0.0;

		// cellIdx (the same cell)
		int cellIdx = h_boids[6 * i + 4];
		calculateWithinCell(
			i, cellIdx,
			h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
			&sepa_count, sepa_f, &sepa_x, &sepa_y,
			&alig_count, alig_f, &align_x, &align_y,
			&cohe_count, cohe_f, &cohe_x, &cohe_y);

		if (mode == 1)
		{
			//x-1, y-1
			cellIdx = h_boids[6 * i + 4] - gridCols - 1;
			if (cellIdx >= 0)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
					&sepa_count, sepa_f, &sepa_x, &sepa_y,
					&alig_count, alig_f, &align_x, &align_y,
					&cohe_count, cohe_f, &cohe_x, &cohe_y);

			//x, y-1
			cellIdx = h_boids[6 * i + 4] - gridCols;
			if (cellIdx >= 0)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
					&sepa_count, sepa_f, &sepa_x, &sepa_y,
					&alig_count, alig_f, &align_x, &align_y,
					&cohe_count, cohe_f, &cohe_x, &cohe_y);

			//x+1, y+1
			cellIdx = h_boids[6 * i + 4] - gridCols + 1;
			if (cellIdx >= 0)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
					&sepa_count, sepa_f, &sepa_x, &sepa_y,
					&alig_count, alig_f, &align_x, &align_y,
					&cohe_count, cohe_f, &cohe_x, &cohe_y);

			//x-1, y
			cellIdx = h_boids[6 * i + 4] - 1;
			if (cellIdx >= 0)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
					&sepa_count, sepa_f, &sepa_x, &sepa_y,
					&alig_count, alig_f, &align_x, &align_y,
					&cohe_count, cohe_f, &cohe_x, &cohe_y);

			//x+1, y
			cellIdx = h_boids[6 * i + 4] + 1;
			if (cellIdx < gridCols*gridRows)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
					&sepa_count, sepa_f, &sepa_x, &sepa_y,
					&alig_count, alig_f, &align_x, &align_y,
					&cohe_count, cohe_f, &cohe_x, &cohe_y);

			//x-1, y+1
			cellIdx = h_boids[6 * i + 4] + gridCols - 1;
			if (cellIdx < gridCols*gridRows)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
					&sepa_count, sepa_f, &sepa_x, &sepa_y,
					&alig_count, alig_f, &align_x, &align_y,
					&cohe_count, cohe_f, &cohe_x, &cohe_y);

			//x, y+1
			cellIdx = h_boids[6 * i + 4] + gridCols;
			if (cellIdx < gridCols*gridRows)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
					&sepa_count, sepa_f, &sepa_x, &sepa_y,
					&alig_count, alig_f, &align_x, &align_y,
					&cohe_count, cohe_f, &cohe_x, &cohe_y);

			//x+1, y+1
			cellIdx = h_boids[6 * i + 4] + gridCols + 1;
			if (cellIdx < gridCols*gridRows)
				calculateWithinCell(
					i, cellIdx,
					h_boids, h_boidsIndices, h_cellsStarts, h_cellsEnds,
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
				h_separation[2 * i] = 0;
				h_separation[2 * i + 1] = 0;
			}
			else
			{
				h_separation[2 * i] = sepa_x / mod_s - h_boids[6 * i + 2];// sep_x = sep_x/mod + i.vx 
				h_separation[2 * i + 1] = sepa_y / mod_s - h_boids[6 * i + 3];// sep_y = sep_y/mod + i.vy 
			}
		}


		// alignment
		if (alig_count > 0)
		{
			float mod_a = sqrt(align_x*align_x + align_y * align_y);
			if (mod_a == 0)
			{
				h_alignment[2 * i] = 0;
				h_alignment[2 * i + 1] = 0;
			}
			else
			{
				h_alignment[2 * i] = align_x / mod_a - h_boids[6 * i + 2]; //alig_x = alig_x.mod + i.vx
				h_alignment[2 * i + 1] = align_y / mod_a - h_boids[6 * i + 3]; //alig_y = alig_y.mod + i.vy
			}
		}


		//cohesion
		if (cohe_count > 0)
		{
			cohe_x = cohe_x / cohe_count - h_boids[6 * i];
			cohe_y = cohe_y / cohe_count - h_boids[6 * i + 1];


			float mod_c = sqrt(cohe_x*cohe_x + cohe_y * cohe_y);
			if (mod_c == 0)
			{
				h_cohesion[2 * i] = 0;
				h_cohesion[2 * i + 1] = 0;
			}
			else
			{
				h_cohesion[2 * i] = cohe_x / mod_c; // coh_x = (coh_x / n  - i.x)/mod
				h_cohesion[2 * i + 1] = cohe_y / mod_c; // coh_y = (coh_y / n  - i.y)/mod
			}
		}
	}
}

void updatePositions(float* h_boids, int boidsCount, float* h_separation, float* h_alignment, float* h_cohesion, float borderL, float borderR, float borderT, float borderB)
{
	for (int i = 0; i < boidsCount; i++)
	{
		float dtime = 0.002;
		if (h_boids[6 * i + 5] == 1)
			dtime *= 7;
		else if (h_boids[6 * i + 5] == 2)
			dtime /= 5;
		float A = 10.0, B = 10.0, C = 10.0;

		float dvx = A * h_separation[2 * i] + B * h_alignment[2 * i] + C * h_cohesion[2 * i];		
		float dvy = A * h_separation[2 * i + 1] + B * h_alignment[2 * i + 1] + C * h_cohesion[2 * i + 1];

		// update velocities
		h_boids[6 * i + 2] += (dvx * dtime);
		h_boids[6 * i + 3] += (dvy * dtime);


		// update pos
		 //-1 1
		if (h_boids[6 * i] <= borderL && h_boids[6 * i + 2] < 0)
		{
			h_boids[6 * i] = 1.0;
			h_boids[6 * i + 1] = h_boids[6 * i + 1] + (h_boids[6 * i + 3] * dtime);
		}
		else if (h_boids[6 * i] >= borderR && h_boids[6 * i + 2] > 0)
		{
			h_boids[6 * i] = -1.0;
			h_boids[6 * i + 1] = h_boids[6 * i + 1] + (h_boids[6 * i + 3] * dtime);
		}
		else if (h_boids[6 * i + 1] <= borderT && h_boids[6 * i + 3] < 0)
		{
			h_boids[6 * i] = h_boids[6 * i] + (h_boids[6 * i + 2] * dtime);
			h_boids[6 * i + 1] = 1.0;
		}
		else if (h_boids[6 * i + 1] >= borderB && h_boids[6 * i + 3] > 0)
		{
			h_boids[6 * i] = h_boids[6 * i] + (h_boids[6 * i + 2] * dtime);
			h_boids[6 * i + 1] = -1.0;
		}
		else
		{
			h_boids[6 * i] = h_boids[6 * i] + (h_boids[6 * i + 2] * dtime);
			h_boids[6 * i + 1] = h_boids[6 * i + 1] + (h_boids[6 * i + 3] * dtime);
		}


		// reset calculation arrays
		h_separation[2 * i] = 0.0;
		h_alignment[2 * i] = 0.0;
		h_cohesion[2 * i] = 0.0;

		h_separation[2 * i + 1] = 0.0;
		h_alignment[2 * i + 1] = 0.0;
		h_cohesion[2 * i + 1] = 0.0;
	}
}

void calculateCells(int* h_boidsIndices, int* h_cells, float* h_boids, unsigned int boidsCount, unsigned int gridRows, unsigned int gridCols, float borderL, float borderR, float borderT, float borderB)
{
	for (int i = 0; i < boidsCount; i++)
	{
		// scale to [0, 1]
		float scalledBoids_x = (h_boids[6 * i] + (-borderL)) / (borderR - borderL);
		float scalledBoids_y = (h_boids[6 * i + 1] + (-borderB)) / (borderT - borderB);

		int cell_j = int(scalledBoids_x*(float)gridCols); // number of column
		int cell_i = int(scalledBoids_y*(float)gridRows); // number of row
		if (cell_j == gridCols)
			cell_j--;
		if (cell_i == gridRows)
			cell_i--;

		int cell = gridCols * cell_i + cell_j;


		// save cell value
		h_boids[6 * i + 4] = cell;
		h_boidsIndices[i] = i;
		h_cells[i] = cell;
	}
}

void setCellsStartAndEnds(int* h_cells, int boidsCount, int* h_cellsStarts, int* h_cellsEnds)
{
	for (int idx = 0; idx < boidsCount; idx++)
	{ 
		if (idx == 0 || h_cells[idx] != h_cells[idx - 1])
		{
			h_cellsStarts[h_cells[idx]] = idx;
		}
		if (idx == boidsCount - 1 || h_cells[idx] != h_cells[idx + 1])
		{
			h_cellsEnds[h_cells[idx]] = idx;
		}
	}
}

NoCudaModule::NoCudaModule(unsigned int boidsCount, unsigned int gridRows, unsigned int gridCols, float borderL, float borderR, float borderT, float borderB, int mode)
{
	this->boidsCount = boidsCount;
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

NoCudaModule::~NoCudaModule()
{
	if (h_boids != nullptr)
		delete this->h_boids;

	if (boidsIndices != nullptr)
		delete this->boidsIndices;

	if (h_separation != nullptr)
		delete this->h_separation;

	if (h_alignment != nullptr)
		delete this->h_alignment;

	if (h_cohesion != nullptr)
		delete this->h_cohesion;

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

	if(this->boidsIndices!= nullptr)
		delete this->h_boidsIndices;

	if (this->h_cells != nullptr)
		delete this->h_cells;

	if (this->h_cellsStarts != nullptr)
		delete this->h_cellsStarts;

	if (this->h_cellsEnds != nullptr)
		delete this->h_cellsEnds;
}

void NoCudaModule::Draw()
{
	renderer->Clear();

	shader->Bind();

	CalculateAndUpdate();

	vb->UpdateData(h_boids, 6 * boidsCount * sizeof(float));
	renderer->Draw(va, ib, shader);
}

void NoCudaModule::CalculateAndUpdate()
{
#ifdef TM
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif

	/////////////////////////////////////////////////////////// calculation ///////////////////////////////////////////////////////////
#ifdef TM
	begin = std::chrono::steady_clock::now();
#endif
	/////////////////////////////////////////////////////////// set in which cell the boid is ///////////////////////////////////////////////////////////	
	calculateCells(this->h_boidsIndices, this->h_cells, this->h_boids, this->boidsCount, this->gridRows, this->gridCols, this->borderL, this->borderR, this->borderT, this->borderB);
#ifdef TM
	end = std::chrono::steady_clock::now();
	std::cout << "2. calculateCells (for each boid calculate cell): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << " ms" << std::endl;
#endif

#ifdef TM
	begin = std::chrono::steady_clock::now();
#endif
	/////////////////////////////////////////////////////////// sort boids by cells ///////////////////////////////////////////////////////////
	int* indices = new int[this->boidsCount];
	for (int i = 0; i < this->boidsCount; i++)
		indices[i] = i;

	int *c = this->h_cells;
	std::sort(indices, indices + this->boidsCount, [c](int i, int j) mutable {return c[i] < c[j]; }  );
	

	int* h_cells2 = new int[this->boidsCount];
	int* h_boidsIndices2 = new int[this->boidsCount];

	for (int i = 0; i < this->boidsCount; i++)
	{
		h_cells2[i] = this->h_cells[indices[i]];
		h_boidsIndices2[i] = this->h_boidsIndices[indices[i]];
	}

	delete this->h_cells;
	delete this->h_boidsIndices;

	this->h_cells = h_cells2;
	this->h_boidsIndices = h_boidsIndices2;

	delete indices;
#ifdef TM
	end = std::chrono::steady_clock::now();
	std::cout << "3. std::sort(sort boids by cells): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;
#endif

#ifdef TM
	begin = std::chrono::steady_clock::now();
#endif
	/////////////////////////////////////////////////////////// set cells starts and ends ///////////////////////////////////////////////////////////
	setCellsStartAndEnds(this->h_cells, this->boidsCount, this->h_cellsStarts, this->h_cellsEnds);
#ifdef TM
	end = std::chrono::steady_clock::now();
	std::cout << "4. setCellsStartAndEnds (set start index and end index for each cell): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;
#endif

#ifdef TM
	begin = std::chrono::steady_clock::now();
#endif
	/////////////////////////////////////////////////////////// boids model algorithm calculation ///////////////////////////////////////////////////////////
	calculatePositions(this->h_boids, this->boidsCount, this->h_boidsIndices, this->h_cells, this->h_cellsStarts, this->h_cellsEnds, this->gridRows, this->gridCols, this->h_separation, this->h_alignment, this->h_cohesion, this->mode, this->borderL, this->borderR);
#ifdef TM
	end = std::chrono::steady_clock::now();
	std::cout << "5. calculatePositions (main part of the boids model algorithm): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0<< " ms" << std::endl;
#endif

#ifdef TM
	begin = std::chrono::steady_clock::now();
#endif
	/////////////////////////////////////////////////////////// update position based on previous calculation ///////////////////////////////////////////////////////////
	updatePositions(this->h_boids, this->boidsCount, this->h_separation, this->h_alignment, this->h_cohesion, this->borderL, this->borderR, this->borderT, this->borderB);
#ifdef TM
	end = std::chrono::steady_clock::now();
	std::cout << "6. updatePositions_kernel (update position of each boid according to previous calculation): "  << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /1000.0<< " ms" << std::endl;
#endif

}

void NoCudaModule::InitBoids()
{
	srand(time(NULL));

#ifdef TM
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif

#ifdef TM
	begin = std::chrono::steady_clock::now();
#endif
	this->h_boids = new float[6 * boidsCount];
	this->boidsIndices = new unsigned int[boidsCount];
	this->h_separation = new float[2 * boidsCount];
	this->h_alignment = new float[2 * boidsCount];
	this->h_cohesion = new float[2 * boidsCount];

	this->h_boidsIndices = new int[this->boidsCount];
	this->h_cells = new int[this->boidsCount];
	this->h_cellsStarts = new int[this->gridRows*this->gridCols];
	this->h_cellsEnds = new int[this->gridRows*this->gridCols];

	for (int i = 0; i < 6 * boidsCount; i += 6)
	{
		this->h_boids[i] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i + 1] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i + 2] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i + 3] = 2.0*((float)rand() / (float)RAND_MAX) - 1.0;
		this->h_boids[i + 4] = 0;
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
		this->h_cells[i] = 0;
	}

	for (int i = 0; i < 2 * boidsCount; i++)
	{
		this->h_separation[i] = 0.0;
		this->h_alignment[i] = 0.0;
		this->h_cohesion[i] = 0.0;
	}

	
#ifdef TM
	end = std::chrono::steady_clock::now();
	std::cout << "0. Preparing data : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " ms" << std::endl;
#endif
}
