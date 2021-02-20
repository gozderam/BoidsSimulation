#ifndef VERTEXARRAY
#define VERTEXARRAY

#include "VertexBuffer.h"

class VertexBufferLayout;

class VertexArray
{
private:
	unsigned int m_Renderer_ID;

public:
	VertexArray();
	~VertexArray();

	void AddBuffer(const VertexBuffer* vb, const VertexBufferLayout& layout);

	void Bind() const;
	void UnBind() const;
};

#endif