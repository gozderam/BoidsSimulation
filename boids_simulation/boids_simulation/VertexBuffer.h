#ifndef VERTEXBUFFER
#define VERTEXBUFFER

class VertexBuffer
{
//private :
public:
	unsigned int m_RendererID;

public:
	VertexBuffer(const void* data, unsigned int size);
	~VertexBuffer();

	void Bind() const;
	void UnBind() const;
	void UpdateData(const void* data, unsigned int size);
};

#endif