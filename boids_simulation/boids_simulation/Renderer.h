#ifndef RENDERER
#define RENDERER

#include <GL/glew.h>
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"

//#define ASSERT(x) if(!(x)) __debugbreak(); 
#define ASSERT(x) if(!(x)) printf("opengl error"); 
#define GLCall(x) GLClearErrors();\
    x;\
    ASSERT(GLLogCall(#x, __FILE__, __LINE__))  


void GLClearErrors();
bool GLLogCall(const char* function, const char* file, int line);

class Renderer
{
public:
    void Clear()  const ;
    void Draw(const VertexArray* va, const IndexBuffer* ib, const Shader* shader) const;

};

#endif

