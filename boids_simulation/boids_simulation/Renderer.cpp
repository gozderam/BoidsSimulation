#include "Renderer.h"
#include <iostream>
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"
using namespace std;

void GLClearErrors()
{
    while (glGetError());
}

bool GLLogCall(const char* function, const char* file, int line)
{
    while (GLenum error = glGetError())
    {
        cout << "Opengl error: " << error << ": " << function << " " << file << ", line: " << line << endl;
        return false;
    }

    return true;
}

void Renderer::Draw(const VertexArray* va, const IndexBuffer* ib, const Shader* shader) const
{
    shader->Bind();
    va->Bind();
    ib->Bind();
    GLCall(glDrawElements(GL_POINTS, ib->GetCount(), GL_UNSIGNED_INT, nullptr));
}

void Renderer::Clear() const 
{
    glClear(GL_COLOR_BUFFER_BIT);
}

