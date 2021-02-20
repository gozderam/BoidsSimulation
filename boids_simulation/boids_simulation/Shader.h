#ifndef SHADER
#define SHADER

#include <string>
#include <unordered_map>
#include "glm/glm.hpp"

using namespace std;

struct ShaderProgramSource
{
	string VertexSource;
	string GeometrySource;
	string FragmentSource;
};


class Shader
{
private:
	string m_FilePath;
	unsigned int m_RendererID;
	unordered_map<string, int> m_UniformLocationCache;

public:
	Shader(const string& filepath);
	~Shader();


	void Bind() const;
	void UnBind() const;

	// set uniforms
	void SetUniform4f(const string& name, float v0, float v1, float v2, float v3);
	void SetUniformMat4f(const string& name, const glm::mat4& matrix);

private:
	unsigned int GetUniformLocation(const string& name);
	int CreateShader(const string& vertexShader, const string& geometryShader, const string& fragmentShader);
	unsigned int CompileShader(unsigned int type, const string& source);
	ShaderProgramSource ParseShader(const string& filepath);
};

#endif