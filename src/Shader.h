#pragma once


#include "../external/glew-1.10.0/include/GL/glew.h"
#include <string>
using namespace std;

class Shader
{
private:
	GLuint handle;
	string shader_path;
public:
	Shader();
	Shader(string path);
	GLuint GetHandle();
	void CompileShader();
	void CompileShaderWithGeomShader();
	
	
};
