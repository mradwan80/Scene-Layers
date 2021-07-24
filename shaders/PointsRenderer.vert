#version 330 

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

uniform mat4 pvm_matrix;

out vec3 colorVF ; 

void main()
{

	vec4 pos4 = vec4(in_position,1.0);

	gl_Position =   pvm_matrix * pos4;

	colorVF = in_color;

	gl_PointSize=5;
}