#version 330 

in vec3 colorVF ; 

void main(void)
{

	gl_FragColor= vec4(colorVF.xyz,1.0);

	
}
