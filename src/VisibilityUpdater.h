#pragma once

#include <vector>
#include <GL/glew.h>
#include "Browser.h"


class VisibilityUpdater
{
	GLuint vao;
	GLuint vbo;
	int pnum;
	vector<float>Visible;
	vector<int>*ObjectIds;

public:
	VisibilityUpdater();
	VisibilityUpdater(int num, GLuint va, GLuint vb, vector<int>*oids);
	void UpdateVisibility(Browser* browser);
};
