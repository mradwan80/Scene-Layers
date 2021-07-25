#pragma once

#include <vector>
#include <GL/glew.h>
#include "Browser.h"


class VisibilityUpdater
{
    static VisibilityUpdater *instance;

	GLuint vao;
	GLuint vbo;
	int pnum;
	vector<float>Visible;
	vector<int>*ObjectIds;

    VisibilityUpdater(int num, GLuint va, GLuint vb, vector<int>*oids);

public:
	
	static VisibilityUpdater *getInstance(int num, GLuint va, GLuint vb, vector<int>*oids);

	void UpdateVisibility(Browser* browser);
};
