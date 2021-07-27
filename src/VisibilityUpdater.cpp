#include "VisibilityUpdater.h"

VisibilityUpdater *VisibilityUpdater::instance = 0;

VisibilityUpdater::VisibilityUpdater(int num, GLuint va, GLuint vb, vector<int>*oids)
{
	vao=va;
	vbo=vb;
	pnum=num;
	ObjectIds=oids;
	Visible.resize(pnum);

}

void VisibilityUpdater::UpdateVisibility(Browser* browser)
{
	int currentLayer = browser->GetCurrentLayer();

	for(int i=0;i<pnum;i++)
	{
		if(browser->GetObjectLayer(ObjectIds->at(i))>=currentLayer)
			Visible[i]=1;
		else
			Visible[i]=0;

	}

	glBindVertexArray(vao); 
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, pnum * 1 * sizeof(GLfloat), &Visible[0], GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}

VisibilityUpdater* VisibilityUpdater::getInstance(int num, GLuint va, GLuint vb, vector<int>*oids) 
{
    if (!instance)
        instance = new VisibilityUpdater(num, va, vb, oids);
    return instance;
}

/*VisibilityUpdater::~VisibilityUpdater() 
{
	if(instance==nullptr) return;
	delete instance;
	instance=nullptr;
}*/

void VisibilityUpdater::DestroyInstance() 
{
	delete instance;
}