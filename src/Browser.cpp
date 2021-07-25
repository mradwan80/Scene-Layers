#include "Browser.h"

Browser::Browser():currentLayer(0){}

void Browser::CreateLayers(int objectsnum)
{
	Layers.resize(objectsnum);

	Layers[0]=0;
	Layers[1]=1;
	Layers[2]=2;

	LayersNum=3;

}

bool Browser::incrementLayer()
{
	if(currentLayer<LayersNum)
	{
		currentLayer++;
		return true;
	}
	else
		return false;
}

bool Browser::decrementLayer()
{
	if(currentLayer>0)
	{
		currentLayer--;
		return true;
	}
	else
		return false;
}

int Browser::GetCurrentLayer()
{
	return currentLayer;
}

int Browser::GetObjectLayer(int obj)
{
	return Layers[obj];
}