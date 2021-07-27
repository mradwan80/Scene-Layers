#include "Browser.h"
//#include <iostream>

Browser *Browser::instance = 0;

Browser::Browser():currentLayer(0){}

void Browser::CreateLayers(int objectsnum, OcclusionGraph* graph)
{
	Layers.resize(objectsnum);

	vector<set<int>>occludedBy(objectsnum);

	for(int obj=0;obj<objectsnum;obj++)
	{
		occludedBy[obj] = graph->GetOccluders(obj);
	}

	vector<bool>taken(objectsnum,false);
	int takenNum=0;

	int layer=0;
	while(takenNum<objectsnum)
	{
		set<int>lvlObjs; //next level objects//
		
		for(int obj=0;obj<objectsnum;obj++)
		{
			if(taken[obj])
				continue;

			//if no non taken occluders, add to set//
			bool cantake=true;
			for(auto occ: occludedBy[obj])
			{
				if(!taken[occ])
					cantake=false;
			}

			if(cantake)
				lvlObjs.insert(obj);
			
		}

		if(lvlObjs.size()==0) //set has no objects//
		{
			bool found=false;
			for(int obj=0;obj<objectsnum && !found;obj++)
			{
				if(!taken[obj])
				{
					lvlObjs.insert(obj);
					found=true;
				}
			}
		}

		//std::cout<<"objects of layer " <<layer<<" :";
		/*for(auto obj : lvlObjs)
		{
			std::cout<<obj<<", ";
		}
		std::cout<<"\n";*/
		

		//update. add to taken, inc takenNum//
		for(auto obj : lvlObjs)
		{
			takenNum++;
			taken[obj]=true;
			Layers[obj]=layer;
		}
		layer++;

	}

	LayersNum=layer;


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

Browser* Browser::getInstance() 
{
      if (!instance)
        instance = new Browser;
      return instance;
}

void Browser::DestroyInstance()
{
	delete instance;
}