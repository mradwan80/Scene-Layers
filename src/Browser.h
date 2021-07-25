#pragma once

#include <vector>
using namespace std;

class Browser{

    static Browser *instance;

	int currentLayer;
	int LayersNum;
	vector<int>Layers;
    Browser();

public:
    
    static Browser *getInstance();
	
	void CreateLayers(int objectsnum);
	bool incrementLayer();
	bool decrementLayer();
	int GetCurrentLayer();
	int GetObjectLayer(int obj);
	
};
