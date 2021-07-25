#pragma once

#include <vector>
using namespace std;

class Browser{

	int currentLayer;
	int LayersNum;
	vector<int>Layers;
public:
	Browser();
	void CreateLayers(int objectsnum);
	bool incrementLayer();
	bool decrementLayer();
	int GetCurrentLayer();
	int GetObjectLayer(int obj);
	
};
