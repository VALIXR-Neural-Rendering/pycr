
#pragma once

#include <compute/FileLoader.h>

using namespace std;
using glm::vec3;
using glm::dvec3;

namespace fs = std::filesystem;


struct LasFile {
	int64_t fileIndex = 0;
	string path;
	int64_t numPoints = 0;
	int64_t numPointsLoaded = 0;
	uint32_t offsetToPointData = 0;
	int pointFormat = 0;
	uint32_t bytesPerPoint = 0;
	dvec3 scale = { 1.0, 1.0, 1.0 };
	dvec3 offset = { 0.0, 0.0, 0.0 };
	dvec3 boxMin;
	dvec3 boxMax;

	int64_t numBatches = 0;

	// index of first point in the sparse gpu buffer
	int64_t sparse_point_offset = 0;

	bool isSelected = false;
	bool isHovered = false;
	bool isDoubleClicked = false;
};

struct LasLoaderSparse : FileLoader<LasFile> {

	LasLoaderSparse(string lfname);

	void addFile(string lfname);

};