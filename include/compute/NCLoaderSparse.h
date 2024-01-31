
#pragma once

#include <compute/FileLoader.h>
#include <netcdf.h>

#define MAX_DIMS 4
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

using namespace std;
using glm::vec3;
using glm::dvec3;

namespace fs = std::filesystem;


struct NCFile {
	int64_t fileIndex = 0;
	string path;
	int64_t numPoints = 0;
	int64_t numPointsLoaded = 0;
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

struct NCLoaderSparse : FileLoader<NCFile> {

	NCLoaderSparse(string lfname);

	void addFile(string lfname);

};