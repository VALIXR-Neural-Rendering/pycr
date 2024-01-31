
#pragma once

#include <string>
#include <filesystem>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include "glm/vec3.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"
#include "Shader.h"
#include "GLBuffer.h"
#include "Resources.h"
#include "RenderUtils.h"

using namespace std;
using glm::vec3;
using glm::dvec3;

namespace fs = std::filesystem;


template <typename FileType>
struct FileLoader {

	int64_t MAX_POINTS = 1'000'000'000;
	int64_t PAGE_SIZE = 0;

	struct LoadTask {
		shared_ptr<FileType> file;
		int64_t firstPoint;
		int64_t numPoints;
	};

	struct UploadTask {
		shared_ptr<FileType> file;
		int64_t sparse_pointOffset;
		int64_t sparse_batchOffset;
		int64_t numPoints;
		int64_t numBatches;
		shared_ptr<Buffer> bXyzLow;
		shared_ptr<Buffer> bXyzMed;
		shared_ptr<Buffer> bXyzHig;
		shared_ptr<Buffer> bColors;
		shared_ptr<Buffer> bVel;
		shared_ptr<Buffer> bBatches;
	};

	vector<shared_ptr<FileType>> files;
	vector<LoadTask> loadTasks;
	vector<UploadTask> uploadTasks;

	int numPointsLoaded;
	int64_t numPoints = 0;
	int64_t numBatches = 0;
	int64_t numBatchesLoaded = 0;
	int64_t bytesReserved = 0;
	int64_t numFiles = 0;

	GLBuffer ssBatches;
	GLBuffer ssXyzLow;
	GLBuffer ssXyzMed;
	GLBuffer ssXyzHig;
	GLBuffer ssColors;
	GLBuffer ssVel;
	GLBuffer ssLoadBuffer;

	virtual void addFile(string lfname) = 0;
};