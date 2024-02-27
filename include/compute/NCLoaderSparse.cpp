#include "NCLoaderSparse.h"
#include "unsuck.hpp"

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

//mutex mtx_debug;

struct LoadResult {
	shared_ptr<Buffer> bBatches;
	shared_ptr<Buffer> bXyzLow;
	shared_ptr<Buffer> bXyzMed;
	shared_ptr<Buffer> bXyzHig;
	shared_ptr<Buffer> bColors;
	shared_ptr<Buffer> bVel;
	int64_t sparse_pointOffset;
	int64_t numBatches;
};

struct Batch {
	int64_t chunk_pointOffset;
	int64_t file_pointOffset;
	int64_t sparse_pointOffset;
	int64_t numPoints;
	int64_t file_index;

	dvec3 min = { Infinity, Infinity, Infinity };
	dvec3 max = { -Infinity, -Infinity, -Infinity };
};

shared_ptr<Buffer> readVariable(int ncid, const char *varname, size_t firstPt, size_t numPts) {
	int retval, varid, vardims;
	nc_type var_type;

	if ((retval = nc_inq_varid(ncid, varname, &varid))) ERR(retval);
	if ((retval = nc_inq_vartype(ncid, varid, &var_type))) ERR(retval);
	if ((retval = nc_inq_varndims(ncid, varid, &vardims))) ERR(retval);
	
	int* varDimids = new int[vardims];
	size_t* varDimlens = new size_t[vardims];
	size_t ptDim;
	if ((retval = nc_inq_vardimid(ncid, varid, varDimids))) ERR(retval);
	if ((retval = nc_inq_dimlen(ncid, varDimids[vardims-1], &ptDim))) ERR(retval);

	float s = 0;
	size_t start[3] = { 0, firstPt, 0 };
	size_t count[3] = { 1, numPts, ptDim };
	auto varBuffer = make_shared<Buffer>(numPts * ptDim * sizeof(float));
	if ((retval = nc_get_vara(ncid, varid, start, count, varBuffer->data))) ERR(retval);

	return varBuffer;
}

shared_ptr<LoadResult> loadNC(shared_ptr<NCFile> ncfile, int64_t firstPoint, int64_t numPoints) {

	string path = ncfile->path;
	int64_t sparse_pointOffset = ncfile->sparse_point_offset + firstPoint;
	int ncid, ndims, nvars, unlimdimid, retval;
	if ((retval = nc_open(ncfile->path.c_str(), NC_NOWRITE, &ncid))) ERR(retval);
	if ((retval = nc_inq(ncid, &ndims, &nvars, NULL, &unlimdimid))) ERR(retval);

	//cout << "ndims: " << ndims << endl;
	//cout << "nvars: " << nvars << endl;
	//cout << "unlimdimid: " << unlimdimid << endl;

	auto posBuffer = readVariable(ncid, "Position", firstPoint, numPoints);
	auto colorBuffer = readVariable(ncid, "color", firstPoint, numPoints);
	auto velBuffer = readVariable(ncid, "vel", firstPoint, numPoints);
	
	if ((retval = nc_close(ncid))) ERR(retval);


	// compute batch metadata
	int64_t numBatches = numPoints / POINTS_PER_WORKGROUP;
	if ((numPoints % POINTS_PER_WORKGROUP) != 0) {
		numBatches++;
	}

	vector<Batch> batches;

	int64_t chunk_pointsProcessed = 0;
	for (int i = 0; i < numBatches; i++) {

		int64_t remaining = numPoints - chunk_pointsProcessed;
		int64_t numPointsInBatch = std::min(int64_t(POINTS_PER_WORKGROUP), remaining);

		Batch batch;

		batch.min = { Infinity, Infinity, Infinity };
		batch.max = { -Infinity, -Infinity, -Infinity };
		batch.chunk_pointOffset = chunk_pointsProcessed;
		batch.file_pointOffset = firstPoint + chunk_pointsProcessed;
		batch.sparse_pointOffset = sparse_pointOffset + chunk_pointsProcessed;
		batch.numPoints = numPointsInBatch;

		batches.push_back(batch);

		chunk_pointsProcessed += numPointsInBatch;
	}

	auto bBatches = make_shared<Buffer>(64 * numBatches);
	auto bXyzLow = make_shared<Buffer>(4 * numPoints);
	auto bXyzMed = make_shared<Buffer>(4 * numPoints);
	auto bXyzHig = make_shared<Buffer>(4 * numPoints);
	auto bColors = make_shared<Buffer>(4 * numPoints);
	auto bVel = make_shared<Buffer>(4 * 3 * numPoints);

	dvec3 boxMin = ncfile->boxMin;
	dvec3 cScale = ncfile->scale;
	dvec3 cOffset = ncfile->offset;

	// load batches/points
	for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
		Batch& batch = batches[batchIndex];

		// compute batch bounding box
		for (int i = 0; i < batch.numPoints; i++) {
			int index_pointFile = batch.chunk_pointOffset + i;

			float X = posBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 0);
			float Y = posBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 4);
			float Z = posBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 8);

			double x = double(X) * cScale.x + cOffset.x - boxMin.x;
			double y = double(Y) * cScale.y + cOffset.y - boxMin.y;
			double z = double(Z) * cScale.z + cOffset.z - boxMin.z;

			batch.min.x = std::min(batch.min.x, x);
			batch.min.y = std::min(batch.min.y, y);
			batch.min.z = std::min(batch.min.z, z);
			batch.max.x = std::max(batch.max.x, x);
			batch.max.y = std::max(batch.max.y, y);
			batch.max.z = std::max(batch.max.z, z);
		}

		dvec3 batchBoxSize = batch.max - batch.min;

		{
			int64_t batchByteOffset = 64 * batchIndex;

			bBatches->set<float>(batch.min.x, batchByteOffset + 4);
			bBatches->set<float>(batch.min.y, batchByteOffset + 8);
			bBatches->set<float>(batch.min.z, batchByteOffset + 12);
			bBatches->set<float>(batch.max.x, batchByteOffset + 16);
			bBatches->set<float>(batch.max.y, batchByteOffset + 20);
			bBatches->set<float>(batch.max.z, batchByteOffset + 24);
			bBatches->set<uint32_t>(batch.numPoints, batchByteOffset + 28);
			bBatches->set<uint32_t>(batch.sparse_pointOffset, batchByteOffset + 32);
			bBatches->set<uint32_t>(ncfile->fileIndex, batchByteOffset + 36);
		}

		int offset_rgb = 0;

		// load data
		for (int i = 0; i < batch.numPoints; i++) {
			int index_pointFile = batch.chunk_pointOffset + i;

			float X = posBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 0);
			float Y = posBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 4);
			float Z = posBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 8);

			double x = double(X) * cScale.x + cOffset.x - boxMin.x;
			double y = double(Y) * cScale.y + cOffset.y - boxMin.y;
			double z = double(Z) * cScale.z + cOffset.z - boxMin.z;

			uint32_t X30 = uint32_t(((x - batch.min.x) / batchBoxSize.x) * STEPS_30BIT);
			uint32_t Y30 = uint32_t(((y - batch.min.y) / batchBoxSize.y) * STEPS_30BIT);
			uint32_t Z30 = uint32_t(((z - batch.min.z) / batchBoxSize.z) * STEPS_30BIT);

			X30 = min(X30, uint32_t(STEPS_30BIT - 1));
			Y30 = min(Y30, uint32_t(STEPS_30BIT - 1));
			Z30 = min(Z30, uint32_t(STEPS_30BIT - 1));

			{ // low
				uint32_t X_low = (X30 >> 20) & MASK_10BIT;
				uint32_t Y_low = (Y30 >> 20) & MASK_10BIT;
				uint32_t Z_low = (Z30 >> 20) & MASK_10BIT;

				uint32_t encoded = X_low | (Y_low << 10) | (Z_low << 20);

				bXyzLow->set<uint32_t>(encoded, 4 * index_pointFile);
			}

			{ // med
				uint32_t X_med = (X30 >> 10) & MASK_10BIT;
				uint32_t Y_med = (Y30 >> 10) & MASK_10BIT;
				uint32_t Z_med = (Z30 >> 10) & MASK_10BIT;

				uint32_t encoded = X_med | (Y_med << 10) | (Z_med << 20);

				bXyzMed->set<uint32_t>(encoded, 4 * index_pointFile);
			}

			{ // hig
				uint32_t X_hig = (X30 >> 0) & MASK_10BIT;
				uint32_t Y_hig = (Y30 >> 0) & MASK_10BIT;
				uint32_t Z_hig = (Z30 >> 0) & MASK_10BIT;

				uint32_t encoded = X_hig | (Y_hig << 10) | (Z_hig << 20);

				bXyzHig->set<uint32_t>(encoded, 4 * index_pointFile);
			}

			{ // RGB

				uint32_t colPtBytes = ncfile->bytesPerPoint / 3 * 4;	// Adjusting for RGBA
				int R = colorBuffer->get<float>(index_pointFile * colPtBytes + 0);
				int G = colorBuffer->get<float>(index_pointFile * colPtBytes + 4);
				int B = colorBuffer->get<float>(index_pointFile * colPtBytes + 8);

				R = R < 256 ? R : R / 256;
				G = G < 256 ? G : G / 256;
				B = B < 256 ? B : B / 256;

				uint32_t color = R | (G << 8) | (B << 16);

				bColors->set<uint32_t>(color, 4 * index_pointFile);
			}

			{ // Velocity

				float vx = velBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 0);
				float vy = velBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 4);
				float vz = velBuffer->get<float>(index_pointFile * ncfile->bytesPerPoint + 8);

				//if (i == 0) {
				//	cout << "vx: " << vx << endl;
				//	cout << "vy: " << vy << endl;
				//	cout << "vz: " << vz << endl;
				//}

				float vel = sqrt((vx * vx) + (vy * vy) + (vz * vz));
				vx /= vel;
				vy /= vel;
				vz /= vel;

				//uint32_t velEnc = packFvec2UI(vx, vy, vz);
				//float dec_a, dec_b, dec_c;
				//unpackUI2Fvec(velEnc, dec_a, dec_b, dec_c);

				//if (i == 0) {
				//	cout << "vx_new: " << vx << endl;
				//	cout << "vy_new: " << vy << endl;
				//	cout << "vz_new: " << vz << endl;
				//	cout << "vel: " << vel << endl;
				//	cout << "velEnc: " << velEnc << endl;
				//	cout << "dec_a: " << dec_a << endl;
				//	cout << "dec_b: " << dec_b << endl;
				//	cout << "dec_c: " << dec_c << endl;
				//	cout << endl;
				//}

				bVel->set<float>(vx, ncfile->bytesPerPoint * index_pointFile + 0);
				bVel->set<float>(vy, ncfile->bytesPerPoint * index_pointFile + 4);
				bVel->set<float>(vz, ncfile->bytesPerPoint * index_pointFile + 8);
			}
		}
	}

	auto result = make_shared<LoadResult>();
	result->bXyzLow = bXyzLow;
	result->bXyzMed = bXyzMed;
	result->bXyzHig = bXyzHig;
	result->bColors = bColors;
	result->bBatches = bBatches;
	result->bVel = bVel;
	result->numBatches = numBatches;
	result->sparse_pointOffset = sparse_pointOffset;

	return result;
}

NCLoaderSparse::NCLoaderSparse(string lfname) {

	int pageSize = 0;
	glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &pageSize);
	PAGE_SIZE = pageSize;

	{ // create (sparse) buffers
		this->ssBatches = createBuffer(64 * 200'000);
		this->ssXyzLow = createSparseBuffer(4 * MAX_POINTS);
		this->ssXyzMed = createSparseBuffer(4 * MAX_POINTS);
		this->ssXyzHig = createSparseBuffer(4 * MAX_POINTS);
		this->ssColors = createSparseBuffer(4 * MAX_POINTS);
		this->ssVel = createSparseBuffer(4 * 3 * MAX_POINTS);
		this->ssLoadBuffer = createBuffer(200 * MAX_POINTS_PER_BATCH);

		GLuint zero = 0;
		glClearNamedBufferData(this->ssBatches.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}

	addFile(lfname);
}

void NCLoaderSparse::addFile(string ncfname)
{
	auto ncfile = make_shared<NCFile>();
	ncfile->path = ncfname;
	
	int ncid, dimid, varid, retval;
	size_t dimlen;
	if ((retval = nc_open(ncfile->path.c_str(), NC_NOWRITE, &ncid))) ERR(retval);
	if ((retval = nc_inq_dimid(ncid, "P", & dimid))) ERR(retval);
	if ((retval = nc_inq_dimlen(ncid, dimid, &dimlen))) ERR(retval);

	ncfile->bytesPerPoint = sizeof(float) * 3;  // las: 26
	ncfile->numPoints = dimlen;
	ncfile->numPoints = min(ncfile->numPoints, 1'000'000'000ll);
	
	ncfile->scale.x = 1.0;
	ncfile->scale.y = 1.0;
	ncfile->scale.z = 1.0;

	ncfile->offset.x = 0.0;
	ncfile->offset.y = 0.0;
	ncfile->offset.z = 0.0;
	
	// read bounding boxes
	float bbpt[3];
	if ((retval = nc_inq_varid(ncid, "bbmin", &varid))) ERR(retval);
	if ((retval = nc_get_var(ncid, varid, bbpt))) ERR(retval);
	ncfile->boxMin = dvec3(bbpt[0], bbpt[1], bbpt[2]);

	if ((retval = nc_inq_varid(ncid, "bbmax", &varid))) ERR(retval);
	if ((retval = nc_get_var(ncid, varid, bbpt))) ERR(retval);
	ncfile->boxMax = dvec3(bbpt[0], bbpt[1], bbpt[2]);
	
	{
		ncfile->numBatches = ncfile->numPoints / POINTS_PER_WORKGROUP + 1;
		ncfile->sparse_point_offset = numPoints;

		files.push_back(ncfile);
		numPoints += ncfile->numPoints;
		numBatches += ncfile->numBatches;

		stringstream ss;
		ss << "load file " << ncfname << endl;
		ss << "numPoints: " << ncfile->numPoints << "\n";
		ss << "numBatches: " << ncfile->numBatches << "\n";
		ss << "sparse_point_offset: " << ncfile->sparse_point_offset << "\n";

		cout << ss.str() << endl;
	}

	{ // create load tasks

		int64_t pointOffset = 0;

		while (pointOffset < ncfile->numPoints) {

			int64_t remaining = ncfile->numPoints - pointOffset;
			int64_t pointsInBatch = min(int64_t(MAX_POINTS_PER_BATCH), remaining);

			LoadTask task;
			task.file = ncfile;
			task.firstPoint = pointOffset;
			task.numPoints = pointsInBatch;

			loadTasks.push_back(task);

			pointOffset += pointsInBatch;

			shared_ptr<LoadResult> result = nullptr;
			result = loadNC(task.file, task.firstPoint, task.numPoints);

			UploadTask uploadTask;
			uploadTask.file = task.file;
			uploadTask.sparse_pointOffset = result->sparse_pointOffset;
			uploadTask.numPoints = task.numPoints;
			uploadTask.numBatches = result->numBatches;
			uploadTask.bXyzLow = result->bXyzLow;
			uploadTask.bXyzMed = result->bXyzMed;
			uploadTask.bXyzHig = result->bXyzHig;
			uploadTask.bColors = result->bColors;
			uploadTask.bVel = result->bVel;
			uploadTask.bBatches = result->bBatches;

			// UPLOAD DATA TO GPU

			{ // commit physical memory in sparse buffers
				int64_t offset = 4 * uploadTask.sparse_pointOffset;
				int64_t pageAlignedOffset = offset - (offset % PAGE_SIZE);

				int64_t size = 4 * uploadTask.numPoints;
				int64_t pageAlignedSize = size - (size % PAGE_SIZE) + PAGE_SIZE;
				pageAlignedSize = std::min(pageAlignedSize, 4 * MAX_POINTS);

				for (auto glBuffer : { ssXyzLow, ssXyzMed, ssXyzHig, ssColors }) {
					glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBuffer.handle);
					glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, pageAlignedOffset, pageAlignedSize, GL_TRUE);
					glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
				}

				// Special case for velocity buffer
				int64_t velOffset = 4 * 3 * uploadTask.sparse_pointOffset;
				int64_t velPageAlignedOffset = velOffset - (velOffset % PAGE_SIZE);

				int64_t velSize = 4 * 3 * uploadTask.numPoints;
				int64_t velPageAlignedSize = velSize - (velSize % PAGE_SIZE) + PAGE_SIZE;
				velPageAlignedSize = std::min(velPageAlignedSize, 4 * 3 * MAX_POINTS);

				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssVel.handle);
				glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, velPageAlignedOffset, velPageAlignedSize, GL_TRUE);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}

			// upload batch metadata
			glNamedBufferSubData(ssBatches.handle,
				64 * this->numBatchesLoaded,
				uploadTask.bBatches->size,
				uploadTask.bBatches->data);

			// upload batch points
			glNamedBufferSubData(ssXyzLow.handle, 4 * uploadTask.sparse_pointOffset, 4 * uploadTask.numPoints, uploadTask.bXyzLow->data);
			glNamedBufferSubData(ssXyzMed.handle, 4 * uploadTask.sparse_pointOffset, 4 * uploadTask.numPoints, uploadTask.bXyzMed->data);
			glNamedBufferSubData(ssXyzHig.handle, 4 * uploadTask.sparse_pointOffset, 4 * uploadTask.numPoints, uploadTask.bXyzHig->data);
			glNamedBufferSubData(ssColors.handle, 4 * uploadTask.sparse_pointOffset, 4 * uploadTask.numPoints, uploadTask.bColors->data);
			glNamedBufferSubData(ssVel.handle, 4 * 3 * uploadTask.sparse_pointOffset, 4 * 3 * uploadTask.numPoints, uploadTask.bVel->data);

			this->numBatchesLoaded += uploadTask.numBatches;
			this->numPointsLoaded += uploadTask.numPoints;
			uploadTask.file->numPointsLoaded += uploadTask.numPoints;
		}
	}

	dvec3 boxMin = { Infinity, Infinity, Infinity };
	dvec3 boxMax = { -Infinity, -Infinity, -Infinity };

	boxMin = glm::min(boxMin, ncfile->boxMin);
	boxMax = glm::max(boxMax, ncfile->boxMax);





	// zoom to point cloud
	/*auto size = boxMax - boxMin;
	auto position = (boxMax + boxMin) / 2.0;
	auto radius = glm::length(size) / 1.5;

	renderer->controls->yaw = 0.53;
	renderer->controls->pitch = -0.68;
	renderer->controls->radius = radius;
	renderer->controls->target = position;*/
}