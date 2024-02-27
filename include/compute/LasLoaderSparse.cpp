#include "LasLoaderSparse.h"
#include "unsuck.hpp"

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

mutex mtx_debug;

struct LoadResult {
	shared_ptr<Buffer> bBatches;
	shared_ptr<Buffer> bXyzLow;
	shared_ptr<Buffer> bXyzMed;
	shared_ptr<Buffer> bXyzHig;
	shared_ptr<Buffer> bColors;
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

shared_ptr<LoadResult> loadLas(shared_ptr<LasFile> lasfile, int64_t firstPoint, int64_t numPoints) {

	string path = lasfile->path;
	int64_t file_byteOffset = lasfile->offsetToPointData + firstPoint * lasfile->bytesPerPoint;
	int64_t file_byteSize = numPoints * lasfile->bytesPerPoint;
	auto source = readBinaryFile(path, file_byteOffset, file_byteSize);
	int64_t sparse_pointOffset = lasfile->sparse_point_offset + firstPoint;

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

	dvec3 boxMin = lasfile->boxMin;
	dvec3 cScale = lasfile->scale;
	dvec3 cOffset = lasfile->offset;

	// load batches/points
	for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
		Batch& batch = batches[batchIndex];

		// compute batch bounding box
		for (int i = 0; i < batch.numPoints; i++) {
			int index_pointFile = batch.chunk_pointOffset + i;

			int32_t X = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 0);
			int32_t Y = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 4);
			int32_t Z = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 8);

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
			bBatches->set<uint32_t>(lasfile->fileIndex, batchByteOffset + 36);
		}

		int offset_rgb = 0;
		if (lasfile->pointFormat == 2) {
			offset_rgb = 20;
		}
		else if (lasfile->pointFormat == 3) {
			offset_rgb = 28;
		}
		else if (lasfile->pointFormat == 7) {
			offset_rgb = 30;
		}
		else if (lasfile->pointFormat == 8) {
			offset_rgb = 30;
		}

		// load data
		for (int i = 0; i < batch.numPoints; i++) {
			int index_pointFile = batch.chunk_pointOffset + i;

			int32_t X = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 0);
			int32_t Y = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 4);
			int32_t Z = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 8);

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


				int R = source->get<uint16_t>(index_pointFile * lasfile->bytesPerPoint + offset_rgb + 0);
				int G = source->get<uint16_t>(index_pointFile * lasfile->bytesPerPoint + offset_rgb + 2);
				int B = source->get<uint16_t>(index_pointFile * lasfile->bytesPerPoint + offset_rgb + 4);

				R = R < 256 ? R : R / 256;
				G = G < 256 ? G : G / 256;
				B = B < 256 ? B : B / 256;

				uint32_t color = R | (G << 8) | (B << 16);

				bColors->set<uint32_t>(color, 4 * index_pointFile);
			}
		}
	}

	auto result = make_shared<LoadResult>();
	result->bXyzLow = bXyzLow;
	result->bXyzMed = bXyzMed;
	result->bXyzHig = bXyzHig;
	result->bColors = bColors;
	result->bBatches = bBatches;
	result->numBatches = numBatches;
	result->sparse_pointOffset = sparse_pointOffset;

	return result;
}

LasLoaderSparse::LasLoaderSparse(string lfname) {

	int pageSize = 0;
	glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &pageSize);
	PAGE_SIZE = pageSize;

	{ // create (sparse) buffers
		this->ssBatches = createBuffer(64 * 200'000);
		this->ssXyzLow = createSparseBuffer(4 * MAX_POINTS);
		this->ssXyzMed = createSparseBuffer(4 * MAX_POINTS);
		this->ssXyzHig = createSparseBuffer(4 * MAX_POINTS);
		this->ssColors = createSparseBuffer(4 * MAX_POINTS);
		this->ssLoadBuffer = createBuffer(200 * MAX_POINTS_PER_BATCH);

		GLuint zero = 0;
		glClearNamedBufferData(this->ssBatches.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}

	addFile(lfname);
}

void LasLoaderSparse::addFile(string lfname)
{
	auto lasfile = make_shared<LasFile>();
	lasfile->path = lfname;

	auto buffer_header = readBinaryFile(lasfile->path, 0, 375);

	int versionMajor = buffer_header->get<uint8_t>(24);
	int versionMinor = buffer_header->get<uint8_t>(25);

	if (versionMajor == 1 && versionMinor < 4) {
		lasfile->numPoints = buffer_header->get<uint32_t>(107);
	}
	else {
		lasfile->numPoints = buffer_header->get<uint64_t>(247);
	}

	lasfile->numPoints = min(lasfile->numPoints, 1'000'000'000ll);

	lasfile->offsetToPointData = buffer_header->get<uint32_t>(96);
	lasfile->pointFormat = buffer_header->get<uint8_t>(104) % 128;
	lasfile->bytesPerPoint = buffer_header->get<uint16_t>(105);

	lasfile->scale.x = buffer_header->get<double>(131);
	lasfile->scale.y = buffer_header->get<double>(139);
	lasfile->scale.z = buffer_header->get<double>(147);

	lasfile->offset.x = buffer_header->get<double>(155);
	lasfile->offset.y = buffer_header->get<double>(163);
	lasfile->offset.z = buffer_header->get<double>(171);

	lasfile->boxMin.x = buffer_header->get<double>(187);
	lasfile->boxMin.y = buffer_header->get<double>(203);
	lasfile->boxMin.z = buffer_header->get<double>(219);

	lasfile->boxMax.x = buffer_header->get<double>(179);
	lasfile->boxMax.y = buffer_header->get<double>(195);
	lasfile->boxMax.z = buffer_header->get<double>(211);

	{
		lasfile->numBatches = lasfile->numPoints / POINTS_PER_WORKGROUP + 1;

		lasfile->sparse_point_offset = numPoints;

		files.push_back(lasfile);
		numPoints += lasfile->numPoints;
		numBatches += lasfile->numBatches;

		stringstream ss;
		ss << "load file " << lfname << endl;
		ss << "numPoints: " << lasfile->numPoints << "\n";
		ss << "numBatches: " << lasfile->numBatches << "\n";
		ss << "sparse_point_offset: " << lasfile->sparse_point_offset << "\n";

		cout << ss.str() << endl;
	}

	{ // create load tasks

		int64_t pointOffset = 0;

		while (pointOffset < lasfile->numPoints) {

			int64_t remaining = lasfile->numPoints - pointOffset;
			int64_t pointsInBatch = min(int64_t(MAX_POINTS_PER_BATCH), remaining);

			LoadTask task;
			task.file = lasfile;
			task.firstPoint = pointOffset;
			task.numPoints = pointsInBatch;

			loadTasks.push_back(task);

			pointOffset += pointsInBatch;

			shared_ptr<LoadResult> result = nullptr;
			result = loadLas(task.file, task.firstPoint, task.numPoints);

			UploadTask uploadTask;
			uploadTask.file = task.file;
			uploadTask.sparse_pointOffset = result->sparse_pointOffset;
			uploadTask.numPoints = task.numPoints;
			uploadTask.numBatches = result->numBatches;
			uploadTask.bXyzLow = result->bXyzLow;
			uploadTask.bXyzMed = result->bXyzMed;
			uploadTask.bXyzHig = result->bXyzHig;
			uploadTask.bColors = result->bColors;
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

			this->numBatchesLoaded += uploadTask.numBatches;
			this->numPointsLoaded += uploadTask.numPoints;
			uploadTask.file->numPointsLoaded += uploadTask.numPoints;
		}
	}

	dvec3 boxMin = { Infinity, Infinity, Infinity };
	dvec3 boxMax = { -Infinity, -Infinity, -Infinity };

	boxMin = glm::min(boxMin, lasfile->boxMin);
	boxMax = glm::max(boxMax, lasfile->boxMax);
}