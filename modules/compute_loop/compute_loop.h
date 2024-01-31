#pragma once
#pragma warning(disable : 4996)

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <string>
#include <queue>
#include <vector>
#include <thread>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>


#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <glm/gtc/type_ptr.hpp>

#include "unsuck.hpp"
#include "Shader.h"
#include "GLBuffer.h"
#include "Method.h"
#include "GLTimerQueries.h"
#include "Debug.h"
#include "RenderUtils.h"
#include "compute/LasLoaderSparse.h"
#include "compute/NCLoaderSparse.h"

using namespace std;
using namespace std::chrono_literals;
namespace py = pybind11;

using glm::dmat4;
using glm::mat4;
using glm::ivec2;


inline mat4 cast2mat4(py::array_t<float> arr) {
	if (arr.ndim() != 2)
		throw std::runtime_error("Input should be 2-D NumPy array");
	auto arr_buf = arr.request();
	if (arr_buf.size != 16)
		throw std::runtime_error("Input should be a 4x4 matrix");
	float* ptr = (float*)arr_buf.ptr;
	mat4 mat = glm::make_mat4(ptr);
	return mat;
}

template<typename LoaderType>
struct ComputeLoop : public Method{

	struct UniformData {
		mat4 world;
		mat4 view;
		mat4 proj;
		mat4 transform;
		mat4 transformFrustum;
		int pointsPerThread;
		int enableFrustumCulling;
		int showBoundingBox;
		int numPoints;
		ivec2 imageSize;
		int colorizeChunks;
		int colorizeOverdraw;
	};

	struct DebugData {
		uint32_t value = 0;
		bool enabled = false;
		uint32_t numPointsProcessed = 0;
		uint32_t numNodesProcessed = 0;
		uint32_t numPointsRendered = 0;
		uint32_t numNodesRendered = 0;
		uint32_t numPointsVisible = 0;
	};

	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;
	//GLFWwindow* window = nullptr;

	GLBuffer ssFramebuffer;
	GLBuffer ssDebug;
	GLBuffer ssBoundingBoxes;
	GLBuffer ssFiles;
	GLBuffer uniformBuffer;
	UniformData uniformData;
	shared_ptr<Buffer> ssFilesBuffer;

	shared_ptr<LoaderType> las = nullptr;
	vector<GLubyte> outFramePixels;
	bool texReg = false;
	struct cudaGraphicsResource* cuda_tex_screen_resource;
	cudaArray_t cuda_tex_screen = nullptr;
	void* pycuda_tex = nullptr;

	ComputeLoop(shared_ptr<LoaderType> las) {

		initGL();

		this->name = "loop_las";
		this->description = R"ER01(
- Each thread renders X points.
- Loads points from LAS file
- encodes point coordinates in 10+10+10 bits
- Workgroup picks 10, 20 or 30 bit precision
  depending on screen size of bounding box
		)ER01";
		this->las = las;
		this->group = "10-10-10 bit encoded";

		csRender = new Shader({ {"../modules/compute_loop/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"../modules/compute_loop/resolve.cs", GL_COMPUTE_SHADER} });

		ssFramebuffer = createBuffer(8 * 2048 * 2048);

		ssFilesBuffer = make_shared<Buffer>(10'000 * 128);

		ssDebug = createBuffer(256);
		ssBoundingBoxes = createBuffer(48 * 1'000'000);
		ssFiles = createBuffer(ssFilesBuffer->size);
		uniformBuffer = createUniformBuffer(512);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssBoundingBoxes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssFiles.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}

	~ComputeLoop() {
		if (pycuda_tex) {
			//checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_screen_resource));
			checkCudaErrors(cudaFreeArray(cuda_tex_screen));
			checkCudaErrors(cudaFree(pycuda_tex));
		}
	}

	void saveFrame(shared_ptr<Framebuffer> fbo) {
		vector<GLubyte> buf(fbo->width * fbo->height * 4);
		glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
		glGetTextureImage(fbo->colorAttachments[0]->handle, 0, GL_RGBA, GL_UNSIGNED_BYTE, fbo->width * fbo->height * 4, buf.data());
		outFramePixels = buf;

		// debug
		///*float meanpix = accumulate(outFramePixels.begin(), outFramePixels.end(), 0) / (float)(outFramePixels.size());
		//cout << "Mean pixels: " << meanpix << endl;*/
	}

	torch::Tensor transferFrame(py::object frmbuf) {
		auto fbo = py::cast<shared_ptr<Framebuffer>>(frmbuf);
		glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_screen_resource));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_tex_screen, cuda_tex_screen_resource, 0, 0));

		cudaChannelFormatDesc desc;
		cudaExtent extent;
		checkCudaErrors(cudaArrayGetInfo(&desc, &extent, NULL, cuda_tex_screen));
		int fmtSize = sizeof(GLubyte);
		size_t width = extent.width;
		size_t height = extent.height;
		
		if (pycuda_tex == nullptr)
			checkCudaErrors(cudaMalloc(&pycuda_tex, width * fmtSize * 4 * height));

		checkCudaErrors(cudaMemcpy2DFromArray(pycuda_tex, width * fmtSize * 4, cuda_tex_screen, 0, 0, width * fmtSize * 4, height, cudaMemcpyDefault));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_screen_resource, 0));

		auto texture_tensor = torch::from_blob(pycuda_tex, { long(4 * extent.width), long(extent.height) }, torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(torch::kCUDA));

		return texture_tensor;
	}

	void update() {

	}

	void render(const py::dict& rview) {

		GLTimerQueries::timestamp("compute-loop-start");

		// store view parameters in separate variables here
		py::array_t<float> rview_view = py::cast<py::array_t<float>>(rview["view"]);
		auto view_mat = glm::transpose(cast2mat4(rview_view));
		py::array_t<float> rview_proj = py::cast<py::array_t<float>>(rview["proj"]);
		auto proj_mat = cast2mat4(rview_proj);

		/*las->process();

		if (las->numPointsLoaded == 0) {
			return;
		}*/

		auto fbo = py::cast<shared_ptr<Framebuffer>>(rview["framebuffer"]);

		// REGISTER TEXTURE ON GPU
		if (!texReg)
		{
			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
			checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_screen_resource, fbo->colorAttachments[0]->handle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
			texReg = true;
		}

		// resize framebuffer storage, if necessary
		if (ssFramebuffer.size < 8 * fbo->width * fbo->height) {

			glDeleteBuffers(1, &ssFramebuffer.handle);

			// make new buffer a little larger to have some reserves when users enlarge the window
			int newBufferSize = 1.5 * double(8 * fbo->width * fbo->height);

			ssFramebuffer = createBuffer(newBufferSize);
		}

		// Update Uniform Buffer
		{
			mat4 world;
			mat4 view = view_mat;
			mat4 proj = proj_mat;
			mat4 worldView = view * world;
			mat4 worldViewProj = proj * view * world;

			//cout << "view: " << glm::to_string(view) << std::endl;
			//cout << "proj: " << glm::to_string(proj) << std::endl;
			//cout << "world: " << glm::to_string(worldViewProj) << std::endl;

			uniformData.world = world;
			uniformData.view = view;
			uniformData.proj = proj;
			uniformData.transform = worldViewProj;
			if (Debug::updateFrustum) {
				uniformData.transformFrustum = worldViewProj;
			}
			uniformData.pointsPerThread = POINTS_PER_THREAD;
			uniformData.numPoints = las->numPointsLoaded;
			uniformData.enableFrustumCulling = Debug::frustumCullingEnabled ? 1 : 0;
			uniformData.showBoundingBox = Debug::showBoundingBox ? 1 : 0;
			uniformData.imageSize = { fbo->width, fbo->height };
			uniformData.colorizeChunks = Debug::colorizeChunks;
			uniformData.colorizeOverdraw = Debug::colorizeOverdraw;

			glNamedBufferSubData(uniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		{ // update file buffer

			for (int i = 0; i < las->files.size(); i++) {
				auto lasfile = las->files[i];

				dmat4 world = glm::translate(dmat4(), lasfile->boxMin);
				dmat4 view = view_mat;
				dmat4 proj = proj_mat;
				dmat4 worldView = view * world;
				dmat4 worldViewProj = proj * view * world;

				mat4 transform = worldViewProj;
				mat4 fWorld = world;

				memcpy(
					ssFilesBuffer->data_u8 + 256 * lasfile->fileIndex + 0,
					glm::value_ptr(transform),
					64);

				if (Debug::updateFrustum) {
					memcpy(
						ssFilesBuffer->data_u8 + 256 * lasfile->fileIndex + 64,
						glm::value_ptr(transform),
						64);
				}

				memcpy(
					ssFilesBuffer->data_u8 + 256 * lasfile->fileIndex + 128,
					glm::value_ptr(fWorld),
					64);

			}

			glNamedBufferSubData(ssFiles.handle, 0, 256 * las->files.size(), ssFilesBuffer->data);
		}

		if (Debug::enableShaderDebugValue) {
			DebugData data;
			data.enabled = true;

			glNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);
		}

		// RENDER
		if (csRender->program != -1) {
			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, las->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyzHig.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyzMed.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyzLow.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 45, ssFiles.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			// int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			int numBatches = las->numBatchesLoaded;

			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("draw-end");
		}

		// RESOLVE
		if (csResolve->program != -1) {
			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = ceil(float(fbo->width) / 16.0f);
			int groups_y = ceil(float(fbo->height) / 16.0f);
			glDispatchCompute(groups_x, groups_y, 1);

			GLTimerQueries::timestamp("resolve-end");
		}

		// READ DEBUG VALUES
		if (Debug::enableShaderDebugValue) {
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			DebugData data;
			glGetNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);

			auto dbg = Debug::getInstance();

			dbg->pushFrameStat("#nodes processed", formatNumber(data.numNodesProcessed));
			dbg->pushFrameStat("#nodes rendered", formatNumber(data.numNodesRendered));
			dbg->pushFrameStat("#points processed", formatNumber(data.numPointsProcessed));
			dbg->pushFrameStat("#points rendered", formatNumber(data.numPointsRendered));
			dbg->pushFrameStat("divider", "");
			dbg->pushFrameStat("#points visible", formatNumber(data.numPointsVisible));

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		// BOUNDING BOXES
		/*if (Debug::showBoundingBox) {
			glMemoryBarrier(GL_ALL_BARRIER_BITS);
			glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);

			auto camera = renderer->camera;
			renderer->drawBoundingBoxes(camera.get(), ssBoundingBoxes);
		}*/

		{ // CLEAR
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			// Save the current viewport
			//saveFrame(fbo);

			GLuint zero = 0;
			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);

			glClearNamedBufferSubData(ssFramebuffer.handle, GL_R32UI, 0, fbo->width * fbo->height * 8, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferSubData(ssBoundingBoxes.handle, GL_R32UI, 0, 48, GL_RED, GL_UNSIGNED_INT, &zero);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		GLTimerQueries::timestamp("compute-loop-end");
	}


};