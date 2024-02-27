#pragma once

////  global compilation flag configuring windows sdk headers
////  preventing inclusion of min and max macros clashing with <limits>
//#define NOMINMAX 1
//
////  override byte to prevent clashes with <cstddef>
//#define byte win_byte_override
//#include <Windows.h>

#include "unsuck.hpp"
#include "GLBuffer.h"
#include "Framebuffer.h"
#include "Texture.h"

using namespace std;

inline void error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

inline void initGL() {

	//if (wglGetCurrentContext() == NULL){
	//	cout << "no context" << endl;
	//}

	//glfwSetErrorCallback(error_callback);
	//if (!glfwInit()) {
	//	// Initialization failed
	//}
	
	//window = glfwGetCurrentContext();

	//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_DECORATED, true);
	//glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	//int numMonitors;
	//GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

	//cout << "<create windows>" << endl;
	//{
	//	//const GLFWvidmode* mode = glfwGetVideoMode(monitors[0]);
	//	window = glfwCreateWindow(width, height, "", nullptr, nullptr);


	//	if (!window) {
	//		int code = glfwGetError(nullptr);
	//		cout << "glfw error: " << code << endl;
	//		glfwTerminate();
	//		exit(EXIT_FAILURE);
	//	}

	//	glfwSetWindowPos(window, 50, 50);
	//}

	//glfwMakeContextCurrent(window);
	//glfwSwapInterval(0);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "glew error: %s\n", glewGetErrorString(err));
	}

	cout << "<glewInit done> " << "(" << now() << ")" << endl;
}

inline GLBuffer createBuffer(int64_t size) {
	GLuint handle;
	glCreateBuffers(1, &handle);
	glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);

	GLBuffer buffer;
	buffer.handle = handle;
	buffer.size = size;

	return buffer;
}

inline GLBuffer createSparseBuffer(int64_t size) {
	GLuint handle;
	glCreateBuffers(1, &handle);
	glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_SPARSE_STORAGE_BIT_ARB);

	GLBuffer buffer;
	buffer.handle = handle;
	buffer.size = size;

	return buffer;
}

inline GLBuffer createUniformBuffer(int64_t size) {
	GLuint handle;
	glCreateBuffers(1, &handle);
	glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT);

	GLBuffer buffer;
	buffer.handle = handle;
	buffer.size = size;

	return buffer;
}

inline shared_ptr<Buffer> readBuffer(GLBuffer glBuffer, uint32_t offset, uint32_t size) {

	auto target = make_shared<Buffer>(size);

	glGetNamedBufferSubData(glBuffer.handle, offset, size, target->data);

	return target;
}

//inline int64_t getAvailableGpuMemory(){
//	GLint available = 0;
//	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &available);
//}

inline shared_ptr<Texture> createTexture(int width, int height, GLuint colorType) {

	auto texture = Texture::create(width, height, colorType);

	return texture;
}

inline shared_ptr<Framebuffer> createFramebuffer(int width, int height) {

	auto framebuffer = Framebuffer::create();
	//framebuffer->setSize(width, height);

	GLenum status = glCheckNamedFramebufferStatus(framebuffer->handle, GL_FRAMEBUFFER);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		cout << "framebuffer incomplete" << endl;
	}

	return framebuffer;
}