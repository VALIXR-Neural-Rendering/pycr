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