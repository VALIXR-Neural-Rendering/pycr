
#pragma once

#include <vector>
#include <memory>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "Texture.h"

using namespace std;

struct Framebuffer {

	vector<shared_ptr<Texture>> colorAttachments;
	vector< vector<GLubyte> > colorBuffers;
	shared_ptr<Texture> depth;
	GLuint handle = -1;

	int width = 0;
	int height = 0;

	Framebuffer() {
		
	}

	static shared_ptr<Framebuffer> create();

	void setSize(int width, int height);

	void savePixels(int index);
};