#pragma once

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include <memory>
#include <vector>

using namespace std;

struct Renderer;

struct Texture {

	GLuint handle = -1;
	GLuint colorType = -1;
	int width = 0;
	int height = 0;

	static shared_ptr<Texture> create(int width, int height, GLuint colorType);

	void setSize(int width, int height);

	vector <GLubyte> getPixels();
};