#pragma once

#include "Texture.h"


shared_ptr<Texture> Texture::create(int width, int height, GLuint colorType){

	GLuint handle;
	glCreateTextures(GL_TEXTURE_2D, 1, &handle);

	auto texture = make_shared<Texture>();
	texture->handle = handle;
	texture->colorType = colorType;

	texture->setSize(width, height);

	return texture;
}

void Texture::setSize(int width, int height) {

	bool needsResize = this->width != width || this->height != height;

	if (needsResize) {

		glDeleteTextures(1, &this->handle);
		glCreateTextures(GL_TEXTURE_2D, 1, &this->handle);

		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(this->handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTextureStorage2D(this->handle, 1, this->colorType, width, height);

		this->width = width;
		this->height = height;
	}

}

vector <GLubyte> Texture::getPixels()
{
	vector<GLubyte> buf(this->width * this->height * 4);
	glBindImageTexture(0, this->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
	glGetTextureImage(this->handle, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->width * this->height * 4, buf.data());
	return buf;
}