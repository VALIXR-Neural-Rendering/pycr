#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Framebuffer.h"
#include "Texture.h"

using namespace std;
namespace py = pybind11;

void export_texture(py::module_ m)
{
	py::class_<Texture, shared_ptr<Texture>>(*m, "Texture")
		.def(py::init<>())
		.def("create", &Texture::create)
		.def("setSize", &Texture::setSize)
		.def("getPixels", &Texture::getPixels);
}

void export_fb(py::module_ m)
{
	py::class_<Framebuffer, shared_ptr<Framebuffer>>(*m, "Framebuffer")
		.def(py::init<>())
		.def_readwrite("colorAttachments", &Framebuffer::colorAttachments)
		.def_readwrite("colorBuffers", &Framebuffer::colorBuffers)
		.def_readwrite("depth", &Framebuffer::depth)
		.def_readwrite("handle", &Framebuffer::handle)
		.def_readwrite("width", &Framebuffer::width)
		.def_readwrite("height", &Framebuffer::height)
		.def("create", &Framebuffer::create)
		.def("setSize", &Framebuffer::setSize)
		.def("savePixels", &Framebuffer::savePixels);
}