#include "RenderUtils.h"
#include <pybind11/pybind11.h>

#include "compute/LasLoaderSparse.h"
#include "compute/NCLoaderSparse.h"


namespace py = pybind11;

void export_las_loader(py::module_ m);
void export_nc_loader(py::module_ m);
void export_compute_loop(py::module_ m);
void export_fb(py::module_ m);
void export_texture(py::module_ m);


PYBIND11_MODULE(pycr, m) {
	m.doc() = "Python wrapper for the CPP implementation of Compute Rasterizer";
	m.def("init_GL", &initGL);

	py::module_ compute_loop = m.def_submodule("compute_loop",
		"Loop optimization implementation for point structures");
	export_las_loader(compute_loop);
	export_nc_loader(compute_loop);
	export_compute_loop(compute_loop);

	py::module_ globj = m.def_submodule("globj",
		"OpenGL objects for Compute Rasterizer");
	export_fb(globj);
	export_texture(globj);
}


/*
TODO:

compute_loop:
LasLoaderSparse:
	1. implement zoom to point cloud
*/