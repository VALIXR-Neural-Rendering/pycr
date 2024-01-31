#include <pybind11/pybind11.h>

#include "compute_loop/compute_loop.h"
#include "compute/LasLoaderSparse.h"
#include "compute/NCLoaderSparse.h"

using namespace std;
namespace py = pybind11;

void export_las_loader(py::module_ m)
{
	py::class_<LasLoaderSparse, shared_ptr<LasLoaderSparse>>(*m, "LasLoader")
		.def(py::init<string>());
}

void export_nc_loader(py::module_ m)
{
	py::class_<NCLoaderSparse, shared_ptr<NCLoaderSparse>>(*m, "NCLoader")
		.def(py::init<string>());
}

void export_compute_loop(py::module_ m)
{
	py::class_<ComputeLoop<LasLoaderSparse>>(*m, "ComputeLoopLas")
		.def(py::init<shared_ptr<LasLoaderSparse>>())
		.def_readwrite("source", &ComputeLoop<LasLoaderSparse>::source)
		.def_readwrite("csRender", &ComputeLoop<LasLoaderSparse>::csRender)
		.def_readwrite("csResolve", &ComputeLoop<LasLoaderSparse>::csResolve)
		.def_readwrite("outFramePixels", &ComputeLoop<LasLoaderSparse>::outFramePixels)
		.def("getFrmTensor", &ComputeLoop<LasLoaderSparse>::transferFrame)
		.def("update", &ComputeLoop<LasLoaderSparse>::update)
		.def("render", &ComputeLoop<LasLoaderSparse>::render);

	py::class_<ComputeLoop<NCLoaderSparse>>(*m, "ComputeLoopNC")
		.def(py::init<shared_ptr<NCLoaderSparse>>())
		.def_readwrite("source", &ComputeLoop<NCLoaderSparse>::source)
		.def_readwrite("csRender", &ComputeLoop<NCLoaderSparse>::csRender)
		.def_readwrite("csResolve", &ComputeLoop<NCLoaderSparse>::csResolve)
		.def_readwrite("outFramePixels", &ComputeLoop<NCLoaderSparse>::outFramePixels)
		.def("getFrmTensor", &ComputeLoop<NCLoaderSparse>::transferFrame)
		.def("update", &ComputeLoop<NCLoaderSparse>::update)
		.def("render", &ComputeLoop<NCLoaderSparse>::render);
}