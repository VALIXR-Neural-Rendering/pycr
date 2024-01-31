#include <pybind11/pybind11.h>

#include "RenderUtils.h"

using namespace std;
namespace py = pybind11;


void export_utils(py::module_ m)
{
	m.def("initGL", &initGL);
}