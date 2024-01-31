
#pragma once

#include <string>
#include <pybind11/pybind11.h>

using std::string;
namespace py = pybind11;

struct Renderer;

struct Method{

	string name = "no name";
	string description = "";
	// int group = 0;
	string group = "no group";

	Method(){

	}

	virtual void update() = 0;
	virtual void render(const py::dict& rview) = 0;

};