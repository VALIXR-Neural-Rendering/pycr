README
========

## pycr - Python Wrapper for Compute Rasterizer

This software is a python wrapper for the [Compute Rasterizer](https://github.com/m-schuetz/compute_rasterizer) project. It provides a python interface to the Compute Rasterizer, and also provides a viewer to visualize the point clouds (currently supporting NetCDF and LAS formats). The viewer is based on [OpenGL](https://www.opengl.org/) and [GLFW](https://www.glfw.org/). 

## Prerequisites

Running this software needs the following dependencies:
1. CUDA 12.1
2. OpenGL 4.6
3. Python 3.10
4. LibTorch 2.1.1+cu121 (native installations both debug and release modes)
5. GLM
6. GLFW
7. GLEW
8. Pybind11 (could be installed from `vcpkg`)
9. NetCDF-C
10. PyCuda

All of the above external libraries are provided in the `libs` folder. The CUDA and OpenGL versions are the ones used for development. The software may work with other versions of CUDA and OpenGL, but it is not tested.

Both the debug and release versions of PyTorch are not included in the `libs` folder. You can download them from [here](https://pytorch.org/get-started/locally/). Refer to this [blog](https://towardsdatascience.com/setting-up-a-c-project-in-visual-studio-2019-with-libtorch-1-6-ad8a0e49e82c) for more details on how to setup LibTorch in Visual Studio.

To install PyCuda, run the following command in the command prompt (assuming you are in the project root directory):
```bash
git clone https://github.com/inducer/pycuda
cd pycuda
git submodule update --init
export PATH=$PATH:/usr/local/cuda/bin
./configure.py --cuda-enable-gl
python setup.py install
cd ..
```

## Running the software

1. Open `pycr` -> `pycr.sln` in Visual Studio (tested on VS2022)
2. Run the solution (in Release-x64 mode)

## Documentation of `pycr`

`pycr` provides the following python modules:
 - `pycr.compute_loop`: Compute loop implementation of Compute Rasterizer. Currently supports only vanilla version of the `compute_loop`. It provides the following classes:
    - `pycr.LasLoader(lfname -> str)`: LAS file loader.
            lfname: LAS file path
    - `pycr.NCLoader(ncfname -> str)`: NetCDF file loader.
            ncfname: NetCDF file path
    - `pycr.compute_loop.ComputeLoopLas`: Compute loop implementation for handling LAS files. (Checkout `computeLoop_wrap.cpp` for the exposed APIs)
    - `pycr.compute_loop.ComputeLoopNC`: Compute loop implementation for handling NetCDF files. (Checkout `computeLoop_wrap.cpp` for the exposed APIs)
 - `pycr.globj`: OpenGL object implementation of Compute Rasterizer. It provides the following classes:
    - `pycr.globj.Framebuffer`: OpenGL framebuffer object. (Checkout `globj_wrap.cpp` for the exposed APIs)
    - `pycr.globj.Texture`: OpenGL texture object. (Checkout `globj_wrap.cpp` for the exposed APIs)


`pycr` provides the following python functions:
 - `pycr.init_GL()`: Initializes OpenGL context

Also checkout the `render_app.py` for a sample usage of the above modules and functions.

## Viewer Controls

On running the software, it opens up a viewer to visualize the point cloud of your choice. You can set the point cloud to visualize by changing the `--inpf` flag of the python script `render_app.py`. The viewer has the following controls:

* `Left Mouse Button` + `Mouse Movement`: Rotate the camera
* `Mouse Wheel`: Zoom in/out
* `S` : Save the current view to `render_app/data/screenshot`

Also check out the `--help` flag of the python script `render_app.py` for more options. The flags could be added in the property pages of the python project in Visual Studio.