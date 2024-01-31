from glumpy import app, gloo, gl

from contextlib import contextmanager
import numpy as np

try:
    import pycuda.driver
    from pycuda.gl import graphics_map_flags, BufferObject
    _PYCUDA = True
except ImportError as err:
    print('pycuda import error:', err)
    _PYCUDA = False

import torch


@contextmanager
def cuda_activate_array(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0,0)
    mapping.unmap()


@contextmanager
def cuda_activate_buffer(buf):
    mapping = buf.map()
    yield mapping.device_ptr()
    mapping.unmap()


def create_shared_texture(arr, map_flags=None):
    """Create and return a Texture2D with gloo and pycuda views."""

    if map_flags is None:
        map_flags = graphics_map_flags.WRITE_DISCARD
    
    gl_view = arr.view(gloo.TextureFloat2D)
    gl_view.activate() # force gloo to create on GPU
    gl_view.deactivate()

    cuda_view = pycuda.gl.RegisteredImage(
        int(gl_view.handle), gl_view.target, map_flags)

    return gl_view, cuda_view


def create_shared_buffer(arr):
    """Create and return a BufferObject with gloo and pycuda views."""
    gl_view = arr.view(gloo.VertexBuffer)
    gl_view.activate() # force gloo to create on GPU
    gl_view.deactivate()
    cuda_view = BufferObject(np.long(gl_view.handle))
    return gl_view, cuda_view


def cpy_texture_to_tensor(texture, tensor):
    """Copy GL texture (cuda view) to pytorch tensor"""
    with cuda_activate_array(texture) as src:
        cpy = pycuda.driver.Memcpy2D()

        cpy.set_src_array(src)
        cpy.set_dst_device(tensor.data_ptr())
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tensor.shape[1] * 4 * 4  
        cpy.height = tensor.shape[0] 
        cpy(aligned=False)

        torch.cuda.synchronize()

    return tensor


def cpy_tensor_to_texture(tensor, texture):
    """Copy pytorch tensor to GL texture (cuda view)"""
    with cuda_activate_array(texture) as ary:
        cpy = pycuda.driver.Memcpy2D()

        cpy.set_src_device(tensor.data_ptr())
        cpy.set_dst_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tensor.shape[1] * 4 * 4
        cpy.height = tensor.shape[0]
        cpy(aligned=False)

        torch.cuda.synchronize()

    return tensor


def cpy_buffer_to_tensor(buffer, tensor):
    """Copy GL buffer (cuda view) to pytorch tensor"""
    n = tensor.numel()*tensor.element_size()    
    with cuda_activate_buffer(buffer) as buf_ptr:
        pycuda.driver.memcpy_dtod(tensor.data_ptr(), buf_ptr, n)


def cpy_tensor_to_buffer(tensor, buffer):
    """Copy pytorch tensor to GL buffer (cuda view)"""
    n = tensor.numel()*tensor.element_size()    
    with cuda_activate_buffer(buffer) as buf_ptr:
        pycuda.driver.memcpy_dtod(buf_ptr, tensor.data_ptr(), n)  


def cpy_tex_addr_to_texture(src_text_addr, dst_texture):
    """Copy GL texture (as address) to GL texture (cuda view)"""
    with cuda_activate_array(dst_texture) as dst:
        cpy = pycuda.driver.Memcpy2D()

        cpy.set_src_device(src_text_addr)
        cpy.set_dst_array(dst)
        # cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = dst_texture.shape[1] * 4 * 4  
        # cpy.height = dst_texture.shape[0] 
        cpy(aligned=False)

        torch.cuda.synchronize()