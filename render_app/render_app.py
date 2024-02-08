import os
import argparse
import threading
import time
from tkinter import SEL
import numpy as np
from glumpy import app, gl, gloo, data, transforms
import glfw
import cv2

try:
    import pycuda.driver
    _PYCUDA = True
except ImportError as err:
    print('pycuda import error:', err)
    _PYCUDA = False

from camera import Trackball
from utils import extrinsics_from_view_matrix, extrinsics_from_xml
from gpu_utils import create_shared_texture, cpy_tensor_to_texture, cpy_tex_addr_to_texture, cpy_texture_to_tensor

import torch
import pycr
from pycr.compute_loop import LasLoader, NCLoader, ComputeLoopLas, ComputeLoopNC
from pycr.globj import Texture, Framebuffer

import pdb


def get_args():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--inpf', type=str, default="C:/UMD/render_pc/data/nasa/h_common.nc4", help='LAS/NC4 file path')
    parser.add_argument('--inpf', type=str, default="C:/UMD/render_pc/data/nasa/hurricane_hd.las", help='LAS/NC4 file path')
    
    parser.add_argument('--viewport', type=str, default='1820,980', help='width,height') # '3200,2000' for server screen
    parser.add_argument('--rmode', choices=['trackball', 'fly'], default='trackball')
    parser.add_argument('--fps', action='store_true', help='show fps')
    parser.add_argument('--replay-camera', type=str, default='', help='path to view_matrix to replay at given fps')
    parser.add_argument('--replay-fps', type=float, default=2., help='view_matrix replay fps')
    parser.add_argument('--clear-color', type=str, default='1,1,1', help='initial background color for renderer')
    args = parser.parse_args()

    args.viewport = tuple([int(x) for x in args.viewport.split(',')]) if args.viewport else None
    args.clear_color = [float(x) for x in args.clear_color.split(',')] if args.clear_color else None

    return args

def get_screen_program(texture):
    vertex = '''
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = <transform>;
        v_texcoord = texcoord;
    } '''
    fragment = '''
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    } '''

    quad = gloo.Program(vertex, fragment, count=4)
    quad["transform"] = transforms.OrthographicProjection(transforms.Position("position"))
    quad['texcoord'] = [( 0, 0), ( 0, 1), ( 1, 0), ( 1, 1)]
    quad['texture'] = texture

    return quad

def start_fps_job():
    def job():
        print(f'FPS {app.clock.get_fps():.1f}')

    threading.Timer(1.0, job).start()

def load_camera_trajectory(path):
    if path[-3:] == 'xml':
        view_matrix, camera_labels = extrinsics_from_xml(path)
    else:
        view_matrix, camera_labels = extrinsics_from_view_matrix(path)
    return view_matrix

def fix_viewport_size(viewport_size, factor=16):
    viewport_w = factor * (viewport_size[0] // factor)
    viewport_h = factor * (viewport_size[1] // factor)
    return viewport_w, viewport_h


class MyApp():
    def __init__(self, args):
        self.viewport_size = fix_viewport_size(args.viewport)
        print('new viewport size ', self.viewport_size)

        init_view = np.array([
           [ 9.64632000e-01, -2.63601000e-01,  5.30000000e-05,
        -9.10121620e+01],
           [-2.60912000e-01, -9.54764000e-01,  1.42659000e-01,
             8.94545460e+02],
           [ 3.75550000e-02,  1.37627000e-01,  9.89772000e-01,
            -1.46066213e+05],
           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             1.00000000e+00]
        ])
        init_proj = np.array([
            [0.932642758, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 1.7320508075688774, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -1.0000010000005000, -1.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.20000010000005000, 0.0000000000000000]
        ])
        self.trackball = Trackball(init_view, self.viewport_size, 1, target=[576.91, 886.62, 10.35], rotation_mode=args.rmode)

        # this also creates GL context necessary for setting up shaders
        # app.use("glfw")
        self.window = app.Window(width=self.viewport_size[0], height=self.viewport_size[1], visible=True, fullscreen=False)
        self.window.set_size(*self.viewport_size)
        pycr.init_GL()

        # pycuda initialization
        assert _PYCUDA, 'pycuda is not available'
        try:
            import pycuda.gl.autoinit  # this may fails in headless mode
        except:
            raise RuntimeError('PyCUDA init failed, cannot use torch buffer')
        _ = torch.rand((1, 3, 512,512), dtype=torch.float32, device='cuda') # needs init here, otherwise does not work

        screen_tex, self.screen_tex_cuda = create_shared_texture(
            np.zeros((self.viewport_size[1], self.viewport_size[0],4), np.float32))

        self.screen_program = get_screen_program(screen_tex)

        # Initialize compute loop
        if args.inpf.endswith("nc4"):
            self.floader = NCLoader(args.inpf)
            self.compute_loop = ComputeLoopNC(self.floader)
        elif args.inpf.endswith('las'):
            self.floader = LasLoader(args.inpf)
            self.compute_loop = ComputeLoopLas(self.floader)
        else:
            raise RuntimeError('Unknown file format')
        
        fb = Framebuffer.create()
        fb.setSize(self.viewport_size[0], self.viewport_size[1])
        tview = np.eye(4) #init_view.T
        tproj = init_proj.T
        self.render_view = dict(
                view=tview,
                proj=tproj,
                framebuffer=fb
            )
        self.compute_loop.render(self.render_view)

        # Initialize camera trajectory
        if args.replay_camera:
            self.camera_trajectory = load_camera_trajectory(args.replay_camera)
        else:
            self.camera_trajectory = None
    
        self.window.attach(self.screen_program['transform'])
        self.window.push_handlers(on_init=self.on_init)
        self.window.push_handlers(on_close=self.on_close)
        self.window.push_handlers(on_draw=self.on_draw)
        self.window.push_handlers(on_resize=self.on_resize)
        self.window.push_handlers(on_key_press=self.on_key_press)
        self.window.push_handlers(on_mouse_press=self.on_mouse_press)
        self.window.push_handlers(on_mouse_drag=self.on_mouse_drag)
        self.window.push_handlers(on_mouse_release=self.on_mouse_release)
        self.window.push_handlers(on_mouse_scroll=self.on_mouse_scroll)

        self.n_frame = 0
        self.t_elapsed = 0
        self.n_save_frame = 0
        self.last_frame = None
        self.last_view_matrix = None
        self.last_gt_image = None

        self.mouse_pressed = False

        self.args = args

    def run(self):
        if self.args.fps:
            start_fps_job()

        app.run()
        
    def render_frame(self, view_matrix):
        H, W = self.viewport_size[1], self.viewport_size[0]
        tview = view_matrix.T
        self.render_view['view'] = tview
        self.compute_loop.render(self.render_view)
        frame = self.compute_loop.getFrmTensor(self.render_view['framebuffer']).to(torch.float32)
        frame = frame.reshape(H, W, 4) / 255.0
        frame[:,:,3] = 1.0
        # pdb.set_trace()
        
        return frame

    def save_screen(self, out_dir='./data/screenshots'):
        os.makedirs(out_dir, exist_ok=True)
        img = cv2.flip(self.last_frame[:,:,:3].detach().cpu().numpy(), 0)*255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, str(self.n_save_frame) + '.png'), img)
        self.n_save_frame += 1
        
        with open(os.path.join(out_dir, 'pose.txt'), 'a') as f:
            np.savetxt(f, self.last_view_matrix)

    def get_next_view_matrix(self, frame_num, elapsed_time):
        if self.camera_trajectory is None:
            return self.trackball.pose
        
        n = int(elapsed_time * args.replay_fps) % len(self.camera_trajectory)
        if int(elapsed_time * args.replay_fps) >= len(self.camera_trajectory):
            n = 0

        return self.camera_trajectory[n]

    # ===== Window events =====

    def on_init(self):
        pass

    def on_key_press(self, symbol, modifiers):
        KEY_PLUS = 61
        if symbol == KEY_PLUS:
            self.point_size = self.point_size + 1
        elif symbol == glfw.KEY_MINUS:
            self.point_size = max(0, self.point_size - 1)
        elif symbol == glfw.KEY_S:
            self.save_screen()
        elif symbol == glfw.KEY_1:
            self.trackball.pose[2,3] += 100
        elif symbol == glfw.KEY_2:
            self.trackball.pose[2,3] -= 100
        else:
            print(symbol, modifiers)

    def on_draw(self, dt):
        self.last_view_matrix = self.get_next_view_matrix(self.n_frame, self.t_elapsed)
        self.last_frame = self.render_frame(self.last_view_matrix)
        
        # Copy the frame to screen texture
        if len(self.last_frame) > 0:
            cpy_tensor_to_texture(self.last_frame.detach().clone(), self.screen_tex_cuda)

        self.window.clear()
        gl.glDisable(gl.GL_CULL_FACE)

        # ensure viewport size is correct (offline renderer could change it)
        gl.glViewport(0, 0, self.viewport_size[0], self.viewport_size[1])
        self.screen_program.draw(gl.GL_TRIANGLE_STRIP)

        self.n_frame += 1
        self.t_elapsed += dt

    def on_resize(self, w, h):
        print(f'on_resize {w}x{h}')
        self.trackball.resize((w, h))
        self.screen_program['position'] = [(0, 0), (0, h), (w, 0), (w, h)]
        self.viewport_size = (w, h)

    def on_close(self):
        pass

    def on_mouse_press(self, x, y, buttons, modifiers=False):
        # print(buttons, modifiers)
        self.trackball.set_state(Trackball.STATE_ROTATE)
        if (buttons == app.window.mouse.LEFT):
            ctrl = (modifiers & app.window.key.MOD_CTRL)
            shift = (modifiers & app.window.key.MOD_SHIFT)
            if (ctrl and shift):
                self.trackball.set_state(Trackball.STATE_ZOOM)
            elif ctrl:
                self.trackball.set_state(Trackball.STATE_ROLL)
            elif shift:
                self.trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == app.window.mouse.MIDDLE):
            self.trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == app.window.mouse.RIGHT):
            self.trackball.set_state(Trackball.STATE_ZOOM)

        self.trackball.down(np.array([x, y]))

        # Stop animating while using the mouse
        self.mouse_pressed = True

    def on_mouse_drag(self, x, y, dx, dy, buttons):
        self.trackball.drag(np.array([x, y]))

    def on_mouse_release(self, x, y, buttons):
        self.mouse_pressed = False

    def on_mouse_scroll(self, x, y, dx, dy):
        self.trackball.scroll(dy)


if __name__ == '__main__':
    args = get_args()

    my_app = MyApp(args)
    my_app.run()

