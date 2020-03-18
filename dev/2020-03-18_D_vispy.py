# import cv2
# video_capture = cv2.VideoCapture(0)
#
#
# def grab(DS=4):
#     ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     return frame[::DS, ::DS, ::-1]# Find all the faces in the current frame of video
#
# rgb_frame = grab()
# N_X, N_Y, three = rgb_frame.shape
#
# import dlib
# import numpy as np
# class FaceExtractor:
#     def __init__(self, N_X, N_Y):
#         import dlib
#         self.detector = dlib.get_frontal_face_detector()
#         self.screen_size = N_X**2 + N_Y**2
#
#     def center_normalized(self, frame):
#         N_X, N_Y, three = frame.shape
#         dets = self.detector(frame, 1)
#         if len(dets) > 0:
#             bbox = dets[0]
#             t, b, l, r = bbox.top(), bbox.bottom(), bbox.left(), bbox.right()
#             x, y, s = t + (b-t)/2, l + (r-l)/2, (b-t)**2 + (r-l)**2
#             return 2*x/N_X - 1, 2*y/N_Y - 1, s / self.screen_size
#         else:
#             return 0, 0, 0
#
# f = FaceExtractor(N_X, N_Y)
#
# import time
# while True:
#     tic = time.time()
#
#     # Grab a single frame of video
#     ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     rgb_frame = grab()
#
#     # detect face and extract position
#     x, y, s = f.center_normalized(rgb_frame)
#     print(f'x, y, s (norm) = {x:.3f}, {y:.3f}, {s:.3f}')
#     toc = time.time()
#
#     print(f'FPS:{1/(toc-tic):.1f}')

# https://github.com/vispy/vispy/blob/master/examples/tutorial/gl/cube.py
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# Author: Nicolas P .Rougier
# Date:   04/03/2014
# -----------------------------------------------------------------------------
# import math
# import numpy as np
#
# from vispy import app
# from vispy.gloo import gl
#
#
# def checkerboard(grid_num=8, grid_size=32):
#     row_even = grid_num // 2 * [0, 1]
#     row_odd = grid_num // 2 * [1, 0]
#     Z = np.row_stack(grid_num // 2 * (row_even, row_odd)).astype(np.uint8)
#     return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)
#
#
# def rotate(M, angle, x, y, z, point=None):
#     angle = math.pi * angle / 180
#     c, s = math.cos(angle), math.sin(angle)
#     n = math.sqrt(x * x + y * y + z * z)
#     x /= n
#     y /= n
#     z /= n
#     cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
#     R = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, 0],
#                   [cx * y + z * s, cy * y + c, cz * y - x * s, 0],
#                   [cx * z - y * s, cy * z + x * s, cz * z + c, 0],
#                   [0, 0, 0, 1]], dtype=M.dtype).T
#     M[...] = np.dot(M, R)
#     return M
#
#
# def translate(M, x, y=None, z=None):
#     y = x if y is None else y
#     z = x if z is None else z
#     T = np.array([[1.0, 0.0, 0.0, x],
#                   [0.0, 1.0, 0.0, y],
#                   [0.0, 0.0, 1.0, z],
#                   [0.0, 0.0, 0.0, 1.0]], dtype=M.dtype).T
#     M[...] = np.dot(M, T)
#     return M
#
#
# def frustum(left, right, bottom, top, znear, zfar):
#     M = np.zeros((4, 4), dtype=np.float32)
#     M[0, 0] = +2.0 * znear / (right - left)
#     M[2, 0] = (right + left) / (right - left)
#     M[1, 1] = +2.0 * znear / (top - bottom)
#     M[3, 1] = (top + bottom) / (top - bottom)
#     M[2, 2] = -(zfar + znear) / (zfar - znear)
#     M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
#     M[2, 3] = -1.0
#     return M
#
#
# def perspective(fovy, aspect, znear, zfar):
#     h = math.tan(fovy / 360.0 * math.pi) * znear
#     w = h * aspect
#     return frustum(-w, w, -h, h, znear, zfar)
#
#
# def makecube():
#     """ Generate vertices & indices for a filled cube """
#
#     vtype = [('a_position', np.float32, 3),
#              ('a_texcoord', np.float32, 2)]
#     itype = np.uint32
#
#     # Vertices positions
#     p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
#                   [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]])
#
#     # Texture coords
#     t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
#
#     faces_p = [0, 1, 2, 3, 0, 3, 4, 5, 0, 5, 6,
#                1, 1, 6, 7, 2, 7, 4, 3, 2, 4, 7, 6, 5]
#     faces_t = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
#                3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
#
#     vertices = np.zeros(24, vtype)
#     vertices['a_position'] = p[faces_p]
#     vertices['a_texcoord'] = t[faces_t]
#
#     indices = np.resize(
#         np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
#     indices += np.repeat(4 * np.arange(6), 6).astype(np.uint32)
#
#     return vertices, indices
#
#
# cube_vertex = """
# uniform mat4 u_model;
# uniform mat4 u_view;
# uniform mat4 u_projection;
# attribute vec3 a_position;
# attribute vec2 a_texcoord;
# varying vec2 v_texcoord;
# void main()
# {
#     gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
#     v_texcoord = a_texcoord;
# }
# """
#
# cube_fragment = """
# uniform sampler2D u_texture;
# varying vec2 v_texcoord;
# void main()
# {
#     gl_FragColor = texture2D(u_texture, v_texcoord);
# }
# """
#
#
# class Canvas(app.Canvas):
#     def __init__(self):
#         app.Canvas.__init__(self, size=(512, 512),
#                             title='Rotating cube (GL version)',
#                             keys='interactive')
#
#     def on_initialize(self, event):
#         # Build & activate cube program
#         self.cube = gl.glCreateProgram()
#         vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
#         fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
#         gl.glShaderSource(vertex, cube_vertex)
#         gl.glShaderSource(fragment, cube_fragment)
#         gl.glCompileShader(vertex)
#         gl.glCompileShader(fragment)
#         gl.glAttachShader(self.cube, vertex)
#         gl.glAttachShader(self.cube, fragment)
#         gl.glLinkProgram(self.cube)
#         gl.glDetachShader(self.cube, vertex)
#         gl.glDetachShader(self.cube, fragment)
#         gl.glUseProgram(self.cube)
#
#         # Get data & build cube buffers
#         vcube_data, self.icube_data = makecube()
#         vcube = gl.glCreateBuffer()
#         gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vcube)
#         gl.glBufferData(gl.GL_ARRAY_BUFFER, vcube_data, gl.GL_STATIC_DRAW)
#         icube = gl.glCreateBuffer()
#         gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, icube)
#         gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
#                         self.icube_data, gl.GL_STATIC_DRAW)
#
#         # Bind cube attributes
#         stride = vcube_data.strides[0]
#         offset = 0
#         loc = gl.glGetAttribLocation(self.cube, "a_position")
#         gl.glEnableVertexAttribArray(loc)
#         gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)
#
#         offset = vcube_data.dtype["a_position"].itemsize
#         loc = gl.glGetAttribLocation(self.cube, "a_texcoord")
#         gl.glEnableVertexAttribArray(loc)
#         gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)
#
#         # Create & bind cube texture
#         crate = checkerboard()
#         texture = gl.glCreateTexture()
#         gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
#                            gl.GL_LINEAR)
#         gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
#                            gl.GL_LINEAR)
#         gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
#                            gl.GL_CLAMP_TO_EDGE)
#         gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
#                            gl.GL_CLAMP_TO_EDGE)
#         gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_LUMINANCE, gl.GL_LUMINANCE,
#                         gl.GL_UNSIGNED_BYTE, crate.shape[:2])
#         gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, gl.GL_LUMINANCE,
#                            gl.GL_UNSIGNED_BYTE, crate)
#         loc = gl.glGetUniformLocation(self.cube, "u_texture")
#         gl.glUniform1i(loc, texture)
#         gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
#
#         # Create & bind cube matrices
#         view = np.eye(4, dtype=np.float32)
#         model = np.eye(4, dtype=np.float32)
#         projection = np.eye(4, dtype=np.float32)
#         translate(view, 0, 0, -7)
#         self.phi, self.theta = 60, 20
#         rotate(model, self.theta, 0, 0, 1)
#         rotate(model, self.phi, 0, 1, 0)
#         loc = gl.glGetUniformLocation(self.cube, "u_model")
#         gl.glUniformMatrix4fv(loc, 1, False, model)
#         loc = gl.glGetUniformLocation(self.cube, "u_view")
#         gl.glUniformMatrix4fv(loc, 1, False, view)
#         loc = gl.glGetUniformLocation(self.cube, "u_projection")
#         gl.glUniformMatrix4fv(loc, 1, False, projection)
#
#         # OpenGL initalization
#         gl.glClearColor(0.30, 0.30, 0.35, 1.00)
#         gl.glEnable(gl.GL_DEPTH_TEST)
#         self._resize(*(self.size + self.physical_size))
#         self.timer = app.Timer('auto', self.on_timer, start=True)
#
#     def on_draw(self, event):
#         gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
#         gl.glDrawElements(gl.GL_TRIANGLES, self.icube_data.size,
#                           gl.GL_UNSIGNED_INT, None)
#
#     def on_resize(self, event):
#         self._resize(*(event.size + event.physical_size))
#
#     def _resize(self, width, height, physical_width, physical_height):
#         gl.glViewport(0, 0, physical_width, physical_height)
#         projection = perspective(35.0, width / float(height), 2.0, 10.0)
#         loc = gl.glGetUniformLocation(self.cube, "u_projection")
#         gl.glUniformMatrix4fv(loc, 1, False, projection)
#
#     def on_timer(self, event):
#         self.theta += .5
#         self.phi += .5
#         model = np.eye(4, dtype=np.float32)
#         rotate(model, self.theta, 0, 0, 1)
#         rotate(model, self.phi, 0, 1, 0)
#         loc = gl.glGetUniformLocation(self.cube, "u_model")
#         gl.glUniformMatrix4fv(loc, 1, False, model)
#         self.update()
#
# c = Canvas()
# c.show()
# app.run()


# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# https://github.com/vispy/vispy/blob/master/examples/basics/visuals/box.py
"""
Simple demonstration of Box visual.
"""

from vispy import app, gloo, visuals
from vispy.geometry import create_box
from vispy.visuals.transforms import MatrixTransform


class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 550))

        vertices, faces, outline = create_box(width=1, height=1, depth=1,
                                              width_segments=4,
                                              height_segments=8,
                                              depth_segments=16)

        self.box = visuals.BoxVisual(width=1, height=1, depth=1,
                                     width_segments=4,
                                     height_segments=8,
                                     depth_segments=16,
                                     vertex_colors=vertices['color'],
                                     edge_color='b')

        self.theta = 0
        self.phi = 0

        self.transform = MatrixTransform()

        self.box.transform = self.transform
        self.show()

        self.timer = app.Timer(connect=self.rotate)
        self.timer.start(0.016)

    def rotate(self, event):
        self.theta += .5
        self.phi += .5
        self.transform.reset()
        self.transform.rotate(self.theta, (0, 0, 1))
        self.transform.rotate(self.phi, (0, 1, 0))
        self.transform.scale((100, 100, 0.001))
        self.transform.translate((200, 200))
        self.update()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        self.box.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, ev):
        gloo.clear(color='white', depth=True)
        self.box.draw()

win = Canvas()
import sys
if sys.flags.interactive != 1:
    app.run()
