import numpy as np
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape


def make_grid():
    ''' Shows a grid of points delimitin the space of action in the x an y directions
    '''

    type = PrimitiveShape.CUBOID
    r, g, b = 1, 0, 0
    w, h, d = 0.01, 0.01, 0.01
    z = 0.5

    center = [-0.1, 0.1]
    side = 0.15
    poses = np.array([[0,  0], [-1,  0], [0, -1], [1,  0], [0,  1], [-1,  1],
                      [1,  -1], [1, 1], [-1, -1]])*side + center

    handles = []

    for pos in poses:
        object_handle = Shape.create(type=type, color=[r, g, b],
                                     size=[w, h, d], static=True,
                                     respondable=False,
                                     position=[pos[0], pos[1], z])
        handles.append(object_handle)

    return handles, poses
