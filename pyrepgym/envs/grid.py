import numpy as np
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import itertools


def make_grid():
    ''' Shows a grid of points delimitin the space of action in the x an y directions
    '''

    type = PrimitiveShape.CUBOID
    r, g, b = 1, 0, 0
    w, h, d = 0.01, 0.01, 0.01
    z = 0.5

    xs = [-0.35, -0.1, -0.05, 0, 0.1, 0.2, 0.3, 0.4]
    ys = np.linspace(-0.19, 0.4, 7)
    poses = [p for p in itertools.product(xs, ys)]

    handles = []

    for pos in poses:
        object_handle = Shape.create(type=type, color=[r, g, b],
                                     size=[w, h, d], static=True,
                                     respondable=False,
                                     position=[pos[0], pos[1], z])
        handles.append(object_handle)

    return handles, poses
