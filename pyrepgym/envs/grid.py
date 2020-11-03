import numpy as np
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import itertools


def make_grid(box_action_space, points=3):
    ''' Shows a grid of points delimitin the space of action in the x an y directions
    '''

    type = PrimitiveShape.CUBOID
    r, g, b = 1, 0, 0
    w, h, d = 0.01, 0.01, 0.01
    z = 0.5

    low_x = box_action_space.low[0][0]
    high_x =  box_action_space.high[0][0]
    low_y = box_action_space.low[0][1]
    high_y =  box_action_space.high[0][1]

    xs = np.linspace(low_x, high_x, points)
    ys = np.linspace(low_y, high_y, points)
    poses = [p for p in itertools.product(xs, ys)]

    handles = []

    for pos in poses:
        object_handle = Shape.create(type=type, color=[r, g, b],
                                     size=[w, h, d], static=True,
                                     respondable=False,
                                     position=[pos[0], pos[1], z])
        handles.append(object_handle)

    return handles, poses
