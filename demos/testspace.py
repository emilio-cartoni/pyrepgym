for h in handles:
    h.remove()

from gym import spaces
xmin = -0.35 -0.8
xmax = -0.10 -0.8
ymin = -0.19 -0.3
ymax = 0.39 -0.3

asp = spaces.Box(
                          low=np.array([[xmin, ymin], [xmin, ymin]]),
                          high=np.array([[xmax, ymax], [xmax, ymax]]),
                          dtype=float)
handles, poses = make_grid(asp)


env.robot.goto_joint('LEFT_ARM', np.zeros((1,7)), np.array([3]))
env.robot.wait_for_goto('LEFT_ARM')


env.objects['cube'].set_pose(pose=[-0.9 -0.1, env.cube_on_table, 0, 0, 0, 1])
env.robot.goto_joint('LEFT_ARM', np.zeros((1,7)), np.array([3]))
env.robot.wait_for_goto('LEFT_ARM')


