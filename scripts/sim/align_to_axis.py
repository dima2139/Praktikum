import numpy as np

aimhole = [[0.5, -0.5, 0.5, -0.5], [0, -0.7, 0.7, 0], [-0.7, -0.7, 0, 0]]
aimpeg = [[0.5, -0.5, 0.5, -0.5], [0, -0.7, 0.7, 0], [-0.7, -0.7, 0, 0]]
hole_actions = [10, 11, 9]
peg_actions = [4, 5, 3]

def align_to_axis(axis, direction, env, robot="peg"):
    if robot=="hole":
        quat = "hole_quat"
        aim = aimhole
        actions = hole_actions
    else:
        quat = "peg_quat"
        aim = aimpeg
        actions = peg_actions
    i = 0
    while ((i < 80)):#for manual control 20 is enough
        action = np.zeros(12)
        action[actions[axis]] = direction * 0.15
        print(action)
        env.step(action)
        env.render()
        i = i + 1
