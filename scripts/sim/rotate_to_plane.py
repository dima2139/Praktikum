import numpy as np

aimhole = [[0.5, -0.5, 0.5, -0.5], [0, -0.7, 0.7, 0], [-0.7, -0.7, 0, 0]]
aimpeg = [[0.5, -0.5, 0.5, -0.5], [0, -0.7, 0.7, 0], [-0.7, -0.7, 0, 0]]
hole_actions = [[11, -1, +1], [11, 1, -1], [10, -1, 1], [9, -1, 1]]
peg_actions = [[5, -1, +1], [5, 1, -1], [4, -1, 1], [3, -1, 1]]

def rotate_to_plane(plane, env, robot="peg"):
    if robot=="hole":
        quat = "hole_quat"
        aim = aimhole
        actions = hole_actions
    else:
        quat = "peg_quat"
        aim = aimpeg
        actions = peg_actions
    obs = env.obs = env._observables[quat].obs
    i = 0
    while ((abs(obs[0] - aim[plane][0]) >= 0.05) or (abs(obs[1] - aim[plane][1]) >= 0.05) or (abs(obs[2] - aim[plane][2]) >= 0.05) or (abs(obs[3] - aim[plane][3]) >= 0.05)):
        while ((abs(obs[i] - aim[plane][i]) >= 0.05)):
            action = np.zeros(12)
            obs = env.obs = env._observables[quat].obs
            if (obs[i] - aim[plane][i]) > 0:
                action[actions[i][0]] = actions[i][1] * 0.15
            else:
                action[actions[i][0]] = actions[i][2] * 0.15
            print(action)
            print(obs)
            env.step(action)
            env.render()
        i = (i + 1) % 4
