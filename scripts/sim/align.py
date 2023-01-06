import numpy as np
from quat_to_euler import quat_to_euler

aim = [-180, -90, 0, 90, 180]
hole_actions = [11, 10, 9]
peg_actions = [5, 3, 4]

def if_angle_equal(angls, angl):
    for i in angls:
        if ((i - 10) <= angl) and ((i + 10) >= angl):
            return True
    return False

def align(plane, env, direction, robot="peg"):
    if robot=="hole":
        quat = "hole_quat"
        actions = hole_actions
    else:
        quat = "peg_quat"
        actions = peg_actions
    angles = quat_to_euler(env._observables[quat].obs[0], env._observables[quat].obs[1], env._observables[quat].obs[2], env._observables[quat].obs[3])
    print(angles)
    angle = angles[plane]
    i = 0
    while ((not if_angle_equal(aim, angle)) and i < 80):
        action = np.zeros(12)
        angles = quat_to_euler(env._observables[quat].obs[0], env._observables[quat].obs[1], env._observables[quat].obs[2], env._observables[quat].obs[3])
        angle = angles[plane]    
        action[actions[plane]] = 0.15 * direction
        print(action)
        print(angles)
        env.step(action)
        env.render()
        i += 1
