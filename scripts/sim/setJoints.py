import numpy as np


def set_joints(free_joints=1, robot="peg"):

    joint_map = [
        ('bottom-pos', 1),
        ('middle-pos', 3),
        ('upper-pos' , 5),
        ('bottom-rot', 0),
        ('middle-rot', 2),
        ('upper-rot' , 4),
        ('end-rot'   , 6)
    ]
    
    if robot=="peg":
        initial_positions = np.array([0, -0.4, 0, -1, 0, 1.7, 0])
    elif robot=="hole":
        initial_positions = np.array([0, -0.4, 0, -1, 0, 1.7, -0.6])
    else:
        raise("Error: robot should be peg or hole for set_joints function")

    # initial_positions = np.array([0, -0.4, 0, -1, 0, 1.7, -0.6])

    for joint_name, joint_idx in joint_map[:free_joints]:
        initial_positions[joint_idx] = np.random.uniform(-0.1,0.1)

    return initial_positions