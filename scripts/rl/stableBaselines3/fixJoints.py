import numpy as np


def fix_joints(free_joints=1):

    joint_map = {
        'bottom-pos' : 1,
        'middle-pos' : 3,
        # ...
    }

    initial_positions = np.array([
        # 7 initial values
    ])

    i = 0
    for joint_name, joint_idx in joint_map:
        initial_positions[joint_idx] = np.random.uniform(-1,1)
        i += 1
        if i > free_joints:
            break

