'''
0.15
Defines constants.
Author: Daniel von Eschwege
Date:   9 November 1022
'''

import numpy as np


LOC       = '/home/dve/Desktop/g2-peg-in-hole'
scriptDIR = f'{LOC}/scripts'
modelsDir = f'{LOC}/models'

DTYPE = np.float32
SEED  = 69

Amin       = [-0.975 , -0.975 , -0.975 , -0.150 , -0.150 , -0.150]
Amax       = [+0.975 , +0.975 , +0.975 , +0.150 , +0.150 , +0.150]
Amax12     = [+0.975 , +0.975 , +0.975 , +0.150 , +0.150 , +0.150, +0.975 , +0.975 , +0.975 , +0.150 , +0.150 , +0.150]
Amax6      = [+0.975 , +0.975 , +0.975 , +0.975 , +0.975 , +0.975]
Amax13     = [+0.975 , +0.975 , +0.975 , +0.150 , +0.150 , +0.150, +0.975 , +0.975 , +0.975 , +0.150 , +0.150 , +0.150, +0.0]
Amax13Init = [+0.975 , +0.975 , +0.975 , +0.450 , +0.450 , +0.450, +0.975 , +0.975 , +0.975 , +0.450 , +0.450 , +0.450, +0.0]
# A          = [-0.975, +0.975, -0.975, +0.975, -0.975, +0.975, -0.150, +0.150, -0.150, +0.150, -0.150, +0.150]
# A          = [+0.975, +0.975, +0.975, +0.975, +0.975, +0.975, +0.300, +0.300, +0.300, +0.300, +0.300, +0.300]

A2min = [-0.975 , -0.975 , -0.975 , -0.150 , -0.150 , -0.150 , -0.975 , -0.975 , -0.975 , -0.150 , -0.150 , -0.150]
A2max = [+0.975 , +0.975 , +0.975 , +0.150 , +0.150 , +0.150 , +0.975 , +0.975 , +0.975 , +0.150 , +0.150 , +0.150]

RPSmin = [-1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -10.000 , -10.000 , -10.000 , -10.000 , -10.000 , -10.000 , -10.000 , -3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000]
RPSmax = [+1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +10.000 , +10.000 , +10.000 , +10.000 , +10.000 , +10.000 , +10.000 , +3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000]

OSmin = [-3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000 , -3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000 , -0.000 , -3.000 , -3.000]
OSmax = [+3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000 , +3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +3.000 , +3.000]

Omin = RPSmin + RPSmin + OSmin
Omax = RPSmax + RPSmax + OSmax

RESET_MODE       = 'check_velocity'  # default, fixed_dimensions, limit_velocity, check_velocity
MOVEMENT_EPSILON = 0.5
STEP_MODE        = 'default'  # default, action
VEC              = True
ACTION_LIM       = 30
ACTION_DIM       = 12

PRIMITIVE_ALIGN_HORIZON = 100
PRIMITIVE_D_HORIZON     = 50
PRIMITIVE_T_HORIZON     = 50
# ENV_HORIZON             = PRIMITIVE_ALIGN_HORIZON + PRIMITIVE_D_HORIZON + PRIMITIVE_T_HORIZON
ENV_HORIZON             = 250

AGENT_HORIZON    = int(ENV_HORIZON / (ACTION_LIM / 2))  # This is necessarily an approximation
NUM_VEC_ENVS     = 4
PRIMITIVE        = 't'
QUAT_ANGLES_PEG  = np.array([0.5, 0.5, 0.5, -0.5])
QUAT_ANGLES_HOLE = np.array([0, -0.7071, 0.7071, 0])
BBOX_PEG         = np.array([[-0.2, +0.2], [-0.4, -0.2], [+1.5, +1.9]])
BBOX_HOLE        = np.array([[-0.2, +0.2], [+0.2, +0.4], [+1.3, +1.7]])
MIN_BBOX_PEG     = BBOX_PEG[:, 0]
MAX_BBOX_PEG     = BBOX_PEG[:, 1]
MIN_BBOX_HOLE    = BBOX_HOLE[:, 0]
MAX_BBOX_HOLE    = BBOX_HOLE[:, 1]