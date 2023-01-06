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

RESET_MODE    = 'default'  # default, fixed_dimensions, limit_velocity
STEP_MODE     = 'default'  # default, primitive
VEC           = True
ACTION_LIM    = 30
ACTION_DIM    = 13 if STEP_MODE== 'primitive' else 12
ENV_HORIZON   = 50
AGENT_HORIZON = int(ENV_HORIZON / (ACTION_LIM / 2))  # This is necessarily an approximation
NUM_VEC_ENVS  = 1