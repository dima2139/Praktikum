'''
0.15
Defines constants.
Author: Daniel von Eschwege
Date:   9 November 2022
'''

import numpy as np


LOC       = '/home/dve/Desktop/g2-peg-in-hole'
scriptDIR = f'{LOC}/scripts'
modelsDir = f'{LOC}/models'

DTYPE = np.float32
SEED  = 69

Amin       = [-3.750 , -3.750 , -3.750 , -0.150 , -0.150 , -0.150]
Amax       = [+3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150]
Amax12     = [+3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150, +3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150]
Amax13     = [+3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150, +3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150, +0.0]
Amax13Init = [+3.750 , +3.750 , +3.750 , +0.450 , +0.450 , +0.450, +3.750 , +3.750 , +3.750 , +0.450 , +0.450 , +0.450, +0.0]
# A          = [-3.750, +3.750, -3.750, +3.750, -3.750, +3.750, -0.150, +0.150, -0.150, +0.150, -0.150, +0.150]
A          = [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

A2min = [-3.750 , -3.750 , -3.750 , -0.150 , -0.150 , -0.150 , -3.750 , -3.750 , -3.750 , -0.150 , -0.150 , -0.150]
A2max = [+3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150 , +3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150]

RPSmin = [-1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -20.000 , -20.000 , -20.000 , -20.000 , -20.000 , -20.000 , -20.000 , -3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000]
RPSmax = [+1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +20.000 , +20.000 , +20.000 , +20.000 , +20.000 , +20.000 , +20.000 , +3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000]

OSmin = [-3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000 , -3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000 , -0.000 , -3.000 , -3.000]
OSmax = [+3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000 , +3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +3.000 , +3.000]

Omin = RPSmin + RPSmin + OSmin
Omax = RPSmax + RPSmax + OSmax

VEC           = True
ACTION_DIM    = 13
ACTION_LIM    = 30
ENV_HORIZON   = 200
AGENT_HORIZON = int(ENV_HORIZON / (ACTION_LIM / 2))  # This is necessarily an approximation