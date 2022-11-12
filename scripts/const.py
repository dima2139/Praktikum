LOC       = '/home/dve/Desktop/g2-peg-in-hole'
scriptDIR = f'{LOC}/scripts'
modelsDir = f'{LOC}/models'

SEED_TF = 69

Amin = [-3.750 , -3.750 , -3.750 , -0.150 , -0.150 , -0.150 , -3.750 , -3.750 , -3.750 , -0.150 , -0.150 , -0.150]
Amax = [+3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150 , +3.750 , +3.750 , +3.750 , +0.150 , +0.150 , +0.150]

RPmin = [-1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -1.000 , -8.000 , -8.000 , -8.000 , -8.000 , -8.000 , -8.000 , -8.000 , -3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000]
RPmax = [+1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +8.000 , +8.000 , +8.000 , +8.000 , +8.000 , +8.000 , +8.000 , +3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000]

OBmin = [-3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000 , -3.000 , -3.000 , -3.000 , -1.000 , -1.000 , -1.000 , -1.000 , -0.000 , -3.000 , -3.000]
OBmax = [+3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000 , +3.000 , +3.000 , +3.000 , +1.000 , +1.000 , +1.000 , +1.000 , +1.000 , +3.000 , +3.000]

OBSmin = RPmin + OBmin
OBSmax = RPmax + OBmax

SUCCESS = 0.15