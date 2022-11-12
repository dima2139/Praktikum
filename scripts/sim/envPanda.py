import numpy as np
import tf_agents as tfa

from scripts.const import *
from scripts.utils import *


class envPanda(tfa.environments.py_environment.PyEnvironment):
    def __init__(self, envName, params, dtype):
        '''
        Instantiate a Panda RL environment.
        '''

        # Environment parameters
        self.envName = envName
        self.params  = params
        self.dtype   = dtype

        # RL parameters
        self.discount = params['gamma']

        # Observation specification
        self._observation_spec = tfa.specs.array_spec.BoundedArraySpec(
            shape = (28 + 28 + 17,),
            dtype = dtype,
            minimum = OBSmin,
            maximum = OBSmax,
            name = 'observation'
        )

        # Action specification
        self._action_spec = tfa.specs.array_spec.BoundedArraySpec(
            shape   = (12,),
            dtype   = self.dtype,
            minimum = Amin,
            maximum = Amax,
            name    = 'action'
        )

    def setState(self):
        '''
        Set the observation state.
        '''