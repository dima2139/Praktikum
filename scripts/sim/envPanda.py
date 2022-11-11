import numpy as np
import tf_agents as tfa

from scripts.const import *
from scripts.utils import *


class envPanda(tfa.environments.py_environment.PyEnvironment):
    def __init__(self, envName, params, dtype):
        '''
        Instantiate a Panda RL environment.
        '''


        # Define Environment Parameters
        self.envName = envName
        self.params  = params
        self.dtype   = dtype

        # Define RL parameters
        self.discount = params['gamma']

        # Define the observation specification
        self._observation_spec = tfa.specs.array_spec.BoundedArraySpec(
            shape = (28 + 17,),  # 28 + 17 + 17
            dtype = dtype,
            # minimum = []
        )