import numpy as np
import tf_agents as tfa
import robosuite as suite
from robosuite.devices import Keyboard
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils.input_utils import input2action

from scripts.const import *
from scripts.utils import *


class envPanda(tfa.environments.py_environment.PyEnvironment):
    def __init__(self, envName, params, dtype):
        '''
        Instantiate a Panda RL environment.
        '''

        # Environment parameters
        self.envName   = envName
        self.params    = params
        self.dtype     = dtype
        self.max_steps = 10

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
            # minimum = Amin,
            # maximum = Amax,
            minimum = -1,
            maximum = +1,
            name    = 'action'
        )


    def setState(self, obs):
        '''
        Set the observation state.
        '''

        self._state = obs

    
    def _reset(self):
        '''
        Reset the Robosuite environment.
        '''

        controller_config = load_controller_config(default_controller="OSC_POSE")
        config = {
            "env_name"          : "TwoArmPegInHole",
            "robots"            : ["Panda", "Panda"],
            "controller_configs": controller_config,
            "env_configuration" : "single-arm-opposed",
        }
        self.env = suite.make(
            **config,
            has_renderer           = True,
            has_offscreen_renderer = False,
            render_camera          = "agentview",
            ignore_done            = True,
            use_camera_obs         = False,
            reward_shaping         = True,
            control_freq           = 20,
            hard_reset             = False,
        )
        self.env = VisualizationWrapper(self.env, indicator_configs=None)
        # np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
        # device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
        # self.env.viewer.add_keypress_callback("any", device.on_press)
        # self.env.viewer.add_keyup_callback("any", device.on_release)
        # self.env.viewer.add_keyrepeat_callback("any", device.on_press)
        obs    = self.env.reset()
        cam_id = 0
        self.env.viewer.set_camera(camera_id=cam_id)
        self.env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # # Initialize device control
        # device.start_control()

        return tfa.trajectories.time_step.restart(self._state)

    
    def _step(self, action):

        # Check for termination
        if self._episode_ended:
            return self.reset()

        action = np.rint(action * self.max_steps)
        for n in max(action):
            pure_action = np.clip(action, -1, 1)
            action[action<0] += 1
            action[action>0] -= 1
            obs, reward, done, info = self.env.step(pure_action)