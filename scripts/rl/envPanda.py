'''
Environment / Interface between the Robosuite environment
and the Tensorflow Agents model training code.
Author: Daniel von Eschwege
Date:   20 November 2022
'''


## Imports
import time
import numpy as np
import tf_agents as tfa
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper

from scripts.const import *
from scripts.utils import *


class envPanda(tfa.environments.py_environment.PyEnvironment):
    def __init__(self, envName, params, eval=False):
        '''
        Instantiate a Panda RL environment.
        '''

        # Environment parameters
        self.envName              = envName
        self.params               = params
        self.eval                 = False
        self.max_steps_per_action = 10
        self.episode_counter      = 0

        # RL parameters
        self.discount = params['gamma']

        # Observation specification
        self._observation_spec = tfa.specs.array_spec.BoundedArraySpec(
            shape   = (28 + 28 + 17,),
            dtype   = DTYPE,
            minimum = Omin,
            maximum = Omax,
            name    = 'observation'
        )

        # Action specification
        self._action_spec = tfa.specs.array_spec.BoundedArraySpec(
            shape   = (12,),
            dtype   = DTYPE,
            minimum = -1,
            maximum = +1,
            name    = 'action'
        )

        controller_config = load_controller_config(default_controller="OSC_POSE")
       
        config = {
            "env_name"          : "TwoArmPegInHole",
            "robots"            : ["Panda", "Panda"],
            "controller_configs": controller_config,
            "env_configuration" : "single-arm-opposed",
        }

        if self.eval:
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
        else:
            self.env = suite.make(
                **config,
                has_renderer           = False,
                has_offscreen_renderer = False,
                ignore_done            = True,
                use_camera_obs         = False,
                reward_shaping         = True,
                control_freq           = 20,
                hard_reset             = False,
            )
        

    def setState(self, obs):
        '''
        Set the observation state.
        '''

        self._state = np.concatenate((
            obs['robot0_proprio-state'],
            obs['robot1_proprio-state'],
            obs['object-state']
        ), dtype=DTYPE)

    
    def _reset(self):
        '''
        Reset the Robosuite environment.
        '''

        self._episode_ended = False
        self.success        = False
        self.reward         = 0
        self.episode_reward = 0
        self.j              = 0

        obs = self.env.reset()
        self.setState(obs)
        
        if self.eval:
            cam_id = 0
            self.env.viewer.set_camera(camera_id=cam_id)
            self.env.render()

        return tfa.trajectories.time_step.restart(self._state)

    
    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self.j += 1

        if self.success:
            action = np.zeros((12,))
            if self.eval:
                self.env.render()
                time.sleep(5)
        else:
            action = np.rint(action * self.max_steps_per_action)
            for n in range(int(max(abs(action)))):
                pure_action = np.clip(action, -1, 1)
                action[action<0] += 1
                action[action>0] -= 1
                obs, self.reward, done, info = self.env.step(pure_action)
                if self.reward > 0.9:
                    self.success = True
                    break
                self.setState(obs)
                if self.eval:
                    self.env.render()

        self.episode_reward += self.reward

        if self.j == self.params['max_episode_steps']:
            self._episode_ended   = True
            self.episode_counter += 1
            self.env.close()
            pl(f'Episode {self.episode_counter} complete -- Episode reward: {self.episode_reward} -- Success: {self.success}')
            return tfa.trajectories.time_step.termination(
                observation = self._state,
                reward      = self.reward,
            )

        else:
            return tfa.trajectories.time_step.transition(
                observation = self._state,
                reward      = self.reward,
                discount    = self.discount
            )



    
    def action_spec(self):
        return self._action_spec

    
    def observation_spec(self):
        return self._observation_spec