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
        self._envName              = envName
        self._params               = params
        self._eval                 = eval
        self._max_steps_per_action = 4
        self._episode_counter      = 0

        # SAC parameters
        self._discount = params['gamma']

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
       
        self._config = {
            "env_name"          : "TwoArmPegInHole",
            "robots"            : ["Panda", "Panda"],
            "controller_configs": controller_config,
            "env_configuration" : "single-arm-opposed",
        }

        if self._eval:
            self._env = suite.make(
                **self._config,
                has_renderer           = True,
                has_offscreen_renderer = False,
                render_camera          = "agentview",
                ignore_done            = True,
                use_camera_obs         = False,
                reward_shaping         = True,
                control_freq           = 20,
                hard_reset             = False,
            )
            cam_id = 0
            self._env.viewer.set_camera(camera_id=cam_id)
            self._env.render()
        else:
            self._env = suite.make(
                **self._config,
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

        obs_concat = np.concatenate((
            obs['robot0_proprio-state'],
            obs['robot1_proprio-state'],
            obs['object-state']
        ))

        self._state = np.divide(obs_concat, Omax, dtype=DTYPE)

        assert np.max(np.abs(self._state)) <= 1.0, f'{obs_concat[np.argmax(np.abs(self._state))]} at index {np.argmax(np.abs(self._state))}'

    
    def _reset(self):
        '''
        Reset the Robosuite environment.
        '''

        self._episode_ended  = False
        self._success        = False
        self._reward         = 0
        self._episode_reward = 0
        self._j              = 0

        obs = self._env.reset()
        self.setState(obs)

        return tfa.trajectories.time_step.restart(self._state)

    
    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self._j += 1

        if self._success:
            action = np.zeros((12,))
            if self._eval:
                self._env.render()
        else:
            action = np.rint(action * self._max_steps_per_action)
            for n in range(int(max(abs(action)))):
                pure_action = np.clip(action, -1, 1) * Amax
                action[action<0] += 1
                action[action>0] -= 1
                obs, self._reward, done, info = self._env.step(pure_action)
                if self._reward > 0.93:
                    self._success = True
                    break
                self.setState(obs)
                if self._eval:
                    self._env.render()

        self._episode_reward += self._reward

        if self._j == self._params['max_episode_steps']:

            self._episode_ended   = True
            self._episode_counter += 1
            # if self._eval:
            #     self._env.close()
            
            pl(f'Episode {self._episode_counter} complete -- Episode reward: {self._episode_reward:.6f} -- Success: {self._success}')
            
            return tfa.trajectories.time_step.termination(
                observation = self._state,
                reward      = self._reward,
            )

        else:
            
            return tfa.trajectories.time_step.transition(
                observation = self._state,
                reward      = self._reward,
                discount    = self._discount
            )

    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def episode_counter(self):
        return self._episode_counter

    def episode_reward(self):
        return self._episode_reward