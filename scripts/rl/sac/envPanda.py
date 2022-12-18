import gym
import time
import robosuite
import pandas as pd
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from scripts.const import *
from scripts.utils import *
from scripts.rl.sac.primitives import *

class envPanda(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, evalEnv=False):
        super(envPanda, self).__init__()

        self.episode_rewards      = []
        self.num_elapsed_episodes = 1
        self.evalEnv              = evalEnv
        self.envType              = 'Evaluation' if evalEnv else 'Training'


        # Set of 3 parameterized action primitives per robot: move, rotate, align
        self.action_space = spaces.Box(
            low   = -ACTION_LIM,
            high  = +ACTION_LIM,
            shape = (3,),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low   = -1,
            high  = +1,
            shape = (7,3),
            dtype = np.float32
        )

        controller_config = robosuite.load_controller_config(default_controller="OSC_POSE")

        config = {
            "env_name"             : "TwoArmPegInHole",
            "robots"               : ["Panda", "Panda"],
            "controller_configs"   : controller_config,
            "env_configuration"    : "single-arm-opposed",
        }

        self.env = robosuite.make(
            **config,
            has_renderer           = True,
            has_offscreen_renderer = True,
            render_camera          = "agentview",
            ignore_done            = False,
            horizon                = ENV_HORIZON,
            use_camera_obs         = False,
            reward_shaping         = True,
            control_freq           = 20,
            hard_reset             = False,
            # initialization_noise   = {'type': 'gaussian', 'magnitude': 0.2}
            initialization_noise   = None,
        )


    def step(self, action):

        def step_env(action_flat):
            env_observation, env_reward, env_done, env_info = self.env.step(action_flat)
            step_observation = self.set_step_observation(env_observation)
            step_reward      = self.set_step_reward(env_reward, env_done)
            step_done        = self.set_step_done(env_done)
            step_info        = self.set_step_info(env_info)
            self.render()

            return step_observation, step_reward, step_done, step_info
        
        step_done = False
        primitive_reward = 0
        action = action.astype(int)
        if action.sum():
            for act, a in zip(action, range(len(action))):
                if act: 
                    sign = np.sign(act)
                    for i in range(act, 0, sign*-1):
                        action_flat = np.zeros(12)
                        if a==0:
                            action_flat[1] = sign * 3.75
                        if a==1:
                            action_flat[4] = sign * 0.15
                        if a==2:
                            pass
                        step_observation, step_reward, step_done, step_info = step_env(action_flat)
                        primitive_reward += step_reward
                        if step_done:
                            break
                if step_done:
                    break
        else:
            action_flat = np.zeros(12)
            step_observation, step_reward, step_done, step_info = step_env(action_flat)

        primitive_observation = step_observation
        primitive_done = step_done
        primitive_info = step_info

        primitive_observation = self.set_primitive_observation(primitive_observation, primitive_reward)
        primitive_reward      = self.set_primitive_reward(primitive_reward)
        primitive_done        = self.set_primitive_done(primitive_done)
        primitive_info        = self.set_primitive_info(primitive_info)

        # time.sleep(1)

        return primitive_observation, primitive_reward, primitive_done, primitive_info

    # ================================== START _step_ ================================== #

    def set_step_observation(self, env_observation):

        return env_observation

    
    def set_step_reward(self, env_reward, env_done):

        if env_done and self.env.timestep != ENV_HORIZON:
            pl(f'Environment solved! env_reward: {env_reward}\n\n\n')
            step_reward = 1

        else:
            # step_reward = env_reward - self.best_reward
            step_reward = env_reward - self.previous_reward
            
            self.previous_reward = env_reward
            if env_reward > self.best_reward:
                self.best_reward = env_reward

            self.prev_step_reward = step_reward
        
        # print(step_reward)
        
        return step_reward


    def set_step_done(self, env_done):
        
        return env_done


    def set_step_info(self, env_info):

        return env_info

    # ================================== END _step_ ================================== #
    
    # ================================== START _primitive_ ================================== #

    def set_primitive_observation(self, primitive_observation, primitive_reward):

        primitive_observation = np.concatenate((
            np.concatenate((
                primitive_observation['robot0_eef_pos'] / 3,
                primitive_observation['peg_quat']
            ))[:,np.newaxis],
            np.concatenate((
                primitive_observation['hole_pos'] / 3,
                primitive_observation['hole_quat'],
            ))[:,np.newaxis],
            np.concatenate((
                primitive_observation['peg_to_hole'] / 3,
                primitive_observation['t'].reshape(1,) / 3,
                primitive_observation['d'].reshape(1,) / 3,
                primitive_observation['angle'].reshape(1,),
                np.clip([primitive_reward], 0, 1).reshape(1,),
            ))[:,np.newaxis]
        ), axis=1, dtype=DTYPE)
        
        if np.max(np.abs(primitive_observation)) > 1.0:
            pl(f'\n\n\nNOTE: {primitive_observation[np.argmax(np.abs(primitive_observation))]} at index {np.argmax(np.abs(primitive_observation))}\n\n\n')
        
        return primitive_observation

    
    def set_primitive_reward(self, primitive_reward):
        
        self.episode_reward += primitive_reward

        # pl(f'timestep: {self.env.timestep} -- primitive_reward: {primitive_reward}\n')
        
        return primitive_reward


    def set_primitive_done(self, primitive_done):

        if primitive_done:
            pl(f'{self.envType} episode {self.num_elapsed_episodes}  --  episode reward: {self.episode_reward}')
            self.num_elapsed_episodes += 1
            self.episode_rewards.append(self.episode_reward)            
        
        return primitive_done


    def set_primitive_info(self, primitive_info):

        return primitive_info

    # ================================== END _primitive_ ================================== #
    
    def reset(self):
        reward = 1
        while reward > 0.93:
            observation                     = self.env.reset()
            reward                          = self.env.reward()
            randomizer      = np.zeros(12)
            randomizer[1]   = (np.random.randn()-0.25) * 5
            randomizer[4]   = (np.random.randn()-0.00) * 5 
            randomizer[7]   = (np.random.randn()+0.25) * 10
            randomizer = np.around(randomizer)
            while randomizer.sum():
                randomizer_unitary  = np.sign(randomizer)
                randomizer         -= randomizer_unitary
                randomizer_unitary *= Amax + Amax
                observation, reward, done, info = self.env.step(randomizer_unitary)
                self.render()
            
            observation                     = self.set_primitive_observation(observation, reward)
            self.previous_reward            = self.env.reward()
            self.best_reward                = self.env.reward()
            self.episode_reward             = 0

            # time.sleep(1)

        return observation

    
    def render(self):
        if self.evalEnv or self.num_elapsed_episodes % 50 in [0,1,2]:
            self.env.viewer.set_camera(camera_id=0)
            self.env.render()
    
    
    def close(self):
        self.env.close()


if __name__ == '__main__':
    env = envPanda()
    check_env(env)