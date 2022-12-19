import gym
import time
import robosuite
from gym import spaces
from pynput import keyboard
from stable_baselines3.common.env_checker import check_env

from scripts.const import *
from scripts.utils import *
from scripts.rl.sac.primitives import *

class envPanda(gym.Env):

    def __init__(self, evalEnv=False):
        super(envPanda, self).__init__()

        self.episode_rewards      = []
        self.num_elapsed_episodes = 1
        self.evalEnv              = evalEnv
        self.envType              = 'Evaluation' if evalEnv else 'Training'

        self.render_episodes = 0
        self.listener        = keyboard.Listener(
            on_press   = self.keypress,
            on_release = self.keyrelease
        )
        self.listener.start()


        self.action_space_map = {
            2: {'idx':0 , 'name':'peg-Z_m', 'delta':0},
            6: {'idx':1 , 'name':'peg-X_m', 'delta':-0.2},
            4: {'idx':2 , 'name':'peg-Y_m', 'delta':0},
            #: {'idx':3 , 'name':'peg-Z_r', 'delta':0},
            0: {'idx':4 , 'name':'peg-X_r', 'delta':0},
            #: {'idx':5 , 'name':'peg-Y_r', 'delta':0},
            
            3: {'idx':6 , 'name':'hole-Z_m', 'delta':0},
            7: {'idx':7 , 'name':'hole-X_m', 'delta':0.2},
            5: {'idx':8 , 'name':'hole-Y_m', 'delta':0},
            #: {'idx':9 , 'name':'hole-Z_r', 'delta':0},
            1: {'idx':10, 'name':'hole-X_r', 'delta':0},
            #: {'idx':11, 'name':'hole-Y_r', 'delta':0},

            8: {'idx':12, 'name':'sleep',    'delta':0},
        }


        # Set of 3 parameterized action primitives per robot: move, rotate, align
        self.action_space = spaces.Box(
            low   = -ACTION_LIM,
            high  = +ACTION_LIM,
            shape = (ACTION_DIM,),
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
            render_camera          = None,
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
            for a in range(ACTION_DIM):
                if action[a]:
                    sign = np.sign(action[a])
                    for i in range(action[a], 0, sign*-1):
                        action_flat = np.zeros(13, dtype=float)
                        action_flat[self.action_space_map[a]['idx']] = sign
                        action_flat *= Amax13
                        step_observation, step_reward, step_done, step_info = step_env(action_flat[:-1])
                        primitive_reward += step_reward
                        if step_done:
                            break
                if step_done:
                    break
        else:
            action_flat = np.zeros(12)
            step_observation, step_reward, step_done, step_info = step_env(action_flat)

        primitive_observation = step_observation
        primitive_done        = step_done
        primitive_info        = step_info

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
            render_episode = '(rendered) ' if self.render_episodes else ''
            pl(f'{render_episode}{self.envType} episode {self.num_elapsed_episodes}  --  episode reward: {self.episode_reward}')
            self.num_elapsed_episodes += 1
            self.episode_rewards.append(self.episode_reward)            
        
        return primitive_done


    def set_primitive_info(self, primitive_info):

        return primitive_info

    # ================================== END _primitive_ ================================== #
    
    def reset(self):
        self.render_episodes -= np.sign(self.render_episodes)
        t = 0
        while t < 0.2:
            observation = self.env.reset()
            reward      = self.env.reward()

            randomizer  = np.zeros(13)
            for k, v in self.action_space_map.items():
                randomizer[v['idx']] = (np.random.rand() - 0.5 + v['delta']) * 20
            
            randomizer = np.around(randomizer)
            while randomizer.sum():
                randomizer_unitary  = np.sign(randomizer)
                randomizer         -= randomizer_unitary
                randomizer_unitary *= Amax13
                observation, reward, done, info = self.env.step(randomizer_unitary[:-1])
                self.render()
            
            t = observation['t']

        observation                     = self.set_primitive_observation(observation, reward)
        self.previous_reward            = self.env.reward()
        self.best_reward                = self.env.reward()
        self.episode_reward             = 0

        self.render()

        return observation


    def keypress(self, key):
        self.render_episodes = 3
        # print(f'Pressed "{key.char}": rendering for 3 episodes')

    def keyrelease(self, key):
        pass

    def render(self):
        # if self.render_episode and (self.evalEnv or self.num_elapsed_episodes % 50 in [0,1,2]):
        if self.render_episodes:
            # self.env.viewer.set_camera(camera_id=0)
            self.env.render()
            
    
    
    def close(self):
        self.env.close()


if __name__ == '__main__':
    env = envPanda()
    check_env(env)