# Imports
import gym
import time
import datetime
import robosuite
from gym import spaces
from pynput import keyboard
from stable_baselines3.common.env_checker import check_env

from scripts.const import *
from scripts.utils import *


## Panda Environment Class
class envRapPanda(gym.Env):

    def __init__(self, savePath=None, evalEnv=False, primitive=None):
        super(envRapPanda, self).__init__()

        self.render_episodes = 0
        if evalEnv:
            envNumber = 1
        else:
            envNumberPath = f'{savePath}/render_env'
            if os.path.exists(envNumberPath):
                with open(envNumberPath, 'r') as f:
                    envNumber = int(f.read())
                    envNumber += 1
            else:
                envNumber = 1
            with open(envNumberPath, 'w') as f:
                f.write(str(envNumber))
        if envNumber==1:
            self.keypresses      = ''
            self.listener        = keyboard.Listener(
                on_press   = self.keypress,
                on_release = self.keyrelease
            )
            self.listener.start()

        self.timeStart            = time.time()
        self.evalEnv              = evalEnv
        self.envType              = 'Evaluation' if evalEnv else 'Training'
        self.num_elapsed_episodes = envNumber
        self.primitive            = primitive

        self.action_space_map = {
            6:  {'idx':0 , 'name':'peg-Z_m', 'delta':0},
            10: {'idx':1 , 'name':'peg-X_m', 'delta':-0.25},
            8:  {'idx':2 , 'name':'peg-Y_m', 'delta':0},
            2:  {'idx':3 , 'name':'peg-Z_r', 'delta':0},
            0:  {'idx':4 , 'name':'peg-X_r', 'delta':0},
            4:  {'idx':5 , 'name':'peg-Y_r', 'delta':0},
            
            7:  {'idx':6 , 'name':'hole-Z_m', 'delta':0},
            11: {'idx':7 , 'name':'hole-X_m', 'delta':0.25},
            9:  {'idx':8 , 'name':'hole-Y_m', 'delta':0},
            3:  {'idx':9 , 'name':'hole-Z_r', 'delta':0},
            1:  {'idx':10, 'name':'hole-X_r', 'delta':0},
            5:  {'idx':11, 'name':'hole-Y_r', 'delta':0},

            12: {'idx':12, 'name':'sleep',    'delta':0},
        }

        self.action_space = spaces.Box(
            low   = -1,
            high  = +1,
            shape = (ACTION_DIM,),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low   = -1,
            high  = +1,
            shape = (73,),
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
            initialization_noise   = {'type': 'uniform', 'magnitude': 0.3}
        )


    def step(self, action):

        def step_env(action_flat):
            env_observation, env_reward, env_done, env_info = self.env.step(action_flat)
            step_observation = self.set_step_observation(env_observation)
            step_reward      = self.set_step_reward(env_reward, env_done, env_observation)
            step_done        = self.set_step_done(env_done)
            step_info        = self.set_step_info(env_info)
            self.render()

            return step_observation, step_reward, step_done, step_info

        action_reward  = 0
        step_done         = False
        action           *= ACTION_LIM
        action            = action.astype(int)
        if action.sum():
            while action.sum():
                sign = np.sign(action)
                action -= sign
                action_flat = sign * Amax12
                step_observation, step_reward, step_done, step_info = step_env(action_flat)
                action_reward += step_reward
                if step_done:
                    break

        else:
            action_flat = np.zeros(12)
            step_observation, step_reward, step_done, step_info = step_env(action_flat)

        action_observation = step_observation
        action_done        = step_done
        action_info        = step_info

        action_observation = self.set_action_observation(action_observation)
        action_reward      = self.set_action_reward(action_reward)
        action_done        = self.set_action_done(action_done)
        action_info        = self.set_action_info(action_info)

        return action_observation, action_reward, action_done, action_info

    # ================================== START _step_ ================================== #

    def set_step_observation(self, env_observation):

        return env_observation

    
    def set_step_reward(self, env_reward, env_done, env_observation, reset=False):

        if env_done and self.env.timestep != ENV_HORIZON:
            pl(f'Environment solved! env_reward: {env_reward}\n\n\n')
            step_reward = 1

        if self.primitive == "align":
            dist = np.linalg.norm(QUAT_ANGLES_PEG - env_observation["peg_quat"])
            dist += np.linalg.norm(QUAT_ANGLES_HOLE - env_observation["hole_quat"])
            dist /= 2
            reward = dist

        elif self.primitive == "angle":
            reward = env_observation[self.primitive]
        
        elif self.primitive == "d":
            reward = 1 - np.tanh(env_observation[self.primitive])

        elif self.primitive == "reach+t":
            hole_pos         = self.env.sim.data.body_xpos[self.env.hole_body_id]
            gripper_site_pos = self.env.sim.data.body_xpos[self.env.peg_body_id]
            dist             = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward  = 1 - np.tanh(1.0 * dist)
            reward           = (reaching_reward + (1 - np.tanh(np.abs(env_observation[self.primitive])))) / 2
        
        if reset:
            self.previous_reward = reward
            return None

        else:
            step_reward = reward - self.previous_reward
            self.previous_reward = reward
            return step_reward


    def set_step_done(self, env_done):
        
        return env_done


    def set_step_info(self, env_info):

        return env_info

    # ================================== END _step_ ================================== #
    
    # ================================== START _action_ ================================== #

    def set_action_observation(self, action_observation):

        action_observation = np.concatenate((
            action_observation['robot0_proprio-state'],
            action_observation['robot1_proprio-state'],
            action_observation['object-state']
        ), dtype=DTYPE)
        action_observation = np.divide(action_observation, Omax, dtype=DTYPE)

        
        if np.max(np.abs(action_observation)) > 1.0:
            pl(f'\n\n\nNOTE: {action_observation[np.argmax(np.abs(action_observation))]} at index {np.argmax(np.abs(action_observation))} and timestep {self.env.timestep}\n\n\n')
        
        return action_observation

    
    def set_action_reward(self, action_reward):
        
        self.episode_reward += action_reward
        
        return action_reward


    def set_action_done(self, action_done):

        if action_done:
            rendered  = '(rendered) ' if self.render_episodes else ''
            elapsed   = f'{self.envType} ep. {self.num_elapsed_episodes}'
            reward    = f' -- ep. reward: {self.episode_reward}'
            timenow   = f' -- time {datetime.datetime.now()}' if self.evalEnv else ''
            timedelta = f' -- timedelta {time.time() - self.timeStart}' if self.evalEnv else ''
            pl(f'{rendered}{elapsed}{reward}{timenow}{timedelta}')
            self.num_elapsed_episodes += 1 if self.evalEnv else NUM_VEC_ENVS 

        return action_done


    def set_action_info(self, action_info):

        return action_info

    # ================================== END _action_ ================================== #
    
    # ================================== START misc ================================== #
    
    def reset(self):
        self.render_episodes -= np.sign(self.render_episodes)
        observation           = self.env.reset()
        self.set_step_reward(self.env.reward(), False, observation, reset=True)  # used to set self.previous_reward
        observation         = self.set_action_observation(observation)
        self.episode_reward = 0

        return observation


    def keypress(self, key):
        try:
            if self.keypresses == 'rmujoco' and key.char.isnumeric():
                self.render_episodes = int(key.char)
                self.keypresses = ''
            elif key.char=='r':
                self.keypresses = 'r'
            elif self.keypresses:
                self.keypresses += key.char
        except:
            pass


    def keyrelease(self, key):
        pass


    def render(self):
        if self.render_episodes:
            self.env.render()
            
    
    def close(self):
        self.env.close()

    # ================================== END misc ================================== #

if __name__ == '__main__':
    env = envRapPanda()
    check_env(env)