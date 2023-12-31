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
class envPanda(gym.Env):

    def __init__(self, savePath=None, evalEnv=False):
        super(envPanda, self).__init__()

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
            shape = (73,),  # (7,3),
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
            initialization_noise   = {'type': 'uniform', 'magnitude': 0.8}
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

        action_reward  = 0
        step_done         = False
        action           *= ACTION_LIM
        action            = action.astype(int)
        if action.sum():
            if STEP_MODE=='action':
                for a in range(ACTION_DIM):
                    if action[a]:
                        sign = np.sign(action[a])
                        for i in range(action[a], 0, sign*-1):
                            action_flat = np.zeros(13, dtype=float)
                            action_flat[self.action_space_map[a]['idx']] = sign
                            action_flat *= Amax13
                            step_observation, step_reward, step_done, step_info = step_env(action_flat[:-1])
                            action_reward += step_reward
                            if step_done:
                                break
                    if step_done:
                        break
            
            else:
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

        action_observation = self.set_action_observation(action_observation, action_reward)
        action_reward      = self.set_action_reward(action_reward)
        action_done        = self.set_action_done(action_done)
        action_info        = self.set_action_info(action_info)

        return action_observation, action_reward, action_done, action_info

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
            # if env_reward > self.best_reward:
            #     self.best_reward = env_reward

            self.prev_step_reward = step_reward
        
        # print(step_reward)
        
        return step_reward


    def set_step_done(self, env_done):
        
        return env_done


    def set_step_info(self, env_info):

        return env_info

    # ================================== END _step_ ================================== #

    # ================================== START _action_ ================================== #

    def set_action_observation(self, action_observation, action_reward):

        # action_observation = np.concatenate((
        #     np.concatenate((
        #         action_observation['robot0_eef_pos'] / 3,
        #         action_observation['peg_quat']
        #     ))[:,np.newaxis],
        #     np.concatenate((
        #         action_observation['hole_pos'] / 3,
        #         action_observation['hole_quat'],
        #     ))[:,np.newaxis],
        #     np.concatenate((
        #         action_observation['peg_to_hole'] / 3,
        #         action_observation['t'].reshape(1,) / 3,
        #         action_observation['d'].reshape(1,) / 3,
        #         action_observation['angle'].reshape(1,),
        #         np.clip([action_reward], 0, 1).reshape(1,),
        #     ))[:,np.newaxis]
        # ), axis=1, dtype=DTYPE)


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

        # pl(f'timestep: {self.env.timestep} -- action_reward: {action_reward}\n')
        
        return action_reward


    def set_action_done(self, action_done):

        if action_done:
            rendered           = '(rendered) ' if self.render_episodes else ''
            elapsed            = f'{self.envType} ep. {self.num_elapsed_episodes}'
            episode_reward     = f' -- ep. reward: {self.episode_reward}'
            environment_reward = f' -- env. reward: {self.env.reward()}'
            timenow            = f' -- time {datetime.datetime.now()}' if self.evalEnv else ''
            timedelta          = f' -- timedelta {time.time() - self.timeStart}' if self.evalEnv else ''
            pl(f'{rendered}{elapsed}{episode_reward}{environment_reward}{timenow}{timedelta}')
            self.num_elapsed_episodes += 1 if self.evalEnv else NUM_VEC_ENVS 

        return action_done


    def set_action_info(self, action_info):

        return action_info

    # ================================== END _action_ ================================== #
    
    # ================================== START misc ================================== #
    
    def reset(self):
        
        if RESET_MODE=='check_velocity':
            flag = False
            observation = self.env.reset()
            time_start = time.time()
            self.render()
            while not flag:
                action_flat = np.zeros(ACTION_DIM, dtype=float)
                observation, reward, done, info = self.env.step(action_flat)
                self.render()
                movement = np.sum(np.abs(observation['robot0_joint_vel'])) + np.sum(np.abs(observation['robot1_joint_vel']))
                if movement < MOVEMENT_EPSILON:
                    pl(f'Movement: {movement}, time taken: {time.time() - time_start}')
                    flag = True
                    time.sleep(1)
                else:
                    delta_t = time.time() - time_start
                    if delta_t > 0.5:
                        pl(f'Movement: {movement}, timeout at 1 second, resetting environment')
                        observation = self.env.reset()
                        time_start = time.time()

        elif RESET_MODE=='fixed_dimensions':
            t = 0
            while t < 0.2:
                observation = self.env.reset()

                randomizer  = np.zeros(13)
                for k, v in self.action_space_map.items():
                    randomizer[v['idx']] = (np.random.rand() - 0.5 + v['delta']) * ACTION_LIM
                
                randomizer = np.around(randomizer)
                while randomizer.sum():
                    randomizer_unitary  = np.sign(randomizer)
                    randomizer         -= randomizer_unitary
                    randomizer_unitary *= Amax13Init
                    observation, reward, done, info = self.env.step(randomizer_unitary[:-1])
                    # self.render()
                
                t = observation['t']

        elif RESET_MODE=='limit_velocity':
            velocityFlag = True
            velocityLim  = 0.1
            while velocityFlag:
                observation  = self.env.reset()
                velocityFlag = np.linalg.norm(observation['robot0_joint_vel']) > velocityLim or np.linalg.norm(observation['robot1_joint_vel']) > velocityLim
                if velocityFlag:
                    pl(f'Joint velocity exceeded, robot0: {observation["robot0_joint_vel"]}, robot1: {observation["robot1_joint_vel"]}')
                self.render()
        
        else:
            observation = self.env.reset()

        self.render_episodes -= np.sign(self.render_episodes)
        observation          = self.set_action_observation(observation, self.env.reward())
        self.previous_reward = self.env.reward()
        # self.best_reward = self.env.reward()
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
        # if self.render_episodes:
            self.env.render()
            
    
    def close(self):
        self.env.close()

    # ================================== END misc ================================== #

if __name__ == '__main__':
    env = envPanda()
    check_env(env)