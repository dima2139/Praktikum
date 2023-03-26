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
class envRapsPanda(gym.Env):

    def __init__(self, savePath=None, evalEnv=False):
        super(envRapsPanda, self).__init__()

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

        self.numSuccesses         = 0
        self.timeStart            = time.time()
        self.evalEnv              = evalEnv
        self.envType              = 'Evaluation' if evalEnv else 'Training'
        self.num_elapsed_episodes = envNumber

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


    def step(self, action, primitive):

        def step_env(action_flat):
            env_observation, env_reward, env_done, env_info = self.env.step(action_flat)
            step_observation = self.set_step_observation(env_observation)
            step_reward      = self.set_step_reward(env_observation, env_reward)
            step_done        = self.set_step_done(env_done)
            step_info        = self.set_step_info(env_info)
            self.render()

            return step_observation, step_reward, step_done, step_info

        self.primitive = primitive
        action_reward  = 0
        step_done      = False
        action        *= ACTION_LIM
        action         = action.astype(int)
        if action.sum():
            while action.sum():
                sign = np.sign(action)
                action -= sign
 
                if self.primitive == 'align':
                    action_flat = sign * Amax12
                    step_observation, step_reward, step_done, step_info = step_env(action_flat)
                    action_reward += step_reward
                    if step_done:
                        break
                
                else:
                    sign_12 = np.zeros(12)
                    sign_12[:3] = sign[:3]
                    sign_12[6:9] = sign[3:]
                    action_flat = sign_12 * Amax12
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

    
    def set_step_reward(self, env_observation, env_reward, reset=False):

        if self.primitive == "align":
            if np.all(MIN_BBOX_PEG <= env_observation['robot0_eef_pos']) and np.all(env_observation['robot0_eef_pos'] <= MAX_BBOX_PEG):
                angle_mag_peg  = np.linalg.norm(QUAT_ANGLES_PEG - env_observation["peg_quat"]) / len(QUAT_ANGLES_PEG)
            else:
                angle_mag_peg  = 0.5

            if np.all(MIN_BBOX_HOLE <= env_observation['robot1_eef_pos']) and np.all(env_observation['robot1_eef_pos'] <= MAX_BBOX_HOLE):
                angle_mag_hole = np.linalg.norm(QUAT_ANGLES_HOLE - env_observation["hole_quat"]) / len(QUAT_ANGLES_HOLE)
            else:
                angle_mag_hole = 0.5
            
            reward    = 1 - (angle_mag_peg + angle_mag_hole)

            if self.i == PRIMITIVE_ALIGN_HORIZON:
                self.primitive_align_done = True
                self.i = 0
            else:
                self.i += 1

        elif self.primitive=='d':
            if np.all(MIN_BBOX_PEG <= env_observation['robot0_eef_pos']) and np.all(env_observation['robot0_eef_pos'] <= MAX_BBOX_PEG) \
            and np.all(MIN_BBOX_HOLE <= env_observation['robot1_eef_pos']) and np.all(env_observation['robot1_eef_pos'] <= MAX_BBOX_HOLE):
                reward = 1 - np.tanh(env_observation['d'])
            else:
                reward = 0

            if self.i == PRIMITIVE_D_HORIZON:
                self.primitive_d_done = True
                self.i = 0
            else:
                self.i += 1

        elif self.primitive == "t":
            reward = env_reward
            # print(reward)
            # hole_pos         = self.env.sim.data.body_xpos[self.env.hole_body_id]
            # gripper_site_pos = self.env.sim.data.body_xpos[self.env.peg_body_id]
            # dist             = np.linalg.norm(gripper_site_pos - hole_pos)
            # reaching_reward  = 1 - np.tanh(1.0 * dist)
            # reward           = (reaching_reward + (1 - np.tanh(np.abs(env_observation[self.primitive])))) / 2

            if self.i == PRIMITIVE_T_HORIZON:
                self.primitive_t_done = True
                self.i = 0
            else:
                self.i += 1
        
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
            if self.env.reward() > 0.9:
                self.numSuccesses += 1 

            rendered           = '(rendered) ' if self.render_episodes else ''
            elapsed            = f'{self.envType} ep. {self.num_elapsed_episodes}'
            episode_reward     = f' -- ep. reward: {self.episode_reward:.4f}'
            environment_reward = f' -- env. reward: {self.env.reward():.4f}'
            success_rate       = f' -- success rate: {self.numSuccesses / self.num_elapsed_episodes:.2f}' if self.evalEnv else ''
            timenow            = f' -- time {datetime.datetime.now()}' if self.evalEnv else ''
            timedelta          = f' -- total timedelta {time.time() - self.timeStart:.2f}' if self.evalEnv else ''
            avg_timedelta      = f' -- avg timedelta {(time.time() - self.timeStart) / self.num_elapsed_episodes:.2f}' if self.evalEnv else ''
    
            pl(f'{rendered}{elapsed}{episode_reward}{environment_reward}{success_rate}{timenow}{timedelta}{avg_timedelta}')
            
            self.num_elapsed_episodes += 1 if self.evalEnv else NUM_VEC_ENVS 

        return action_done


    def set_action_info(self, action_info):

        return action_info

    # ================================== END _action_ ================================== #
    
    # ================================== START misc ================================== #
    
    def reset(self):
        self.primitive_align_done  = False
        self.primitive_d_done      = False
        self.primitive_t_done      = False
        self.i                     = 0
        self.primitive             = 'align'
        self.render_episodes      -= np.sign(self.render_episodes)
        observation                = self.env.reset()
        reward                     = self.env.reward()
        self.set_step_reward(observation, reward, reset=True)  # used to set self.previous_reward
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
        # if self.render_episodes:
            self.env.render()
            
    
    def close(self):
        self.env.close()

    # ================================== END misc ================================== #

if __name__ == '__main__':
    env = envRapsPanda()
    check_env(env)