import gym
import robosuite
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from scripts.const import *
from scripts.utils import *
from scripts.rl.sac.primitives import *

class envPanda(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, evalEnv=False):
        super(envPanda, self).__init__()

        self.num_elapsed_episodes = 1
        self.max_episode_len      = 100
        self.evalEnv              = evalEnv
        self.envType              = 'Evaluation' if evalEnv else 'Training'


        # Set of 3 parameterized action primitives per robot: move, rotate, align
        self.action_space = spaces.Box(
            low   = -10,
            high  = +10,
            shape = (6,2),
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
            has_offscreen_renderer = False,
            render_camera          = "agentview",
            ignore_done            = True,
            horizon                = 100,
            use_camera_obs         = False,
            reward_shaping         = True,
            control_freq           = 20,
            hard_reset             = False,
            initialization_noise   = {'type': 'gaussian', 'magnitude': 0.5}
        )


    def set_observation(self, observation):
        observation = np.concatenate((
            observation['robot0_proprio-state'],
            observation['robot1_proprio-state'],
            observation['object-state']
        ), dtype=DTYPE)
        observation = np.divide(observation, Omax, dtype=DTYPE)
        # assert np.max(np.abs(observation)) <= 1.0, f'{observation[np.argmax(np.abs(observation))]} at index {np.argmax(np.abs(observation))}'
        return observation

    
    def step(self, action):
        arr_Amax = np.array(Amax).reshape(6,1)
        action *= arr_Amax
        action_flat = action.ravel()
        observation, absolute_reward, done, info = self.env.step(action_flat)
        observation = self.set_observation(observation)
        reward      = self.set_reward(absolute_reward)
        done        = self.set_done(observation, reward, done, info)
        info        = self.set_info(info)
        if self.evalEnv or self.num_elapsed_episodes%1==0:
            self.render()

        return observation, reward, done, info
    

    def set_reward(self, absolute_reward):
        # if self.best_reward is False:
        #     reward = 0
        #     self.best_reward = absolute_reward
        
        # if absolute_reward > 0.93:
        #     reward = (absolute_reward - self.best_reward) * 5
        # else:
        #     reward = absolute_reward - self.best_reward
        
        # if absolute_reward > self.best_reward:
        #     self.best_reward = absolute_reward
    
        # # print(reward)

        reward = absolute_reward

        return reward


    def set_done(self, observation, reward, done, info):

        if done:
            pl('\n\n\nDONE!')
            pl('observation')
            pl(observation)
            pl('reward')
            pl(reward)
            pl('done')
            pl(done)
            pl('info')
            pl(info)
            pl('\n\n\n')
        
        if self.i == self.max_episode_len:
            pl(f'{self.envType} episode {self.num_elapsed_episodes}  --  episode reward: {self.episode_reward}')
            done   = True
            self.num_elapsed_episodes += 1
        else:
            self.i += 1
            self.episode_reward += reward
        
        return done


    def set_info(self, info):
        return info

    
    def reset(self):
        observation         = self.env.reset()
        observation         = self.set_observation(observation)
        self.episode_reward = 0
        self.best_reward    = False
        self.i              = 1

        return observation

    
    def render(self, mode='human'):
        self.env.viewer.set_camera(camera_id=0)
        self.env.render()
    
    
    def close(self):
        self.env.close()


if __name__ == '__main__':
    env = envPanda()
    check_env(env)