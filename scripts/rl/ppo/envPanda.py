import gym
import robosuite
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from scripts.rl.sac.primitives import *
from scripts.const import *
from scripts.utils import *

class envPanda(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, evalEnv=False):
        super(envPanda, self).__init__()

        self.num_elapsed_episodes = 1
        self.max_episode_len      = 10
        self.evalEnv              = evalEnv
        self.envType              = 'Evaluation' if evalEnv else 'Training'

        self.action_space = spaces.MultiDiscrete(
            nvec = [2,2,6,30]  # primitive, robot, axis, amount
        ) 

        self.observation_space = spaces.Box(
            low   = -1,
            high  = +1,
            shape = (20,),
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
            render_visual_mesh     = True,
            render_collision_mesh  = True,
            ignore_done            = False,
            horizon                = ,
            use_camera_obs         = False,
            reward_shaping         = True,
            control_freq           = 20,
            hard_reset             = False,
            initialization_noise   = {'type': 'gaussian', 'magnitude': 0.25}
        )


    def set_observation(self, observation):
        observation = np.concatenate((
            # observation['robot0_proprio-state'],
            # observation['robot1_proprio-state'],
            # observation['object-state']
            observation['robot0_eef_pos'] / 2,
            observation['peg_quat'],
            observation['hole_pos'] / 2,
            observation['hole_quat'],
            observation['peg_to_hole'] / 2,
            observation['angle'].reshape(1,),
            observation['t'].reshape(1,) / 2,
            observation['d'].reshape(1,) / 2,   
        ), dtype=DTYPE)
        # observation = np.divide(observation, Omax, dtype=DTYPE)
        if np.max(np.abs(observation)) > 1.0:
            pl(f'\n\n\nNOTE: {observation[np.argmax(np.abs(observation))]} at index {np.argmax(np.abs(observation))}\n\n\n')
        return observation

    
    def step(self, action):
        '''
        Apply the agent's action on the environment.
        '''
        

        [primitive, robot, axis, amount] = action
        # print(action)

        # observation, absolute_reward, done, info = rotate_or_move(self.env, primitive, robot, axis, amount)
        
        # if primitive != 2:
        sign = axis%2*2-1
        axis = axis//2
        if amount:
            for i in range(amount):
                action = np.zeros(12)
                action[axis + primitive*3 + robot*6] = sign
                action *= np.array(Amax+Amax)
                
                observation, absolute_reward, done, info = self.env.step(action)
                # print(action)
                if self.evalEnv or self.i%1==0:
                    self.render()
        else:
                observation, absolute_reward, done, info = self.env.step(np.zeros(12))
                # print(action)
                # self.render()
        # else:



        observation = self.set_observation(observation)
        reward      = self.set_reward(absolute_reward)
        done        = self.set_done(observation, reward, done, info)
        info        = self.set_info(info)

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
    
        if absolute_reward > 0.93:
            reward = absolute_reward * 2
        else:
            reward = absolute_reward
        
        # print(reward)

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