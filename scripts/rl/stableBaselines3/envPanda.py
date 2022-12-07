import gym
from gym import spaces
import robosuite
from stable_baselines3.common.env_checker import check_env
from scripts.const import *

class envPanda(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(envPanda, self).__init__()

        self.max_episode_len = 100
        self.episode_reward  = 0

        # Set of 7 actions per robot: 6 actions for 6 joints, 1 action to do nothing
        self.action_space = spaces.MultiDiscrete(nvec = [7,7]) 

        self.observation_space = spaces.Box(
            low   = -1,
            high  = +1,
            shape = (73,),
            dtype = np.float32
        )

        controller_config = robosuite.load_controller_config(default_controller="OSC_POSE")

        config = {
            "env_name"          : "TwoArmPegInHole",
            "robots"            : ["Panda", "Panda"],
            "controller_configs": controller_config,
            "env_configuration" : "single-arm-opposed",
        }

        self.env = robosuite.make(
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


    def set_observation(self, observation):
        observation = np.concatenate((
            observation['robot0_proprio-state'],
            observation['robot1_proprio-state'],
            observation['object-state']
        ), dtype=DTYPE)
        observation = np.divide(observation, Omax, dtype=DTYPE)
        assert np.max(np.abs(observation)) <= 1.0, f'{observation[np.argmax(np.abs(observation))]} at index {np.argmax(np.abs(observation))}'
        return observation

    
    def step(self, action):
        action_peg = np.zeros((7,))
        action_peg[action[0]] = 1
        action_peg = action_peg[:6] * Amax

        action_hole = np.zeros((7,))
        action_hole[action[1]] = 1
        action_hole = action_hole[:6] * Amax

        action_combined = np.concatenate((action_peg, action_hole))
        # print(action_combined)

        if self.success:
            action_combined = np.zeros(12)

        observation, reward, done, info = self.env.step(action_combined)
        observation = self.set_observation(observation)
        reward = self.set_reward(reward)
        

        reward = set_reward(reward)
        done   = set_done(observation, reward, done, info)
        info   = set_info(info)
        render()

        return observation, reward, done, info
    

    def set_reward(self, reward):
        if reward > 0.93:
            self.success = True

        return reward


    def set_done(self, observation, reward, done, info):

        if done:
            print('\n\n\nDONE!')
            print('observation')
            print(observation)
            print('reward')
            print(reward)
            print('done')
            print(done)
            print('info')
            print(info)
            print('\n\n\n')
        
        if self.i == self.max_episode_len:
            print('Episode is done:')
            done   = True
        else:
            self.i += 1
            self.episode_reward += reward
        
        return np.array([done]) 


    def set_info(self, info):
        return [info]

    
    def reset(self):
        observation = self.env.reset()
        observation = self.set_observation(observation)
        print(self.episode_reward)
        self.episode_reward = 0
        self.success        = False
        self.i              = 1


        return observation

    
    def render(self, mode='human'):
        self.env.viewer.set_camera(camera_id=0)
        self.env.render()
    
    
    def close (self):
        self.env.close()


if __name__ == '__main__':
    env = envPanda()
    check_env(env)