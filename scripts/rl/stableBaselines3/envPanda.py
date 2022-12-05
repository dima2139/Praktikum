import gym
from gym import spaces
import robosuite
from stable_baselines3.common.env_checker import check_env
from scripts.const import *

class envPanda(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(envPanda, self).__init__()
        
        self.action_space = spaces.Discrete(6)

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
        action_onehot = np.zeros((12,))
        action_onehot[action] = 1
        action_onehot *= Amax
        # print(action_onehot)
        observation, reward, done, info = self.env.step(action_onehot)
        observation = self.set_observation(observation)
        reward = self.set_reward(reward)
        
        return observation, reward, done, info
    

    def set_reward(self, reward):
        return reward


    def set_done(self, done):
        if done:
            print('Episode is done:')
            print(done)
        return np.array([done]) 


    def set_info(self, info):
        return [info]

    
    def reset(self):
        observation = self.env.reset()
        observation = self.set_observation(observation)

        return observation

    
    def render(self, mode='human'):
        self.env.viewer.set_camera(camera_id=0)
        self.env.render()
    
    
    def close (self):
        self.env.close()


if __name__ == '__main__':
    env = envPanda()
    check_env(env)