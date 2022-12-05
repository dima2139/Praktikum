from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from scripts.rl.stableBaselines3.envPanda import envPanda

# env_cartpole   = make_vec_env("CartPole-v1", n_envs=1)
# model_cartpole = PPO("MlpPolicy", env_cartpole, verbose=1)
# model_cartpole.learn(total_timesteps=5000)
# model_cartpole.save("ppo_cartpole")
# del model_cartpole
# model_cartpole = PPO.load("ppo_cartpole")
# obs_cartpole   = env_cartpole.reset()
# for i in range(500):
#     action_cartpole, _states_cartpole = model_cartpole.predict(obs_cartpole)
#     obs_cartpole, rewards_cartpole, dones_cartpole, info_cartpole = env_cartpole.step(action_cartpole)
#     env_cartpole.render()

print('Check if continuous observation works with PPO')

env_panda   = envPanda()
# model_panda = PPO("MlpPolicy", env_panda, verbose=1)
# model_panda.learn(total_timesteps=75000, progress_bar=True)
# model_panda.save("ppo_panda")
# del model_panda
model_panda = PPO.load("ppo_panda")
obs_panda         = env_panda.reset()
for i in range(5000):
    action_panda, _states_panda = model_panda.predict(obs_panda)
    obs_panda, rewards_panda, dones_panda, info_panda = env_panda.step(action_panda)
    env_panda.render()