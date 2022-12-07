import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from scripts.rl.stableBaselines3.envPanda import envPanda

from scripts.utils import *

## Logging
logging.getLogger().setLevel(logging.INFO)
logFile = f'train.log'
logging.basicConfig(filename=logFile, format='%(message)s')

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

pl('Check if continuous observation works with PPO')

envTrain = envPanda()
envEval  = envPanda(evalEnv=True)

ppo = PPO(
      policy              = 'MlpPolicy',
      env                 = envTrain,
      learning_rate       = 0.0003,
      n_steps             = 2048,
      batch_size          = 64,
      n_epochs            = 10,
      gamma               = 0.99,
      gae_lambda          = 0.95,
      clip_range          = 0.2,
      clip_range_vf       = None,
      normalize_advantage = True,
      ent_coef            = 0,
      vf_coef             = 0.5,
      max_grad_norm       = 0.5,
      use_sde             = False,
      sde_sample_freq     = -1,
      target_kl           = None,
      tensorboard_log     = None,
      create_eval_env     = False,
      policy_kwargs       = None,
      verbose             = 1,
      seed                = None,
      device              = 'cuda',
      _init_setup_model   = True
)

ppo.learn(
    total_timesteps     = 350000,
    callback            = None,
    log_interval        = 1,
    eval_env            = envEval,
    eval_freq           = 250 * 15,
    n_eval_episodes     = 3,
    tb_log_name         = 'PPO',
    eval_log_path       = 'logs',
    reset_num_timesteps = True,
    progress_bar        = True
)

ppo.save("ppo_panda")

del ppo

ppo = PPO.load("ppo_panda")
obs_panda         = envTrain.reset()
for i in range(5000):
    action_panda, _states_panda = ppo.predict(obs_panda)
    pl(action_panda)
    obs_panda, rewards_panda, dones_panda, info_panda = envTrain.step(action_panda)
    envTrain.render()