'''
Instantiate or Load a Proximal Policy Optimization
Agent for training or inference.
'''

print('Check if continuous observation works with PPO')


## Imports
import time
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env  # !!!!!!!!!!!!!!!!!!
from scripts.rl.stableBaselines3.envPanda import envPanda
from stable_baselines3.common.callbacks import CheckpointCallback

from scripts.utils import *


## Setup
RESUME = False
if not RESUME:
    MODEL = f'rl_model_{int(time.time())}'
else:
    MODEL = 'rl_model_1671028126'
    CHECKPOINT = 100000
savePath  = f'models/ppo/{MODEL}'
mkdirs(savePath)


## Environments
envTrain = envPanda()
envEval  = envPanda(evalEnv=True)


## Logging
logging.getLogger().setLevel(logging.INFO)
logFile = f'{savePath}/train.log'
logging.basicConfig(filename=logFile, format='%(message)s')


## Training
if not RESUME:
    pl('Starting training...\n\n\n')
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

else:
    pl('Continuing training...\n\n\n')
    ppo = PPO.load(f'{savePath}/{CHECKPOINT}_steps')
    ppo.set_env(envTrain)

checkpoint_callback = CheckpointCallback(
    save_freq          = 5000,
    save_path          = savePath,
    name_prefix        = 'ppo',
    save_replay_buffer = True,
    save_vecnormalize  = True,
)
ppo.learn(
    total_timesteps     = 500000,
    callback            = checkpoint_callback,
    log_interval        = 1,
    eval_env            = envEval,
    eval_freq           = 250 * 15,
    n_eval_episodes     = 3,
    tb_log_name         = 'PPO',
    eval_log_path       = 'logs',
    reset_num_timesteps = not RESUME,
    progress_bar        = True,
)

ppo.save(f'{savePath}/ppo_')