import time
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env  # !!!!!!!!!!!!!!!!!!
from scripts.rl.stableBaselines3.envPanda import envPanda
from stable_baselines3.common.callbacks import CheckpointCallback

from scripts.utils import *

TRAIN     = True
savePath  = f'models/ppo/rl_model_{int(time.time())}'
mkdirs(savePath)

envTrain = envPanda()
envEval  = envPanda(evalEnv=True)

if TRAIN:
    ## Logging
    logging.getLogger().setLevel(logging.INFO)
    logFile = f'{savePath}/train.log'
    logging.basicConfig(filename=logFile, format='%(message)s')

    pl('Check if continuous observation works with PPO')

    checkpoint_callback = CheckpointCallback(
        save_freq          = 5000,
        save_path          = f'{savePath}/',
        name_prefix        = 'ppo_',
        save_replay_buffer = True,
        save_vecnormalize  = True,
    )

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
        total_timesteps     = 100000,
        callback            = checkpoint_callback,
        log_interval        = 1,
        eval_env            = envEval,
        eval_freq           = 250 * 15,
        n_eval_episodes     = 3,
        tb_log_name         = 'PPO',
        eval_log_path       = 'logs',
        reset_num_timesteps = True,
        progress_bar        = True,
    )

    ppo.save(f'{savePath}/final')

    del ppo

ppo       = PPO.load(f'{savePath}/final')
obs_panda = envEval.reset()
for i in range(250):
    action_panda, _states_panda = ppo.predict(obs_panda)
    obs_panda, rewards_panda, dones_panda, info_panda = envEval.step(action_panda)