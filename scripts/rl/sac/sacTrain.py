
'''
Soft Actor-Critic.
'''


## Imports
import time
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from scripts.utils import *
from scripts.rl.sac.envPanda import envPanda


## Setup
MODEL = f'{int(time.time())}'
savePath  = f'models/sac/{MODEL}'
mkdirs(savePath)


## Environment
envTrain = envPanda()
envEval  = envPanda(evalEnv=True)


## Logging
logging.getLogger().setLevel(logging.INFO)
logFile = f'{savePath}/train.log'
logging.basicConfig(filename=logFile, format='%(message)s')


## Model
model = SAC(
    policy                 = 'MlpPolicy',
    env                    = envTrain,
    learning_rate          = 0.003,
    buffer_size            = 1000000,
    learning_starts        = 100,
    batch_size             = 256,
    tau                    = 0.005,
    gamma                  = 0.99,
    train_freq             = 1,
    gradient_steps         = 1,
    action_noise           = None,
    replay_buffer_class    = None,
    replay_buffer_kwargs   = None,
    optimize_memory_usage  = False,
    ent_coef               = 'auto',
    target_update_interval = 1,
    target_entropy         = 'auto',
    use_sde                = False,
    sde_sample_freq        = -1,
    use_sde_at_warmup      = False,
    policy_kwargs          = None,
    verbose                = 1,
    tensorboard_log        = f'{savePath}/tensorboard',
    seed                   = None,
    device                 = 'auto',
    _init_setup_model      = True
)


## Training
pl('Starting training...\n\n\n')
checkpoint_callback = CheckpointCallback(
    save_freq          = AGENT_HORIZON * 100,
    save_path          = savePath,
    name_prefix        = 'sac',
    save_replay_buffer = True,
    save_vecnormalize  = True,
)
model.learn(
    total_timesteps     = 250000,
    callback            = checkpoint_callback,
    reset_num_timesteps = True,
    progress_bar        = True,

    log_interval        = 10,
    tb_log_name         = 'SAC',
    
    eval_freq           = AGENT_HORIZON * 50,
    eval_env            = envEval,
    n_eval_episodes     = 3,
)
model.save(f'{savePath}/sac_final')
wpkl(f'{savePath}/eval_episode_rewards.pkl', envEval.episode_rewards)
wpkl(f'{savePath}/train_episode_rewards.pkl', envTrain.episode_rewards)