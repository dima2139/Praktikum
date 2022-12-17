
'''
Soft Actor-Critic.
'''


## Imports
import time
from stable_baselines3 import SAC
from scripts.utils import *
from scripts.rl.sac.envPanda import envPanda
from stable_baselines3.common.callbacks import CheckpointCallback


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
    learning_rate          = 0.005,
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
    tensorboard_log        = None,
    policy_kwargs          = None,
    verbose                = 1,
    seed                   = None,
    device                 = 'auto',
    _init_setup_model      = True
)


## Training
pl('Starting training...\n\n\n')
checkpoint_callback = CheckpointCallback(
    save_freq          = 5000,
    save_path          = savePath,
    name_prefix        = '',
    save_replay_buffer = True,
    save_vecnormalize  = True,
)
model.learn(
    total_timesteps     = 1000000,
    callback            = checkpoint_callback,
    log_interval        = 10,
    eval_env            = envEval,
    eval_freq           = 100 * 20,
    n_eval_episodes     = 4,
    tb_log_name         = 'SAC',
    reset_num_timesteps = True,
    progress_bar        = True
)
model.save(f'{savePath}/final')