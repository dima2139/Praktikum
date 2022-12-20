
'''
Soft Actor-Critic.
'''


## Imports
import time
import shutil
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from scripts.utils import *
from scripts.rl.sac.envPanda import envPanda



## Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, help='Optional, model to resume training from')
args = parser.parse_args()


## Setup
if args.resume:
    savePath  = args.resume
    MODEL     = savePath.split('/')[-1]
else:
    MODEL = f'{int(time.time())}'
    savePath  = f'models/sac/{MODEL}'
    mkdirs(savePath)

addDir('scripts', f'{savePath}/scripts')


## Environment
numVecEnvs    = 4
gradientSteps = 2
envTrain      = envPanda()
envEval       = envPanda(evalEnv=True)
vecEnvTrain   = make_vec_env(envPanda, n_envs=numVecEnvs, seed=SEED)


## Logging
logging.getLogger().setLevel(logging.INFO)
logFile = f'{savePath}/train.log'
logging.basicConfig(filename=logFile, format='%(message)s')


## Model
if args.resume:
    pl('\n\n\nContinuing training...\n\n\n')
    artefacts = sorted_nicely(os.listdir(args.resume))
    for a in artefacts:
        if 'steps.zip' in a:
            saved_model = a
        elif 'replay_buffer' in a and 'pkl' in a:
            saved_replay_buffer = a
    model = SAC.load(f'{args.resume}/{saved_model}')
    model.load_replay_buffer(f'{args.resume}/{saved_replay_buffer}')
    print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")
    model.set_env(vecEnvTrain)
    # params = model.get_parameters()
    # for entity in params:
    #     if 'optimizer' in entity:
    #         params[entity]['param_groups'][0]['lr'] = 0.0001
    # model.set_parameters(params)
    # model.learning_rate = 0.0001

else:
    pl('Starting training...\n\n\n')
    model = SAC(
        policy                 = 'MlpPolicy',
        env                    = vecEnvTrain,
        learning_rate          = 0.0003,
        buffer_size            = 1000000,
        learning_starts        = 100,
        batch_size             = 256,
        tau                    = 0.005,
        gamma                  = 0.99,
        train_freq             = 1,
        gradient_steps         = gradientSteps,  # 1 or -1 or n for vec_env 
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
checkpoint_callback = CheckpointCallback(
    save_freq          = AGENT_HORIZON * 50,
    save_path          = savePath,
    name_prefix        = 'sac_resume',
    save_replay_buffer = True,
    save_vecnormalize  = True,
)
model.learn(
    total_timesteps     = 50000,
    callback            = checkpoint_callback,
    reset_num_timesteps = True,
    progress_bar        = True,

    log_interval        = 30,
    tb_log_name         = 'SAC',
    
    eval_freq           = AGENT_HORIZON * 30,
    eval_env            = envEval,
    n_eval_episodes     = 3,
)
model.save(f'{savePath}/sac_final')