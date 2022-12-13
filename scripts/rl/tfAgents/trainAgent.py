'''
Interface to train the Reinforcement Learning Soft Actor-Critic
on the supplied environment using command-line arguments.
Author: Daniel von Eschwege
Date:   24 November 2022
'''


## Imports
import os
import glob
import time
import shutil
import logging
import argparse
import tensorflow as tf

from scripts.const import *
from scripts.utils import *
from scripts.rl.envPanda import *
from scripts.rl.softActorCritic import softActorCritic


## Setup
tf.random.set_seed(SEED_TF)


## Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--i',     type=int,            help='Agent maximum training timesteps')
parser.add_argument('--j',     type=int,            help='Agent maximum episode timesteps')
parser.add_argument('--rbc',   type=int,            help='Replay buffer capacity')

parser.add_argument('--ics',   type=float,          help='Agent initial_collect_steps')
parser.add_argument('--psi',   type=float,          help='Agent policy_save_interval i.t.o. episodes')
parser.add_argument('--ei',    type=float,          help='Agent eval_interval i.t.o. episodes')
parser.add_argument('--nee',   type=int,            help='Agent num_eval_episodes')

parser.add_argument('--n',     type=int,            help='batch_size & layer_params')
parser.add_argument('--gamma', type=float,          help='Reward discount factor gamma')
parser.add_argument('--rsf',   type=float,          help='Reward scale factor')

parser.add_argument('--ilr',   type=float,          help='Initial learning rate')
parser.add_argument('--dcr',   type=float,          help='Learning rate decay rate')
parser.add_argument('--dcs',   type=int,            help='Learning rate decay steps interval')

parser.add_argument('--temp',  action='store_true', help='Perform a temporary run which can be discarded')
parser.add_argument('--dsoff', action='store_true', help='Turn off distributed strategy parallel processing')
parser.add_argument('--resume',type=str,            help='Resume a previous training run')
parser.add_argument('--text',  type=str,            help='Some text')

args = parser.parse_args()
argsDict = vars(args)


## File management
if args.resume:
    exp     = args.resume
    expPath = f'{modelsDir}/{exp}'
else:
    if args.temp:
        exp = 'temp'
    else:
        exp = 'sac_'
        for k, v in argsDict.items():
            exp += f'{k}{v}_'

    expPath = f'{modelsDir}/{exp}'
    if os.path.isdir(expPath) and not args.temp:
        # raise Exception(f'"{exp}" already exists')
        response = input(f'"{exp}" already exists. Overwrite? [y]/n\n').lower()
        if not response in ['y', '']:
            exit()

    try:
        shutil.rmtree(expPath)
    except:
        pass

modelPath = f'{expPath}/model'
mkdirs(modelPath)

plotPath = f'{expPath}/plot'
mkdirs(plotPath)

varsPath = f'{expPath}/vars'
mkdirs(varsPath)

if args.resume:
    prevArgsDict = ljson(glob.glob(f'{varsPath}/args*')[0])
    for k in argsDict:
        if k in ['n', 'gamma', 'rsf', 'nee']:
            assert argsDict[k] == prevArgsDict[k]
wjson(f'{varsPath}/args--{int(time.time())}.json', argsDict)

try:
    shutil.copytree('scripts/', f'{expPath}/scripts/')
except:
    pass # make that this reuses the old scripts upon resume

## Logging
logging.getLogger().setLevel(logging.INFO)
logFile = f'{expPath}/log--train--{int(time.time())}.log'
logging.basicConfig(filename=logFile, format='%(message)s')
if args.resume:
    pl('\n\n\n###Continuing training...###')
else:
    pl('\n\n\n###Soft Actor-Critic with the Actor-Learner API for Franka Emika Panda###')
pl('\n\n\n#---------------Setup---------------#')


## Panda Environment Hyperparameters
envParams = {
    'max_episode_steps': args.j,
    'gamma'            : args.gamma,
}
wjson(f'{varsPath}/envParams--{int(time.time())}.json', envParams)


## Soft Actor-Critic Hyperparameters
sacParams = {
    'max_train_steps'                : args.i,
    'replay_buffer_capacity_steps'   : args.rbc,
    'initial_collect_steps'          : args.ics,
    'policy_save_interval_steps'     : args.psi,
    'eval_interval'                  : args.ei,
    'num_eval_episodes'              : args.nee,

    'batch_size'                     : args.n,
    'actor_fc_layer_params'          : (args.n, args.n),
    'critic_joint_fc_layer_params'   : (args.n, args.n),
    
    'gamma'                          : args.gamma, # why is there gamma for sacParams & gamma for envParams?
    'reward_scale_factor'            : args.rsf,
    'target_update_tau'              : 0.005,
    'target_update_period'           : 1,
    
    'kernel_initializer'             : 'glorot_uniform',

    'initial_learning_rate'          : args.ilr,
    'decay_rate'                     : args.dcr,
    'decay_steps'                    : args.dcs,
    
    'observer_sequence_length'       : 2,
    'observer_stride_length'         : 1,
    
    'collect_actor_steps_per_run'    : 1,
    'collect_actor_buffer_size'      : 10,

    'replay_sequence_length'         : 2,
    'dataset_num_steps'              : 2,
    'dataset_buffer_size'            : 50,

    'log_trigger_interval'           : 1000,

    'max_to_keep'                    : 5000
}
wjson(f'{varsPath}/sacParams--{int(time.time())}.json', sacParams)


## Environment
envCollect = envPanda(
    envName = 'envCollect',
    params  = envParams,
    eval    = False
)
envEval = envPanda(  
    envName = 'envEval',
    params  = envParams,
    eval    = True
)


## Agent
distributed = True
if args.dsoff:
    distributed = False
    pl('Note: DISTRIBUTED STRATEGY is currently TURNED OFF.\n')
sacAgent = softActorCritic(
    envCollect  = envCollect,
    envEval     = envEval,
    params      = sacParams,
    modelPath   = modelPath,
    plotPath    = plotPath,
    distributed = distributed
)


## Train
sacAgent.fit(resume=args.resume)