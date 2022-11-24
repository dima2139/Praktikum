'''
Implements the Soft Actor-Critic model
and training loop using Tensorflow-Agents.
Author: Daniel von Eschwege
Date:   12 November 2022
'''


## Imports
import os
import reverb
import datetime
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy, random_py_policy, policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, train_utils, strategy_utils
from tf_agents.utils import common

from scripts.const import *
from scripts.utils import *


## Setup
mpl.use('Agg')
tf.random.set_seed(SEED_TF)


## Class
class softActorCritic:
    def __init__(self, envCollect, envEval, params, modelPath, plotPath, distributed=True, useGPU=True, useTPU=False):
        '''
        Instantiate the Soft Actor-Critic.
        '''

        
        ## Class Variables
        self.envCollect = envCollect
        self.envEval    = envEval
        self.p          = params
        self.modelPath  = modelPath
        self.plotPath   = plotPath
        if distributed:
            self.strategy = strategy_utils.get_strategy(use_gpu=useGPU, tpu=useTPU)
        else:
            self.strategy = None

        
        ## Agent
        def createAgent():
            '''
            Define the reinforcement learning agent.
            '''


            observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(self.envCollect)

            critic_net = critic_network.CriticNetwork(
                input_tensor_spec       = (observation_spec, action_spec),
                joint_fc_layer_params   = self.p['critic_joint_fc_layer_params'],
                kernel_initializer      = self.p['kernel_initializer'],
                last_kernel_initializer = self.p['kernel_initializer']            
            )

            actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec         = observation_spec,
                output_tensor_spec        = action_spec,
                fc_layer_params           = self.p['actor_fc_layer_params'],
                continuous_projection_net = (tanh_normal_projection_network.TanhNormalProjectionNetwork),
                dtype                     = DTYPE
            )

            self.global_step = tf.compat.v1.train.get_or_create_global_step()

            critic_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = self.p['initial_learning_rate'],
                decay_steps           = self.p['decay_steps'],
                decay_rate            = self.p['decay_rate']
            )
            actor_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = self.p['initial_learning_rate'],
                decay_steps           = self.p['decay_steps'],
                decay_rate            = self.p['decay_rate']
            )
            alpha_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = self.p['initial_learning_rate'],
                decay_steps           = self.p['decay_steps'],
                decay_rate            = self.p['decay_rate']
            )
            self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
            self.actor_optimizer  = tf.keras.optimizers.Adam(actor_learning_rate)
            self.alpha_optimizer  = tf.keras.optimizers.Adam(alpha_learning_rate)

            self.agent = sac_agent.SacAgent(
                time_step_spec       = time_step_spec,
                action_spec          = action_spec,
                critic_network       = critic_net,
                actor_network        = actor_net,
                critic_optimizer     = self.critic_optimizer,
                actor_optimizer      = self.actor_optimizer,
                alpha_optimizer      = self.alpha_optimizer,
                target_update_tau    = self.p['target_update_tau'],
                target_update_period = self.p['target_update_period'],
                td_errors_loss_fn    = tf.math.squared_difference,
                gamma                = self.p['gamma'],
                reward_scale_factor  = self.p['reward_scale_factor'],
                train_step_counter   = self.global_step
            )
        
            self.agent.initialize()

        if self.strategy is not None:
            with self.strategy.scope():
                createAgent()
        else:
            createAgent()

    
    ## Training
    def fit(self, resume):
        '''
        Train the agent.
        '''


        ## Replay Buffer
        self.table_name = 'uniform_table'

        table = reverb.Table(
            name         = self.table_name,
            sampler      = reverb.selectors.Uniform(),
            remover      = reverb.selectors.Fifo(),
            max_size     = self.p['replay_buffer_capacity_steps'],
            rate_limiter = reverb.rate_limiters.MinSize(1)
        )

        self.reverb_server = reverb.Server([table])

        self.reverb_replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec       = self.agent.collect_data_spec,
            table_name      = self.table_name,
            sequence_length = self.p['replay_sequence_length'],
            local_server    = self.reverb_server
        )

        reverb_replay_buffer_dataset = self.reverb_replay_buffer.as_dataset(
            sample_batch_size = self.p['batch_size'],
            num_steps         = self.p['dataset_num_steps']
        ).prefetch(self.p['dataset_buffer_size'])

        self.experience_dataset_fn = lambda: reverb_replay_buffer_dataset


        ## Policies
        self.tf_eval_policy = self.agent.policy
        self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            policy          = self.tf_eval_policy,
            use_tf_function = True
        )

        self.tf_collect_policy = self.agent.policy
        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            policy          = self.tf_collect_policy,
            use_tf_function = True
        )

        self.random_policy = random_py_policy.RandomPyPolicy(
            time_step_spec = self.envCollect.time_step_spec(),
            action_spec    = self.envCollect.action_spec()
        )


        ## Actors
        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            py_client       = self.reverb_replay_buffer.py_client,
            table_name      = self.table_name,
            sequence_length = self.p['observer_sequence_length'],
            stride_length   = self.p['observer_stride_length']
        )


        pl('\n\n\n#----------------------Initial Collect----------------------#')
        pl(f'Time: {datetime.datetime.now()}')
        initial_collect_actor = actor.Actor(
            env           = self.envCollect,
            policy        = self.random_policy,
            train_step    = self.global_step,
            steps_per_run = self.p['initial_collect_steps'],
            observers     = [rb_observer]
        )
        initial_collect_actor.run()

        env_step_metric = py_metrics.EnvironmentSteps()

        collect_summary_dir = os.path.join(self.modelPath, learner.TRAIN_DIR)

        collect_actor = actor.Actor(
            env           = self.envCollect,
            policy        = self.collect_policy,
            train_step    = self.global_step,
            steps_per_run = self.p['collect_actor_steps_per_run'],
            metrics       = actor.collect_metrics(self.p['collect_actor_buffer_size']),
            summary_dir   = collect_summary_dir,
            observers     = [rb_observer, env_step_metric]
        )

        eval_summary_dir = os.path.join(self.modelPath, 'eval')

        eval_actor = actor.Actor(
            env              = self.envEval,
            policy           = self.eval_policy,
            train_step       = self.global_step,
            episodes_per_run = self.p['num_eval_episodes'],
            metrics          = actor.eval_metrics(self.p['num_eval_episodes']),
            summary_dir      = eval_summary_dir
        )


        ## Learners
        pl('\n\n\n#----------------------Policy Creation----------------------#')
        pl(f'Time: {datetime.datetime.now()}')
        saved_model_dir = os.path.join(self.modelPath, learner.POLICY_SAVED_MODEL_DIR)
        learning_triggers = [
            triggers.PolicySavedModelTrigger(
                saved_model_dir = saved_model_dir,
                agent           = self.agent,
                train_step      = self.global_step,
                interval        = self.p['policy_save_interval_steps']
            ),
            triggers.StepPerSecondLogTrigger(
                train_step = self.global_step,
                interval   = self.p['log_trigger_interval']
            )
        ]

        pl('\n\n\nThe agent already restores the checkpoint.\n\n\n')
        agent_learner = learner.Learner(
            root_dir              = self.modelPath,
            train_step            = self.global_step,
            agent                 = self.agent,
            experience_dataset_fn = self.experience_dataset_fn,
            triggers              = learning_triggers,
            strategy              = self.strategy
        )

        
        ## Checkpointing
        bufferCheckpointDir = os.path.join(self.modelPath, 'bufferCheckpointer')
        bufferCheckpointer = common.Checkpointer(
            ckpt_dir      = bufferCheckpointDir,
            max_to_keep   = self.p['max_to_keep'],
            replay_buffer = self.reverb_replay_buffer
        )
        bufferCheckpointer.initialize_or_restore()

        agentCheckpointDir = os.path.join(self.modelPath, 'agentCheckpointer')
        agentCheckpointer = common.Checkpointer(
            ckpt_dir      = agentCheckpointDir,
            max_to_keep   = self.p['max_to_keep'],
            agent         = self.agent,
            policy        = self.agent.policy,
            replay_buffer = self.reverb_replay_buffer,
            global_step   = self.global_step
        )
        agentCheckpointer.initialize_or_restore()

        policyDir = os.path.join(self.modelPath, 'policies', 'saved_policy')
        policySaver = policy_saver.PolicySaver(self.agent.policy)

        self.global_step = tf.compat.v1.train.get_global_step()


        ## Visualization
        def visualize(perf, name):
            '''
            Visualize the Soft Actor-Critic.
            '''
            for col in perf.columns:
                plt.plot(perf.index, perf[col])
                plt.ylabel(col)
                plt.xlabel('steps')
                plt.title(name)
                plt.savefig(f'{self.plotPath}/{name}_{col}.png')
                plt.close()

        
        # ## Training Loop
        # if resume:
        #     perfTrain = lpkl(f'{self.modelsPath}/perfTrain.pkl')
        #     perfEval  = lpkl(f'{self.modelsPath}/perfEval.pkl')
        # else:
        #     perfTrain = pd.DataFrame(columns=['reward'])
        #     perfEval  = pd.DataFrame(columns=['reward'])

        for i in range(self.p['max_train_steps']):
            pl(i)

            # Training
            step = agent_learner.train_step_numpy
            collect_actor.run()
            loss_info = agent_learner.run(iterations=1)

            # # Logging
            # if self.envCollect.current_time_step().is_last():
            #     # perfTrain.loc[step] = ??
            #     # if len(perfTrain) > self.p['numfuncs_train'] and perfTrain.loc[step, 'TrainGBN'] < bestTrainGBN:
            #     #     policySaver.save(f'{policyDir}_{step}')

            # Evaluation
            if i % self.p['eval_interval'] == 0:
                # count = len(perfEval)
                # pl(f'\n\n\n#---------------Evaluation Set {count} Start---------------#')
                pl(f'Time: {datetime.datetime.now()}')

                eval_actor.run()

                # metrics = {}
                # for metric in eval_actor.metrics:
                #     metrics[metric.name] = metric.result()
                # # perfEval.loc[step] = ?
                # # if perfEval.loc[step, 'EvalGBN'] < bestEvalGBN:
                # #     policySaver.save(f'{policyDir}_{step}')

                bufferCheckpointer.save(global_step=self.global_step)
                agentCheckpointer.save(global_step=self.global_step)

                # pl(', '.join([
                #     f'Step = {step}',
                #     f'TrainReward = {perfTrain.loc[step, "reward"]:.6f}',
                #     f'EvalReward = {perfEval.loc[step, "reward"]:.6f}',
                #     f'LearningRate = {self.alpha_optimizer._decayed_lr("float32").numpy():.6f}',
                #     f'EvalAverageEpisodeLength = {metrics["AverageEpisodeLength"]:.6f}',
                # ]))

                # visualize(perfTrain, name='Training')
                # visualize(perfEval,  name='Evaluation')
            
                # wpkl(f'{self.plotPath}/perfTrain.pkl', perfTrain)
                # wpkl(f'{self.plotPath}/perfEval.pkl',  perfEval)

                # pl(f'#---------------Evaluation Set {count} End---------------#\n\n\n')
                pl(f'Time: {datetime.datetime.now()}')


        rb_observer.close()
        self.reverb_server.stop()