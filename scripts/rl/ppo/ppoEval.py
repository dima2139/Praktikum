## Inference
from stable_baselines3 import PPO
from scripts.rl.ppo.envPanda import envPanda

envEval  = envPanda(evalEnv=True)
ppo       = PPO.load('/home/dve/Desktop/g2-peg-in-hole/models/ppo/rl_model_1671218080/ppo_final.zip')
obs_panda = envEval.reset()
for i in range(10):
    action_panda, _states_panda = ppo.predict(obs_panda)
    obs_panda, rewards_panda, dones_panda, info_panda = envEval.step(action_panda)