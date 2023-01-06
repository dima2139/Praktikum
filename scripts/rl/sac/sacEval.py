from stable_baselines3 import SAC
from scripts.rl.sac.envPanda import envPanda
import torch

model = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/peg-XYZ--hole-XYZ--no_primitives/sac_resume_499800_steps.zip")
envEval  = envPanda(evalEnv=True)

obs = envEval.reset()
# obs = envEval.get_observation()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = envEval.step(action)
    envEval.render()
    if done:
      obs = envEval.reset()