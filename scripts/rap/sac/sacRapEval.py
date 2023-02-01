from stable_baselines3 import SAC
from scripts.rap.sac.envRapPanda import envRapPanda
from scripts.const import *

model = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/primitive_align/sac_resume_135000_steps.zip")
envEval  = envRapPanda(evalEnv=True, primitive=PRIMITIVE)

obs = envEval.reset()
# obs = envEval.get_observation()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = envEval.step(action)
    envEval.render()
    if done:
      obs = envEval.reset()