from stable_baselines3 import SAC
from scripts.rl.sac.envPanda import envPanda

model = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/1671459832/sac_35750_steps.zip")
envEval  = envPanda(evalEnv=True)

obs = envEval.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = envEval.step(action)
    envEval.render()
    if done:
      obs = envEval.reset()