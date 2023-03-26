from stable_baselines3 import SAC
from scripts.rl.sac.envPanda import envPanda

model = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/end_to_end/sac_resume_68000_steps.zip")
envEval  = envPanda(evalEnv=True, savePath=None)

obs = envEval.reset()
# obs = envEval.get_observation()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = envEval.step(action)
    envEval.render()
    if done:
      obs = envEval.reset()