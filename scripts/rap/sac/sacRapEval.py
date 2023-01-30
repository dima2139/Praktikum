from stable_baselines3 import SAC
from scripts.rap.sac.envRapPanda import envRapPanda

model = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/peg_XYZ__hole_XYZ__no_primitives/checkpoints/sac_resume_499800_steps.zip")
envEval  = envRapPanda(evalEnv=True)

obs = envEval.reset()
# obs = envEval.get_observation()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = envEval.step(action)
    envEval.render()
    if done:
      obs = envEval.reset()