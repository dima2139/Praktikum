import time
from stable_baselines3 import SAC
from scripts.rap.sac.envRapsPanda import envRapsPanda
from scripts.const import *

model_align = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/primitive_align/checkpoints/sac_resume_horizon100_102000_steps.zip")
model_d     = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/primitive_d/checkpoints/sac_resume_216000_steps.zip")
model_t     = SAC.load("/home/dve/Desktop/g2-peg-in-hole/models/sac/primitive_t/checkpoints/sac_resume_189000_steps.zip")
envEval     = envRapsPanda(evalEnv=True)

obs  = envEval.reset()
done = False
while True:
    time.sleep(2)
    while not envEval.primitive_align_done:
        action, _states = model_align.predict(obs, deterministic=True)
        obs, reward, done, info = envEval.step(action, primitive="align")
        envEval.render()

    time.sleep(2)
    while not envEval.primitive_d_done:
        action, _states = model_d.predict(obs, deterministic=True)
        obs, reward, done, info = envEval.step(action, primitive="d")
        envEval.render()

    time.sleep(2)
    while not done:
        action, _states = model_t.predict(obs, deterministic=True)
        obs, reward, done, info = envEval.step(action, primitive="t")
        envEval.render()

    obs = envEval.reset()