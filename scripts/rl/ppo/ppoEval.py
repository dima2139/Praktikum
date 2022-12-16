## Inference
else:
    ppo       = PPO.load(f'{savePath}/{MODEL}')
    obs_panda = envEval.reset()
    for i in range(250):
        action_panda, _states_panda = ppo.predict(obs_panda)
        obs_panda, rewards_panda, dones_panda, info_panda = envEval.step(action_panda)