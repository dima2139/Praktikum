"""
    Teleoperate robot with keyboard.
    adapted from robosuite.demos.demo_device_control.py
"""

import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard

if __name__ == "__main__":

    # Get controller config
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # Create argument configuration
    config = {
        "env_name"          : "TwoArmPegInHole",
        "robots"            : ["Panda", "Panda"],
        "controller_configs": controller_config,
        "env_configuration" : "single-arm-opposed",
    }

    # Create environment
    env = suite.make(
        **config,
        has_renderer           = True,
        has_offscreen_renderer = False,
        render_camera          = "agentview",
        ignore_done            = True,
        use_camera_obs         = False,
        reward_shaping         = True,
        control_freq           = 20,
        hard_reset             = False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device        
    device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
    env.viewer.add_keypress_callback("any", device.on_press)
    env.viewer.add_keyup_callback("any", device.on_release)
    env.viewer.add_keyrepeat_callback("any", device.on_press)
    
    while True:
        # Reset the environment
        obs   = env.reset()
        # amin  = np.full((12), np.inf)
        # amax  = np.full((12), -np.inf)
        # p0min = np.full((28), np.inf)
        # p0max = np.full((28), -np.inf)
        # p1min = np.full((28), np.inf)
        # p1max = np.full((28), -np.inf)
        # omin  = np.full((17), np.inf)
        # omax  = np.full((17), -np.inf)

        # Setup rendering
        cam_id = 0
        env.viewer.set_camera(camera_id=cam_id)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        arm = "right"
        # i = 0
        while True:
            # i += 1
            # Set active robot
            active_robot = env.robots[arm == "left"]

            # Get the newest action
            action, grasp = input2action(
                device            = device,
                robot             = active_robot,
                active_arm        = arm,
                env_configuration = config
            )

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed), toggle arm control 
            if last_grasp < 0 < grasp:
                arm = "left" if arm == "right" else "right"
            
            # Update last grasp
            last_grasp = grasp
            # print(grasp)

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                if arm == "right":
                    action = np.concatenate([action, rem_action])
                else:
                    action = np.concatenate([rem_action, action])
                
            # p2Step through the simulation and render
            obs, reward, done, info = env.step(action)

            # mask = action < amin
            # amin[mask] = action[mask]
            # mask = action > amax
            # amax[mask] = action[mask]

            # mask = obs['robot0_proprio-state'] < p0min
            # p0min[mask] = obs['robot0_proprio-state'][mask]
            # mask = obs['robot0_proprio-state'] > p0max
            # p0max[mask] = obs['robot0_proprio-state'][mask]

            # mask = obs['robot1_proprio-state'] < p1min
            # p1min[mask] = obs['robot1_proprio-state'][mask]
            # mask = obs['robot1_proprio-state'] > p1max
            # p1max[mask] = obs['robot1_proprio-state'][mask]

            # mask = obs['object-state'] < omin
            # omin[mask] = obs['object-state'][mask]
            # mask = obs['object-state'] > omax
            # omax[mask] = obs['object-state'][mask]

            # if i%2000==0:
            #     print(i)
            #     print('\n\n')
            #     print(amin)
            #     print(amax)
            #     print('\n\n')
            #     print(p0min)
            #     print(p0max)
            #     print('\n\n')
            #     print(p1min)
            #     print(p1max)
            #     print('\n\n')
            #     print(omin)
            #     print(omax)
            #     print('\n\n')

5            env.render()
