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
from setJoints import set_joints
from rotate_to_plane import rotate_to_plane
from quat_to_euler import quat_to_euler
from align_to_axis import align_to_axis
from align import align

if __name__ == "__main__":  


    #controller_fpath = "/home/dima/Desktop/Praktikum/g2-peg-in-hole/scripts/sim/osc_pose.json"
    # Get controller config
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # Create argument configuration
    config = {
        "env_name": "TwoArmPegInHole",
        "robots": ["Panda","Panda"],
        "controller_configs": controller_config,
        "env_configuration": "single-arm-opposed",
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
        hard_reset             = False
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device        
    device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
    env.viewer.add_keypress_callback('any', device.on_press)
    env.viewer.add_keyup_callback('any', device.on_release)
    env.viewer.add_keyrepeat_callback('any', device.on_press)
    
    while True:
        # Reset the environment
        obs = env.reset()
        flag = True

        #env.robots[0].reset(deterministic=True)
        #env.robots[1].reset(deterministic=True)
        #env.robots[0].set_robot_joint_positions(np.array([0, 0.5, 0.4, 0.3, -0.01, 0.1, 0.5]))
        #env.robots[0].set_robot_joint_positions(set_joints(free_joints=1, robot="peg"))
        #env.robots[1].set_robot_joint_positions(set_joints(free_joints=1, robot="hole"))
        # Setup rendering
        cam_id = 0
        env.viewer.set_camera(camera_id=cam_id)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        arm = "right"
        while True:
            # Set active robot
            active_robot = env.robots[arm == "left"]

            # Get the newest action
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm=arm, env_configuration=config
            )

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control 
            if last_grasp < 0 < grasp:
                arm = "left" if arm == "right" else "right"
            # Update last grasp
            last_grasp = grasp

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
                
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: env.action_dim]
            if action.sum():
                quat = "hole_quat"
                quatp = "peg_quat"
                print(env._observables[quatp].obs[0], env._observables[quatp].obs[1], env._observables[quatp].obs[2], env._observables[quatp].obs[3])
                print(env._observables[quat].obs[0], env._observables[quat].obs[1], env._observables[quat].obs[2], env._observables[quat].obs[3])
                #print(quat_to_euler(env._observables[quat].obs[0], env._observables[quat].obs[1], env._observables[quat].obs[2], env._observables[quat].obs[3]))
            # Step through the simulation and render
            if flag:
                align(1, direction=1, env=env, robot="peg")
                flag = False
            
            action /= 10
            obs, reward, done, info = env.step(action)
            env.render()
