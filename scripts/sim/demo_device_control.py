"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more computationally efficient since IK relies on the backend pybullet IK solver.


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""

import argparse

import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper



if __name__ == "__main__":

    def set_reward(best_reward, absolute_reward):
        if best_reward is False:
            reward = 0
            best_reward = absolute_reward
        
        if absolute_reward > 0.93:
            reward = absolute_reward
        else:
            reward = absolute_reward - best_reward
        
        if absolute_reward > best_reward:
            best_reward = absolute_reward

        return best_reward, reward


    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    controller_name = "OSC_POSE"
    
    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # controller_config['initial_qpos'] = [0.050, -0.259, 0.052, -1.442, -0.016, 1.169, -2.0]

    # controller_config["robot_configs"] = [
    #     {'gripper_type': None, 'initial_qpos': [0.050, -0.259, 0.052, -1.442, -0.016, 1.169, -2.0]},
    #     {'gripper_type': None, 'initial_qpos': [0.050, -0.259, 0.052, -1.442, -0.016, 1.169, -2.0]},
    # ]

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

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
        initialization_noise   = None,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    from robosuite.devices import Keyboard
    device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    env.viewer.add_keypress_callback(device.on_press)


    while True:
        # Reset the environment
        obs = env.reset()
        best_reward = False

        # env.robots[1].reset(deterministic=True)
        # env.robots[1].set_robot_joint_positions([-0.030, -0.470, -0.077, -2.078, -0.0, 1.82, 2.3])
        # env.robots[0].set_robot_joint_positions([0.050, -0.259, 0.052, -1.442, -0.016, 1.169, -0.676])
        # env.robots[1].set_robot_joint_positions([0.050, -0.259, 0.052, -1.442, -0.016, 1.169, -0.676])

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        while True:
            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

            # Get the newest action
            action, grasp = input2action(device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config)

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if last_grasp < 0 < grasp:
                if args.switch_on_grasp:
                    args.arm = "left" if args.arm == "right" else "right"
                if args.toggle_camera_on_grasp:
                    cam_id = (cam_id + 1) % num_cam
                    env.viewer.set_camera(camera_id=cam_id)
            # Update last grasp
            last_grasp = grasp

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                if args.arm == "right":
                    action = np.concatenate([action, rem_action])
                elif args.arm == "left":
                    action = np.concatenate([rem_action, action])
                else:
                    # Only right and left arms supported
                    print(
                        "Error: Unsupported arm specified -- "
                        "must be either 'right' or 'left'! Got: {}".format(args.arm)
                    )
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: env.action_dim]

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            # print(obs['hole_quat'])
            # print(np.sum(np.array(obs['hole_quat'])[1:]))
            # print(obs['peg_quat'])
            best_reward, reward = set_reward(best_reward, reward)
            # print(reward)
            # print(best_reward)
            # print()
            # env.robots[1].set_robot_joint_positions(np.array([0.020, -0.176, -0.038, -2.517, 0.0, 2.374, 0.71]))
            # if action.sum():
            #     argmax = np.abs(action).argmax()
            #     print(argmax)
            #     print(action[argmax])
            #     print()
            # print()
            # print(env.robots[0]._joint_positions)
            # print(env.robots[1]._joint_positions)
            print(obs['d'])
            print(obs['t'])
            print()
            env.render()


