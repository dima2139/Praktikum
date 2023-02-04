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

from scripts.const import *



def set_step_reward(env, env_observation, previous_reward=None, reset=False):

        if PRIMITIVE=='align':
            if np.all(MIN_BBOX_PEG <= env_observation['robot0_eef_pos']) and np.all(env_observation['robot0_eef_pos'] <= MAX_BBOX_PEG):
                angle_mag_peg  = np.linalg.norm(QUAT_ANGLES_PEG - env_observation["peg_quat"]) / len(QUAT_ANGLES_PEG)
            else:
                angle_mag_peg  = 0.5

            if np.all(MIN_BBOX_HOLE <= env_observation['robot1_eef_pos']) and np.all(env_observation['robot1_eef_pos'] <= MAX_BBOX_HOLE):
                angle_mag_hole = np.linalg.norm(QUAT_ANGLES_HOLE - env_observation["hole_quat"]) / len(QUAT_ANGLES_HOLE)
            else:
                angle_mag_hole = 0.5
            
            reward = 1 - (angle_mag_peg + angle_mag_hole)
        
        elif PRIMITIVE=='d':
            if np.all(MIN_BBOX_PEG <= env_observation['robot0_eef_pos']) and np.all(env_observation['robot0_eef_pos'] <= MAX_BBOX_PEG) \
            and np.all(MIN_BBOX_HOLE <= env_observation['robot1_eef_pos']) and np.all(env_observation['robot1_eef_pos'] <= MAX_BBOX_HOLE):
                reward = 1 - np.tanh(env_observation[PRIMITIVE])
            else:
                reward = 0

        elif PRIMITIVE == "t":
            hole_pos         = env.sim.data.body_xpos[env.hole_body_id]
            gripper_site_pos = env.sim.data.body_xpos[env.peg_body_id]
            dist             = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward  = 1 - np.tanh(1.0 * dist)
            reward           = (reaching_reward + (1 - np.tanh(np.abs(env_observation[PRIMITIVE])))) / 2
        

        if reset:
            previous_reward = reward
            
            return previous_reward

        else:
            step_reward = reward - previous_reward
            previous_reward = reward

            return step_reward, previous_reward



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary")
    # parser.add_argument("--robot_init_qpos", nargs="+", type=float, default=[ 0. , 0.19634954,  0. , -2.61799388,  0., 2.94159265, 0.78539816], help="initial q_pos of robot(s)")
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
    
    # Create argument configuration
    config = {
        "env_name"          : args.environment,
        "robots"            : args.robots,
        # "initial_qpos"      : np.array(args.robot_init_qpos),
        # "initial_qpos"      : np.array([[0.008, -1.226, 0.026, -2.680, -0.074, 2.987, 0.815], [0.045, -0.776, -0.044, -2.217, 0.006, 1.452, -0.744]]),
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
        initialization_noise   = {'type': 'uniform', 'magnitude': 0.05}
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
        previous_reward = set_step_reward(env, env_observation=obs, reset=True)
        best_reward = False
        
        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)

        # env.robots[0].set_robot_joint_positions([0.041, -0.829, -0.055, -2.301, 0.022, 3.058, 0.706])
        # env.robots[1].set_robot_joint_positions([-0.046, -1.132, -0.176, -2.620, -0.138, 1.518, -0.914])
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        episode_reward = 0

        # env.robots[1].set_robot_joint_positions([0.79456863, 0.93868992, 0.61284948, -2.55324803, 1.06315638,  3.67426987, -0.68658793])

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

            # # Step through the simulation and render
            if action.sum():
                action_flag = True
            else:
                action_flag = False
 
            obs, reward, done, info = env.step(action)
            
            step_reward, previous_reward = set_step_reward(env, env_observation=obs, previous_reward=previous_reward, reset=False)
            episode_reward += step_reward

            if action_flag:
                print(f'peg_quat: {obs["peg_quat"]}, hole_quat: {obs["hole_quat"]}')
                print(f'robot0_eef_pos: {obs["robot0_eef_pos"]}, robot1_eef_pos: {obs["robot1_eef_pos"]}')
                print(f'reward: {reward:.7f}, step_reward: {step_reward:.7f}, episode_reward: {episode_reward:.7f}')
                print()

            env.render()