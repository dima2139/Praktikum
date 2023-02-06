#made with the help of chatGPT
#WARNING!!!!! NOT TESTED
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def ros_planning(current_position, goal_position)

    rospy.init_node("moveit_test_node")

    # Step 1: Set up a MoveIt planning scene
    group = moveit_commander.MoveGroupCommander("end_effector_link")

    # Step 2: Get the current robot state
    start_state = current_position

    # Step 3: Specify the goal pose
    goal_pose = Pose()
    goal_pose.position.x = goal_position[0]
    goal_pose.position.y = goal_position[1]
    goal_pose.position.z = goal_position[2]
    goal_pose.orientation.w = goal_position[3]

    # Step 4: Plan the motion
    return group.plan(goal_pose)

