#made with the help of chatGPT
#WARNING!!!!! Its just the example how to use MoveIt library for motion planning
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

rospy.init_node("moveit_test_node")

# Step 1: Set up a MoveIt planning scene
group = moveit_commander.MoveGroupCommander("end_effector_link")

# Step 2: Get the current robot state
start_state = group.get_current_state()

# Step 3: Specify the goal pose
goal_pose = Pose()
goal_pose.position.x = 0.5
goal_pose.position.y = 0.5
goal_pose.position.z = 0.5
goal_pose.orientation.w = 1.0

# Step 4: Plan the motion
plan = group.plan(goal_pose)

# Step 5: Execute the motion
group.execute(plan)
