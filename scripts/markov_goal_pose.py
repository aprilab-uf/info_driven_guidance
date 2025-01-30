#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry


class MarkovChain:
    def __init__(self):
        self.is_time_mode = (
            False  # Change this to False if you want to use distance mode
        )

        self.tolerance_radius = 0.2  # meters

        self.init_time = np.array(rospy.get_time())
        self.prev_goal_in = 1  # Start at the second state
        self.prev_mult = 0
        self.position = np.array([0, 0])

        # ROS stuff
        self.is_sim = rospy.get_param("/is_sim", False)
        rospy.loginfo(
            "Initializing markov_goal_pose node with parameter is_sim: {}".format(
                self.is_sim
            )
        )
        if self.is_sim:
            self.pose_pub = rospy.Publisher("goal_pose", Pose, queue_size=2)
            self.p = Pose()
            self.turtle_odom_sub = rospy.Subscriber(
                "/robot0/odom", Odometry, self.odom_cb, queue_size=1
            )
        else:
            self.pose_pub = rospy.Publisher("goal_pose", PoseStamped, queue_size=2)
            self.p = PoseStamped()
            self.turtle_odom_sub = rospy.Subscriber(
                "agent_pose", PoseStamped, self.odom_cb, queue_size=1
            )

        self.goal_pose_square()

    def goal_pose_square(self):
        """Generates an square of sides 2*k"""
        self.goal_list = []

        z = 0  # turtlebot on the ground
        qx = qy = 0  # no roll or pitch
        k = 0.8  # Multiplier  change this to make square bigger or smaller
        x_offset = -1.25  # change this to not crash to the net
        y_offset = 0.2
        self.goal_list.append(
            {
                "curr_goal": 0,
                "x": x_offset + 0 * k,
                "y": y_offset + 0 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0,
                "qw": 1,
            }
        )
        self.goal_list.append(
            {
                "curr_goal": 1,
                "x": x_offset + 0 * k,
                "y": y_offset + -1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0.707,
                "qw": 0.707,
            }
        )  # -90 degrees orientation
        self.goal_list.append(
            {
                "curr_goal": 2,
                "x": x_offset + 2 * k,
                "y": y_offset + -1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0,
                "qw": 1,
            }
        )  # 0 degrees orientation
        self.goal_list.append(
            {
                "curr_goal": 3,
                "x": x_offset + 2 * k,
                "y": y_offset + 1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 0.707,
                "qw": -0.707,
            }
        )  # 90 degrees orientation
        self.goal_list.append(
            {
                "curr_goal": 4,
                "x": x_offset + 1 * k,
                "y": y_offset + 2 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 1,
                "qw": 0,
            }
        )  # 180 degrees orientation
        self.goal_list.append(
            {
                "curr_goal": 5,
                "x": x_offset + 0 * k,
                "y": y_offset + 1 * k,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": 1,
                "qw": 0,
            }
        )  # 180 degrees orientation
        self.n_states = len(self.goal_list)

        # transition matrix:
        # prob of going from state i to state j
        # in the goal_list states where i is the
        # row and j is the column
        self.trans_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.6, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    def odom_cb(self, msg):
        if self.is_sim:
            self.position = np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y]
            )
        else:
            self.position = np.array([msg.pose.position.x, msg.pose.position.y])

    def pub_goal_pose(self):
        """Publishes a goal pose after the goal is reached within tolerance_radius"""
        curr_goal_in = np.copy(self.prev_goal_in)
        curr_goal_pose = self.goal_list[self.prev_goal_in]
        if self.is_time_mode:
            time_step = 10  # amount of seconds until next goal pose change if desired
            now = rospy.get_time() - self.init_time
            mult = np.floor(now / time_step)
            # change goal pose if time is greater than time_step
            change = True if mult > self.prev_mult else False

            if now > 0 and now < time_step:
                curr_goal_in = 0  # Start at the first state
            elif change:
                curr_goal_in = np.random.choice(
                    np.arange(self.n_states), p=self.trans_matrix[curr_goal_in, :]
                )

            self.prev_mult = mult
        else:
            dist_to_goal = np.linalg.norm(
                self.position - np.array([curr_goal_pose["x"], curr_goal_pose["y"]])
            )
            #  print("dist to goal: ", dist_to_goal)
            if dist_to_goal < self.tolerance_radius:
                curr_goal_in = np.random.choice(
                    np.arange(self.n_states), p=self.trans_matrix[self.prev_goal_in, :]
                )

        goal_pose = self.goal_list[curr_goal_in]
        if curr_goal_in != self.prev_goal_in:
            rospy.logwarn(
                "New goal pose: x={:.2f}, y={:.2f} with index {}".format(
                    goal_pose["x"], goal_pose["y"], curr_goal_in
                )
            )

        # Restart previous values
        self.prev_goal_in = np.copy(curr_goal_in)
        # Publish the goal pose
        self.create_pose_msg(goal_pose)

    def create_pose_msg(self, goal_pose):
        if self.is_sim:
            self.p.position.x = goal_pose["x"]
            self.p.position.y = goal_pose["y"]
            self.p.position.z = goal_pose["z"]
            self.p.orientation.x = goal_pose["qx"]
            self.p.orientation.y = goal_pose["qy"]
            self.p.orientation.z = goal_pose["qz"]
            self.p.orientation.w = goal_pose["qw"]
        else:
            self.p.pose.position.x = goal_pose["x"]
            self.p.pose.position.y = goal_pose["y"]
            self.p.pose.position.z = goal_pose["z"]
            self.p.pose.orientation.x = goal_pose["qx"]
            self.p.pose.orientation.y = goal_pose["qy"]
            self.p.pose.orientation.z = goal_pose["qz"]
            self.p.pose.orientation.w = goal_pose["qw"]


if __name__ == "__main__":
    rospy.init_node("goal_pose_node", anonymous=True)
    rate = rospy.Rate(10)  # Hz
    square_chain = MarkovChain()
    while not rospy.is_shutdown():
        square_chain.pub_goal_pose()
        square_chain.pose_pub.publish(square_chain.p)
        rate.sleep()
