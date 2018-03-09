#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D
from std_msgs.msg import Float32MultiArray, String
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi

# control gains
K1 = 0.4
K2 = 0.8
K3 = 0.8

# tells the robot to stay still
# if it doesn't get messages within that time period
TIMEOUT = .5

# maximum velocity
V_MAX = 0.2

# maximim angular velocity
W_MAX = 1

class PoseController:

    def __init__(self):
        rospy.init_node('turtlebot_pose_controller', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = 0.0
        self.y_g = 0.0
        self.theta_g = 0.0

        # time last pose command was received
        self.cmd_pose_time = rospy.get_rostime()

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)

        ### YOUR CODE HERE ###

        # create a subscriber that receives Pose2D messages and
        # calls cmd_pose_callback. It should subscribe to '/cmd_pose'

        ### END OF YOUR CODE ###

    def gazebo_callback(self, data):
        if "turtlebot3_burger" in data.name:
            pose = data.pose[data.name.index("turtlebot3_burger")]
            twist = data.twist[data.name.index("turtlebot3_burger")]
            self.x = pose.position.x
            self.y = pose.position.y
            quaternion = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta = euler[2]

    def cmd_pose_callback(self, data):
        self.x_g = data.x
        self.y_g = data.y
        self.theta_g = data.theta
        self.cmd_pose_time = rospy.get_rostime()

    def get_ctrl_output(self):
        """ runs a simple feedback pose controller """

        if (rospy.get_rostime()-self.cmd_pose_time)<rospy.Duration.from_sec(TIMEOUT):
            rel_coords = np.array([self.x-self.x_g, self.y-self.y_g])
            R = np.array([[np.cos(self.theta_g), np.sin(self.theta_g)], [-np.sin(self.theta_g), np.cos(self.theta_g)]])
            rel_coords_rot = np.dot(R,rel_coords)

            th_rot = self.theta-self.theta_g
            rho = linalg.norm(rel_coords)
            ang = np.arctan2(rel_coords_rot[1],rel_coords_rot[0])+np.pi
            angs = wrapToPi(np.array([ang-th_rot, ang]))
            alpha = angs[0]
            delta = angs[1]

            V = K1*rho*np.cos(alpha)
            om = K2*alpha + K1*np.sinc(2*alpha/np.pi)*(alpha+K3*delta)

            # Apply saturation limits
            cmd_x_dot = np.sign(V)*min(V_MAX, np.abs(V))
            cmd_theta_dot = np.sign(om)*min(W_MAX, np.abs(om))
        else:
            cmd_x_dot = 0
            cmd_theta_dot = 0

        cmd = Twist()
        cmd.linear.x = cmd_x_dot
        cmd.angular.z = cmd_theta_dot
        return cmd

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            ctrl_output = self.get_ctrl_output()
            self.pub.publish(ctrl_output)
            rate.sleep()

if __name__ == '__main__':
    pctrl = PoseController()
    pctrl.run()
