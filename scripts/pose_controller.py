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

# maximum velocity
V_MAX = 0.25

# maximim angular velocity
W_MAX = 2

class PoseController:

    def __init__(self):
        rospy.init_node('turtlebot_pose_controller', anonymous=True)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = 0.0
        self.y_g = 0.0
        self.theta_g = 0.0

        self.trans_listener = tf.TransformListener()

        rospy.Subscriber('/cmd_pose', Pose2D, self.cmd_pose_callback)

    def cmd_pose_callback(self, data):
        self.x_g = data.x
        self.y_g = data.y
        self.theta_g = data.theta
        self.run_pose_controller()

    def run_pose_controller(self):
        """ runs a simple feedback pose controller """

        try:
            (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            self.x = translation[0]
            self.y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            self.theta = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

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

        cmd_msg = Twist()
        cmd_msg.linear.x = cmd_x_dot
        cmd_msg.angular.z = cmd_theta_dot
        self.cmd_vel_publisher.publish(cmd_msg)


if __name__ == '__main__':
    pctrl = PoseController()
    rospy.spin()
