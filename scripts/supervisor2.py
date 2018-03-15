#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from asl_turtlebot.msg import DetectedObject
import tf
import math
from enum import Enum
import numpy as np
import time

# threshold at which we consider the robot at a location
POS_EPS = .1
THETA_EPS = .3
POS_THRESHOLD = .1
THETA_THRESHOLD = 1.5

# time to stop at a stop sign
STOP_TIME = 7

# minimum distance from a stop sign to obey it
STOP_MIN_DIST = .5

#minimum distance from cat to log it
CAT_MIN_DIST = 0.7

# time taken to cross an intersection
CROSSING_TIME = 3

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 1
    POSE = 2
    STOP = 3
    CROSS = 4
    NAV = 5
    MANUAL = 6
    CAT = 7

class Supervisor:
    """ the state machine of the turtlebot """

    def __init__(self):
        rospy.init_node('turtlebot_supervisor', anonymous=True)

        # current pose
        self.x = 0
        self.y = 0
        self.theta = 0

        # pose goal
        self.x_g = 0
        self.y_g = 0
        self.theta_g = 0

        self.start_time = time.time()

        # current mode
        self.mode = Mode.IDLE
        self.last_mode_printed = None

        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.cat_location_publisher = rospy.Publisher('/cat_loc', Pose2D, queue_size=10)

        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber('/detector/cat', DetectedObject, self.cat_detector_callback)

        self.goal_bool = [False for i in range(7)]
        self.goal_number = 0

        self.goals = np.array([[8.5, 33, np.pi/2], [13.5, 38.5, 0], [43, 33, -np.pi/2], [43, -30, -np.pi/2], [13, -40, np.pi], [8.5, -32, np.pi/2], [8.5, 5, np.pi/2]])

        self.trans_listener = tf.TransformListener()

    def inch_to_meter_pose(self, pose):
        return np.array([pose[0]*.0254, pose[1]*.0254, pose[2]])

    def meter_to_inch_pose(self, pose):
        return np.array([pose[0]/.0254, pose[1]/.0254, pose[2]])

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """

        self.x_g = msg.pose.position.x
        self.y_g = msg.pose.position.y
        rotation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(rotation)
        self.theta_g = euler[2]
        self.mode = Mode.NAV

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance
        print "msg", msg.corners
        # if close enough and in nav mode, stop
        if dist >= 0 and dist < STOP_MIN_DIST and self.mode == Mode.NAV:
            self.init_stop_sign()

    def cat_detector_callback(self, msg):
        dist = msg.distance
        print "I see a cat!", msg.corners
        #if dist >= 0 and dist < CAT_MIN_DIST and self.mode == Mode.NAV:
        if dist >= 0:
            self.init_cat()

    def go_to_pose(self):
        """ sends the current desired pose to the pose controller """

        pose_g_msg = Pose2D()
        pose_g_msg.x = self.x_g
        pose_g_msg.y = self.y_g
        pose_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(pose_g_msg)

    def nav_to_pose(self):
        """ sends the current desired pose to the naviagtor """

        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.nav_goal_publisher.publish(nav_g_msg)

    def nav_to_pose_goal(self):
        """ sends the current desired pose to the naviagtor """
        pose = self.inch_to_meter_pose(self.goals[self.goal_number])
        nav_g_msg = Pose2D()
        nav_g_msg.x = pose[0]
        nav_g_msg.y = pose[1]
        nav_g_msg.theta = pose[2]
        print nav_g_msg.x, nav_g_msg.y, nav_g_msg.theta
        self.nav_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self,x,y,theta):
        """ checks if the robot is at a pose within some threshold """
        print self.x, x, abs(x-self.x)
        return (abs(x-self.x)<POS_EPS and abs(y-self.y)<POS_EPS and abs(theta-self.theta)<THETA_EPS)

    def close_to2(self,x,y,theta):
        """ checks if the robot is at a pose within some threshold """
        print theta, self.theta
        print abs(x-self.x)<POS_THRESHOLD and abs(y-self.y)<POS_THRESHOLD and abs(theta-self.theta)<THETA_THRESHOLD
        print self.x, x, abs(x-self.x)
        return (abs(x-self.x)<POS_THRESHOLD and abs(y-self.y)<POS_THRESHOLD and abs(theta-self.theta)<THETA_THRESHOLD)

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def init_cat(self):

        #self.cat_start = rospy.get_rostime()
        self.mode = Mode.CAT

    def log_cat(self):
        cat_loc_msg = Pose2D()
        cat_loc_msg.x = self.x
        cat_loc_msg.y = self.y
        cat_loc_msg.theta = self.theta

        self.cat_location_publisher.publish(cat_loc_msg)



    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return (self.mode == Mode.STOP and (rospy.get_rostime()-self.stop_sign_start)>rospy.Duration.from_sec(STOP_TIME))


    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return (self.mode == Mode.CROSS and (rospy.get_rostime()-self.cross_start)>rospy.Duration.from_sec(CROSSING_TIME))

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        try:
            (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            self.x = translation[0]
            self.y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            self.theta = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        #if time.time() - self.start_time > 5.0 and \
           #(self.goal_bool[self.goal_number] == False):
            #self.mode = Mode.NAV

        if (all(val for val in self.goal_bool)):
            self.mode = Mode.IDLE

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode

        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.IDLE:
            # send zero velocity
            self.stay_idle()

        elif self.mode == Mode.POSE:
            # moving towards a desired pose
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.IDLE
            else:
                self.go_to_pose()

        elif self.mode == Mode.STOP:
            # at a stop sign
            print 'hi'
            self.stay_idle()
            if self.has_stopped():
                self.init_crossing()

        elif self.mode == Mode.CROSS:
            # crossing an intersection
            if self.has_crossed():
                self.mode = Mode.NAV
            else:
                self.nav_to_pose()

        elif self.mode == Mode.NAV:
            print self.goals[self.goal_number],\
                 self.meter_to_inch_pose(np.array([self.x, self.y, self.theta]))
            if self.close_to(self.goals[self.goal_number][0]*0.0254,\
                             self.goals[self.goal_number][1]*0.0254,\
                             self.goals[self.goal_number][2]):
                print "moving to the next one"
                self.stay_idle()
                time.sleep(1)
                self.goal_bool[self.goal_number] = True
                self.goal_number += 1
            else:
                self.nav_to_pose_goal()

        elif self.mode == Mode.CAT:
            self.log_cat()
            self.mode = Mode.NAV

        else:
            raise Exception('This mode is not supported: %s'
                % str(self.mode))

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

if __name__ == '__main__':
    sup = Supervisor()
    sup.run()

# Thhings to ask ED:
# - Cat detection is bad
# - How are we supposed to do turns
# - Whats the best way to autonomously explore
# - Why is our robot keep running into obstacles. Shouldn't A* take care of things


# current crop - 100px top, 150px bottom
# 22in away, cat is 6in off the ground- first trial (19, 84, 73, 169),
# 19.5in cat is 6 in off the ground- ssecond trial (5, 104, 70, 200), 94
# 22in away, cat is 6in off the ground (19, 98, 70, 175), at an angle
# 31in away, cat is 6in off the ground (50, 118, 90, 177) straight ahead
# new cat, is 4.5in tall, 88px tall, 18in away
# cat #2, is 4.5in tall


# stop sign:
# the height was 64. stop sign was 13.5 in away
# 75px tall, stop sign was 10.5in
# 41p ttall, stop sign was 24.25in away
