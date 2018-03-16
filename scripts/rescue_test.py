#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool

class RescueTest:

    def __init__(self):
        rospy.init_node('rescuers_down_under_mate', anonymous=True)
        rospy.Subscriber('ready_to_rescue', Bool, self.rescue_callback)
        self.rescuing = False
        self.rescue_pub = rospy.Publisher('rescue_on', Bool, queue_size=10)

    def rescue_callback(self, msg):
        self.rescuing = msg.data

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.rescue_pub.publish(Bool(self.rescuing))
            rate.sleep()

if __name__ == '__main__':
    s = RescueTest()
    s.run()
