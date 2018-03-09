#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, CameraInfo

CHOP_TOP = 0
CHOP_BOT = 200

class ImageTransformer:

    def __init__(self):
        rospy.init_node('tb3_image_transformer', anonymous=True)
        rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.transform_image_and_repub, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/raspicam_node/camera_info', CameraInfo, self.transform_camera_info_and_repub)
        self.image_repub = rospy.Publisher('/camera/image_raw/compressed', CompressedImage, queue_size=10)
        self.camera_info_repub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=10)

    def transform_image_and_repub(self, msg):
        img = cv2.imdecode(np.fromstring(msg.data, np.uint8), cv2.IMREAD_COLOR)
        img_rotated = np.flip(np.transpose(img, (1, 0, 2)), 0)[CHOP_TOP:-CHOP_BOT]
        repub_msg = CompressedImage()
        repub_msg.header = msg.header
        repub_msg.format = "jpeg"
        repub_msg.data = np.array(cv2.imencode('.jpg', img_rotated)[1]).tostring()
        self.image_repub.publish(repub_msg)

    def transform_camera_info_and_repub(self, msg):
        msg.height, msg.width = msg.width, msg.height
        D = list(msg.D)
        D[2], D[3] = -D[3], D[2]    # I'm reasonably sure about this
        msg.D = D
        K = list(msg.K)
        K[0], K[2], K[4], K[5] = K[4], K[5], K[0], msg.height - K[2] - CHOP_TOP
        msg.K = K
        P = list(msg.P)
        P[0], P[2], P[5], P[6] = P[5], P[6], P[0], msg.height - P[2] - CHOP_TOP
        msg.P = P
        self.camera_info_repub.publish(msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    r = ImageTransformer()
    r.run()
