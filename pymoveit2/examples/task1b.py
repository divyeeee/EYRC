#!/usr/bin/env python3

import rclpy
import sys
import cv2
import numpy as np
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
import transforms3d as t3d

def calculate_rectangle_area(coordinates):
    if len(coordinates) != 4:
        raise ValueError("Input should contain 4 sets of coordinates.")

    width = abs(coordinates[0][0] - coordinates[1][0])
    height = abs(coordinates[0][1] - coordinates[3][1])
    area = width * height

    return area, width

def detect_aruco(image):
    aruco_area_threshold = 1500
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    size_of_aruco_m = 0.15

    center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids, tvecs, rvecs = [], [], [], [], [], [], []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    corners, aruco_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if aruco_ids is not None:
        image = cv2.aruco.drawDetectedMarkers(image, corners, aruco_ids)
    else:
        return [], [], [], [], [], [], []

    for i, aruco_id in enumerate(aruco_ids):
        area = cv2.contourArea(corners[i])
        if area < aruco_area_threshold:
            continue

        cX = ((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
        cY = ((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
        center_aruco_list.append((cX, cY))

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_of_aruco_m, cam_mat, dist_mat)
        rotation_matrix = cv2.Rodrigues(np.array([0, np.pi, 0]))[0]
        cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 1)

        if rvec is not None and rvec.shape == (1, 1, 3):
            distance_from_rgb = np.linalg.norm(tvec)
            tvecs.append(tvec)
            rvecs.append(rvec)

            roll, pitch, yaw = R.from_rotvec(rvec[0, 0]).as_euler('xyz', degrees=True)
            angle_aruco = yaw

            distance_from_rgb_list.append(distance_from_rgb)
            angle_aruco_list.append(angle_aruco)
            width_aruco_list.append(size_of_aruco_m)
            ids.append(aruco_id[0])

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids, tvecs, rvecs

def euler_to_quaternion(roll, pitch, yaw):
    return t3d.euler.euler2quat(roll, pitch, yaw, 'sxyz').tolist()

class ArucoTf(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.2, self.process_image)

        self.cv_image = None
        self.depth_image = None

    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.get_logger().info(f'Depth image shape: {self.depth_image.shape}')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            self.get_logger().info(f'Color image shape: {self.cv_image.shape}')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def process_image(self):
        if self.cv_image is None:
            return

        center_aruco_list, distance_from_rgb_list, angle_aruco_list, _, ids, tvecs, rvecs = detect_aruco(self.cv_image)

        for i, marker_id in enumerate(ids):
            cX, cY = center_aruco_list[i]
            distance_from_rgb = distance_from_rgb_list[i]
            tvec = tvecs[i]
            rvec = rvecs[i]

            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'camera_color_optical_frame'
            transform.child_frame_id = f'cam_{marker_id}'
            transform.transform.translation.x = float(tvec[0][0][0])
            transform.transform.translation.y = float(tvec[0][0][1])
            transform.transform.translation.z = float(tvec[0][0][2])

            rot_mat = R.from_rotvec(rvec[0, 0])
            rot_180_y = R.from_euler('xyz', [0, 180, 0], degrees=True)
            q = (rot_mat * rot_180_y).as_quat()
            transform.transform.rotation.x = float(q[0])
            transform.transform.rotation.y = float(q[1])
            transform.transform.rotation.z = float(q[2])
            transform.transform.rotation.w = float(q[3])

            self.br.sendTransform(transform)

            if self.tf_buffer.can_transform('base_link', f'cam_{marker_id}', rclpy.time.Time()):
                base_transform = self.tf_buffer.lookup_transform('base_link', f'cam_{marker_id}', rclpy.time.Time())
                base_transform.header.stamp = self.get_clock().now().to_msg()
                base_transform.header.frame_id = 'base_link'
                base_transform.child_frame_id = f'obj_{marker_id}'
                self.br.sendTransform(base_transform)

                self.get_logger().info(f'Successfully received data for marker {marker_id}')

        cv2.imshow("ArUco Detection", self.cv_image)
        cv2.waitKey(1)

def main():
    rclpy.init(args=sys.argv)
    aruco_tf_node = ArucoTf()
    rclpy.spin(aruco_tf_node)
    aruco_tf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()