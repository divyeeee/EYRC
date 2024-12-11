#!/usr/bin/env python3

import time
from copy import deepcopy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
import cv2
import cv2.aruco as aruco
import numpy as np
import tf2_ros

class ArucoServoControl(Node):
    def __init__(self):
        super().__init__('aruco_servo_control')

        # Camera subscribers
        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimage_callback, 10)

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.jointstate_callback, 10)
        self.joint_positions = []

        self.bridge = CvBridge()
        self.cv_image = None

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Servo publisher
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # Timer to process the image periodically
        self.timer = self.create_timer(0.2, self.process_image)

        # Service client to start the servo node
        self.servo_start_client = self.create_client(Trigger, '/servo_node/start_servo')

        # Variables for controlling movement
        self.aruco_positions = []

        # Proportional gain
        self.linear_gain = 0.5
        self.angular_gain = 0.5
        self.position_tolerance = 0.02

        # Default damping factor for DLS (dynamic adjustment)
        self.damping_factor_base = 0.1  # Base damping factor
        self.max_damping_factor = 1.0   # Maximum damping factor
        self.singularity_threshold = 50  # Condition number threshold for singularity

        # Start the servo node (triggered once)
        self.start_servo()

    def start_servo(self):
        self.get_logger().info("Waiting for the servo start service...")
        self.servo_start_client.wait_for_service()

        self.get_logger().info("Triggering the servo node...")
        trigger_request = Trigger.Request()
        future = self.servo_start_client.call_async(trigger_request)
        future.add_done_callback(self.servo_start_callback)

    def servo_start_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Servo node started successfully.")
            else:
                self.get_logger().error(f"Failed to start servo node: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def jointstate_callback(self, msg):
        """Update the current joint states."""
        self.joint_positions = msg.position

    def colorimage_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def detect_aruco(self, image):
        cam_mat = np.array([[931.1829833984375, 0.0, 640.0], 
                            [0.0, 931.1829833984375, 360.0], 
                            [0.0, 0.0, 1.0]])
        dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        size_of_aruco_m = 0.15
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, aruco_ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        tvecs = []
        rvecs = []
        if aruco_ids is None:
            return [], [], []  # No markers detected

        for corner in corners:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, size_of_aruco_m, cam_mat, dist_mat)
            tvecs.append(tvec)
            rvecs.append(rvec)

        return corners, aruco_ids, tvecs, rvecs

    def process_image(self):
        if self.cv_image is None:
            self.get_logger().warn("No image received yet.")
            return

        corners, aruco_ids, tvecs, rvecs = self.detect_aruco(self.cv_image)
        self.aruco_positions.clear()

        if len(aruco_ids) > 0:
            for tvec, rvec, aruco_id in zip(tvecs, rvecs, aruco_ids):
                aruco_position_base = self.transform_aruco_to_base(tvec)
                aruco_orientation_base = self.transform_aruco_orientation_to_base(rvec)

                self.aruco_positions.append((aruco_id, aruco_position_base, aruco_orientation_base))

                self.get_logger().info(f'Detected ArUco {aruco_id}: (x, y, z) = {aruco_position_base}, quaternion = {aruco_orientation_base}')

                # Apply singularity check before moving
                if self.check_singularity():
                    self.get_logger().warn("Singularity detected! Scaling velocity.")
                else:
                    # Trigger servo motion to move to ArUco marker
                    self.move_ur5_to_aruco(aruco_position_base, aruco_orientation_base)

                cv2.aruco.drawDetectedMarkers(self.cv_image, corners, aruco_ids)
        else:
            self.get_logger().info("No ArUco markers detected.")

        cv2.imshow("ArUco Detection", self.cv_image)
        cv2.waitKey(1)

    def transform_aruco_to_base(self, tvec):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_color_optical_frame', rclpy.time.Time())
            
            trans_vector = np.array([trans.transform.translation.x, 
                                     trans.transform.translation.y, 
                                     trans.transform.translation.z])
            rot_quat = [trans.transform.rotation.x, 
                        trans.transform.rotation.y, 
                        trans.transform.rotation.z, 
                        trans.transform.rotation.w]
            
            rot_matrix = R.from_quat(rot_quat).as_matrix()
            aruco_position_camera = np.array([tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]])
            aruco_position_base = np.dot(rot_matrix, aruco_position_camera) + trans_vector

            return aruco_position_base
        except Exception as e:
            self.get_logger().error(f"Failed to transform ArUco coordinates: {e}")
            return [0.0, 0.0, 0.0]

    def transform_aruco_orientation_to_base(self, rvec):
        """Transform ArUco orientation (rvec) to base frame."""
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_color_optical_frame', rclpy.time.Time())
            rot_quat = [trans.transform.rotation.x, 
                        trans.transform.rotation.y, 
                        trans.transform.rotation.z, 
                        trans.transform.rotation.w]
            
            rot_matrix = R.from_quat(rot_quat).as_matrix()
            aruco_rot_matrix = R.from_rotvec(rvec[0][0]).as_matrix()
            aruco_rot_base = np.dot(rot_matrix, aruco_rot_matrix)

            return R.from_matrix(aruco_rot_base).as_quat()  # Return as quaternion
        except Exception as e:
            self.get_logger().error(f"Failed to transform ArUco orientation: {e}")
            return [0.0, 0.0, 0.0, 1.0]

    def check_singularity(self):
        """Check for singularity based on joint positions and Jacobian condition number."""
        J = np.eye(6)  # Placeholder for actual Jacobian

        # Calculate condition number of the Jacobian matrix
        condition_number = np.linalg.cond(J)
        self.get_logger().info(f"Jacobian condition number: {condition_number}")
        
        if condition_number > self.singularity_threshold:
            self.damping_factor_base = min(self.max_damping_factor, condition_number / self.singularity_threshold)
            return True
        return False

    def move_ur5_to_aruco(self, aruco_position_base, aruco_orientation_base):
        """Move UR5 robot to the detected ArUco marker using a twist command."""
        distance_x = aruco_position_base[0]
        distance_y = aruco_position_base[1]
        distance_z = aruco_position_base[2]

        # Check if the position is within tolerance
        if abs(distance_x) < self.position_tolerance and abs(distance_y) < self.position_tolerance and abs(distance_z) < self.position_tolerance:
            self.get_logger().info("Aruco marker is within tolerance. No need to move.")
            return

        twist_msg = TwistStamped()
        twist_msg.twist.linear.x = self.linear_gain * distance_x
        twist_msg.twist.linear.y = self.linear_gain * distance_y
        twist_msg.twist.linear.z = self.linear_gain * distance_z

        current_orientation = R.from_quat(aruco_orientation_base)
        desired_orientation = R.from_quat([0.0, 0.0, 0.0, 1.0])  # Assume a desired orientation

        angular_error = desired_orientation.inv() * current_orientation
        angular_error_quat = angular_error.as_quat()

        twist_msg.twist.angular.x = self.angular_gain * angular_error_quat[0]
        twist_msg.twist.angular.y = self.angular_gain * angular_error_quat[1]
        twist_msg.twist.angular.z = self.angular_gain * angular_error_quat[2]

        self.get_logger().info(f"Publishing twist: linear = ({twist_msg.twist.linear.x}, {twist_msg.twist.linear.y}, {twist_msg.twist.linear.z}), "
                               f"angular = ({twist_msg.twist.angular.x}, {twist_msg.twist.angular.y}, {twist_msg.twist.angular.z})")
        self.twist_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoServoControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
