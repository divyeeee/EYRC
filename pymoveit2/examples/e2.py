#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TwistStamped
from scipy.spatial.transform import Rotation as R
import cv2
import cv2.aruco as aruco
import tf2_ros
from std_srvs.srv import Trigger
from math import sqrt

class ArucoServoControl(Node):
    def __init__(self):
        super().__init__('aruco_servo_control')

        # Camera subscribers
        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimage_callback, 10)

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.jointstate_callback, 10)
        self.joint_positions = [0.0] * 6  # Initialize with zeros for 6 joint positions

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

        # Max speed limits (initialize)
        self.max_linear_speed = 0.1
        self.max_angular_speed = 0.1

        # Start the servo node
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
        self.joint_positions = list(msg.position)  # Update with actual joint states

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

        return corners, aruco_ids, tvecs

    def process_image(self):
        if self.cv_image is None:
            self.get_logger().warn("No image received yet.")
            return

        corners, aruco_ids, tvecs = self.detect_aruco(self.cv_image)
        self.aruco_positions.clear()

        if aruco_ids is not None:
            for tvec, aruco_id in zip(tvecs, aruco_ids):
                aruco_position_base = self.transform_aruco_to_base(tvec)
                self.aruco_positions.append((aruco_id, aruco_position_base))

                self.get_logger().info(f'Detected ArUco {aruco_id}: (x, y, z) = {aruco_position_base}')

                # Apply singularity check before moving
                if self.check_singularity():
                    self.get_logger().warn("Singularity detected! Scaling velocity.")
                else:
                    self.move_ur5_to_aruco(aruco_position_base)

                cv2.aruco.drawDetectedMarkers(self.cv_image, corners, aruco_ids)

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

    def check_singularity(self):
        """Check for singularity based on joint positions and Jacobian condition number."""
        # Placeholder for actual Jacobian (should be retrieved from robot model or MoveIt!)
        J = np.eye(6)  # Replace with actual Jacobian retrieval

        # Calculate condition number of the Jacobian
        condition_number = np.linalg.cond(J)
        self.get_logger().info(f"Jacobian condition number: {condition_number}")

        if condition_number > self.singularity_threshold:
            # Adjust damping factor based on proximity to singularity
            self.damping_factor = min(self.damping_factor_base * condition_number / self.singularity_threshold, self.max_damping_factor)
            return True

        return False

    def move_ur5_to_aruco(self, aruco_position_base):
        # Define the target position and orientation based on the ArUco marker
        target_position = {
            "x": aruco_position_base[0],
            "y": aruco_position_base[1],
            "z": aruco_position_base[2],
            "roll": 0.0,  # Replace with actual roll if needed
            "pitch": 0.0,  # Replace with actual pitch if needed
            "yaw": 0.0    # Replace with actual yaw if needed
        }

        # Current position can be updated using actual feedback from the robot
        current_position = {
            "x": self.joint_positions[0],  # Replace with actual feedback data
            "y": self.joint_positions[1],  # Replace with actual feedback data
            "z": self.joint_positions[2],  # Replace with actual feedback data
            "roll": 0.0,  # Replace with actual roll if available
            "pitch": 0.0,  # Replace with actual pitch if available
            "yaw": 0.0    # Replace with actual yaw if available
        }

        # Error in position (Euclidean distance)
        error_x = target_position["x"] - current_position["x"]
        error_y = target_position["y"] - current_position["y"]
        error_z = target_position["z"] - current_position["z"]
        position_error = sqrt(error_x**2 + error_y**2 + error_z**2)

        # Check if the robot is within tolerance
        if position_error <= self.position_tolerance:
            self.get_logger().info("Target reached.")
            return

        # Proportional control for velocity scaling
        linear_velocity = TwistStamped()
        linear_velocity.twist.linear.x = self.linear_gain * error_x
        linear_velocity.twist.linear.y = self.linear_gain * error_y
        linear_velocity.twist.linear.z = self.linear_gain * error_z

        # Limit speed to avoid excessive movement
        linear_velocity.twist.linear.x = max(-self.max_linear_speed, min(self.max_linear_speed, linear_velocity.twist.linear.x))
        linear_velocity.twist.linear.y = max(-self.max_linear_speed, min(self.max_linear_speed, linear_velocity.twist.linear.y))
        linear_velocity.twist.linear.z = max(-self.max_linear_speed, min(self.max_linear_speed, linear_velocity.twist.linear.z))

        self.twist_pub.publish(linear_velocity)

def main(args=None):
    rclpy.init(args=args)
    aruco_servo_control = ArucoServoControl()
    rclpy.spin(aruco_servo_control)
    aruco_servo_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
