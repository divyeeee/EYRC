#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
import tf_transformations


def detect_aruco(image):
    aruco_area_threshold = 1500
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], 
                        [0.0, 931.1829833984375, 360.0], 
                        [0.0, 0.0, 1.0]])
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    size_of_aruco_m = 0.15

    center_aruco_list, ids, tvecs, quaternions = [], [], [], []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    corners, aruco_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if aruco_ids is not None:
        image = cv2.aruco.drawDetectedMarkers(image, corners, aruco_ids)
    else:
        return center_aruco_list, ids, tvecs, quaternions

    for i, aruco_id in enumerate(aruco_ids):
        area = cv2.contourArea(corners[i])
        if area < aruco_area_threshold:
            continue

        # Calculate center of the marker
        cX = (corners[i][0][0][0] + corners[i][0][2][0]) / 2
        cY = (corners[i][0][0][1] + corners[i][0][2][1]) / 2
        center_aruco_list.append((cX, cY))

        # Estimate pose of the marker
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_of_aruco_m, cam_mat, dist_mat)

        if rvec is not None and rvec.shape == (1, 1, 3):
            try:
                # Convert rvec to a 3x3 rotation matrix
                rot_matrix, _ = cv2.Rodrigues(rvec)

                if rot_matrix.shape == (3, 3):
                    # Create a 4x4 matrix and embed the 3x3 rotation matrix
                    rot_matrix_4x4 = np.eye(4)
                    rot_matrix_4x4[:3, :3] = rot_matrix

                    # Convert to quaternion
                    quaternion = tf_transformations.quaternion_from_matrix(rot_matrix_4x4)

                    # Store the detected values
                    tvecs.append(tvec)
                    quaternions.append(quaternion)
                    ids.append(aruco_id[0])
                else:
                    print(f"Unexpected rot_matrix shape: {rot_matrix.shape}")
            except cv2.error as e:
                print(f"Error in Rodrigues transformation: {e}")
        else:
            print(f"Invalid rvec shape: {rvec.shape}")

    return center_aruco_list, ids, tvecs, quaternions


class ArucoTwistPublisher(Node):
    def __init__(self):
        super().__init__('aruco_twist_publisher')

        # Create a client for the /servo_node/start_servo service
        self.start_servo_client = self.create_client(Trigger, '/servo_node/start_servo')

        # Wait for the service to be available
        while not self.start_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /servo_node/start_servo service...')

        # Create the service request
        start_servo_request = Trigger.Request()

        # Call the service to start the servo
        self.future = self.start_servo_client.call_async(start_servo_request)
        rclpy.spin_until_future_complete(self, self.future)

        if self.future.result() is not None:
            self.get_logger().info('Servo started successfully!')
        else:
            self.get_logger().error('Failed to start servo.')
            return

        # Create a publisher for the /servo_node/delta_twist_cmds topic
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # Set the publishing rate to 60 Hz
        self.timer = self.create_timer(1 / 60, self.publish_twist)

        # Create subscriptions for color and depth images
        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10)
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # Display timer for OpenCV window (every 100ms)
        self.display_timer = self.create_timer(0.1, self.display_image)

        # Initialize state variables
        self.ids = []
        self.tvecs = []
        self.quaternions = []

        # Variables to store coordinates and orientation
        self.x_coord = None
        self.y_coord = None
        self.z_coord = None

        # Target position and smooth path
        self.target_position = None
        self.smooth_path = []
        self.current_position_index = 0

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def depthimagecb(self, data):
        try:
            # Ensure proper depth image conversion
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f'Depth CvBridgeError: {e}')

    def detect_and_store_aruco(self):
        if self.cv_image is None:
            return

        # Detect the ArUco markers in the image
        center_aruco_list, ids, tvecs, quaternions = detect_aruco(self.cv_image)

        # Update state variables
        self.ids = ids
        self.tvecs = tvecs
        self.quaternions = quaternions

        if ids and len(center_aruco_list) > 0:
            # Get the center coordinates of the first detected marker
            cX, cY = center_aruco_list[0]
            
            # Convert to integer coordinates for depth image indexing
            cX_int = int(cX)
            cY_int = int(cY)

            # Get translation vector components
            self.x_coord, self.y_coord, _ = tvecs[0][0][0]

            # Safely get depth value
            if self.depth_image is not None:
                # Ensure coordinates are within image bounds
                height, width = self.depth_image.shape
                if 0 <= cY_int < height and 0 <= cX_int < width:
                    # Get depth value and handle potential invalid values
                    depth_value = self.depth_image[cY_int, cX_int]
                    if np.isfinite(depth_value) and depth_value > 0:
                        self.z_coord = float(depth_value)
                    else:
                        # If the depth value is invalid, try averaging the surrounding pixels
                        window_size = 5
                        y_start = max(0, cY_int - window_size//2)
                        y_end = min(height, cY_int + window_size//2)
                        x_start = max(0, cX_int - window_size//2)
                        x_end = min(width, cX_int + window_size//2)
                        
                        depth_window = self.depth_image[y_start:y_end, x_start:x_end]
                        valid_depths = depth_window[np.isfinite(depth_window) & (depth_window > 0)]
                        
                        if len(valid_depths) > 0:
                            self.z_coord = float(np.mean(valid_depths))
                        else:
                            self.z_coord = 0.0
                else:
                    self.z_coord = 0.0
            else:
                self.z_coord = 0.0

            self.get_logger().info(
                f"Detected ArUco at x={self.x_coord:.3f}, y={self.y_coord:.3f}, z={self.z_coord:.3f}"
            )

            # Set target position for smooth motion
            self.target_position = self.transform_to_robot_frame(
                self.x_coord, self.y_coord, self.z_coord
            )
            self.smooth_path = self.get_smooth_path()
    def transform_to_robot_frame(self, x, y, z):
        # Transform the detected ArUco position from the camera frame to the robot frame
        camera_to_robot_transform = np.array([[1, 0, 0, 0],
                                              [0, -1, 0, 0],
                                              [0, 0, -1, 0],
                                              [0, 0, 0, 1]])

        camera_position = np.array([x, y, z, 1]).reshape(4, 1)
        robot_position = np.dot(camera_to_robot_transform, camera_position)

        return robot_position[0:3].flatten()

    def get_smooth_path(self):
        if self.target_position is None:
            return []

        current_position = (0, 0, 0)  # Could be updated based on the arm's current state
        target_position = self.target_position
        points_count = 50

        smooth_path = []
        for i in range(points_count + 1):
            alpha = i / points_count
            x = current_position[0] * (1 - alpha) + target_position[0] * alpha
            y = current_position[1] * (1 - alpha) + target_position[1] * alpha
            z = current_position[2] * (1 - alpha) + target_position[2] * alpha
            smooth_path.append((x, y, z))

        return smooth_path

    def send_coordinates_to_servo(self):
        if self.current_position_index >= len(self.smooth_path):
            return

        # Get the next target position
        target_position = self.smooth_path[self.current_position_index]

        # Create a TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()

        k_trans = 0.5  # Proportional gain for translation

        # Calculate differences
        dx = target_position[0]
        dy = target_position[1]
        dz = target_position[2]

        # Set the twist command based on the difference
        twist_msg.twist.linear.x = k_trans * dx
        twist_msg.twist.linear.y = k_trans * dy
        twist_msg.twist.linear.z = k_trans * dz

        self.twist_pub.publish(twist_msg)
        self.current_position_index += 1

    def publish_twist(self):
        if self.cv_image is None:
            return

        # Detect ArUco marker and store its coordinates
        self.detect_and_store_aruco()

        # Send updated twist commands to the servo
        self.send_coordinates_to_servo()

    def display_image(self):
        if self.cv_image is not None:
            cv2.imshow("Camera Feed", self.cv_image)
            # Add waitKey to ensure window stays open
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    aruco_twist_publisher = ArucoTwistPublisher()
    rclpy.spin(aruco_twist_publisher)

    aruco_twist_publisher.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # Clean up OpenCV windows


if __name__ == '__main__':
    main()