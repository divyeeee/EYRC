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
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    size_of_aruco_m = 0.15

    center_aruco_list, ids, tvecs, quaternions = [], [], [], []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, aruco_ids, _ = detector.detectMarkers(gray)

    if aruco_ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, aruco_ids)
    else:
        return center_aruco_list, ids, tvecs, quaternions

    for i, aruco_id in enumerate(aruco_ids):
        area = cv2.contourArea(corners[i])
        if area < aruco_area_threshold:
            continue

        cX = ((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
        cY = ((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
        center_aruco_list.append((cX, cY))

        # Estimate pose for each marker
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_of_aruco_m, cam_mat, dist_mat)
        if rvec is not None and rvec.shape == (1, 1, 3):
            tvecs.append(tvec[0][0])  # Store the translation vector directly
            
            # Convert rotation vector to a full 4x4 transformation matrix
            rot_matrix = cv2.Rodrigues(rvec)[0]
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix

            # Extract quaternion from transformation matrix
            quaternion = tf_transformations.quaternion_from_matrix(transform_matrix)
            quaternions.append(quaternion)
            ids.append(aruco_id[0])

            # Draw coordinate axes using lines
            axis_points = np.array([[0, 0, 0], 
                                  [0.1, 0, 0], 
                                  [0, 0.1, 0], 
                                  [0, 0, 0.1]], dtype=np.float32)
            
            projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, cam_mat, dist_mat)
            
            # Draw the axes
            origin = tuple(map(int, projected_points[0].ravel()))
            point_x = tuple(map(int, projected_points[1].ravel()))
            point_y = tuple(map(int, projected_points[2].ravel()))
            point_z = tuple(map(int, projected_points[3].ravel()))
            
            # X axis - Red
            cv2.line(image, origin, point_x, (0, 0, 255), 2)
            # Y axis - Green
            cv2.line(image, origin, point_y, (0, 255, 0), 2)
            # Z axis - Blue
            cv2.line(image, origin, point_z, (255, 0, 0), 2)

            # Extract position values
            x = float(tvec[0][0][0])
            y = float(tvec[0][0][1])
            z = float(tvec[0][0][2])

            # Add text to show marker ID and coordinates
            text_pos = (int(cX), int(cY) - 10)
            position_text = f'ID: {aruco_id[0]} ({x:.2f}, {y:.2f}, {z:.2f}m)'
            cv2.putText(image, position_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log the position for debugging
            print(f"Marker {aruco_id[0]} position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    return center_aruco_list, ids, tvecs, quaternions
class ArucoTwistPublisher(Node):
    def __init__(self):
        super().__init__('aruco_twist_publisher')

        # Initialize services and subscribers as before
        self.start_servo_client = self.create_client(Trigger, '/servo_node/start_servo')
        while not self.start_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /servo_node/start_servo service...')

        start_servo_request = Trigger.Request()
        self.future = self.start_servo_client.call_async(start_servo_request)
        rclpy.spin_until_future_complete(self, self.future)

        if self.future.result() is not None:
            self.get_logger().info('Servo started successfully!')
        else:
            self.get_logger().error('Failed to start servo.')
            return

        # Publishers and subscribers
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10)
        
        # Initialize variables
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.marker_positions = []
        self.current_target = None
        self.target_reached = False
        self.last_control_time = None
        self.movement_timeout = 5.0
        self.visited_markers = set()
        
        # Modified control parameters
        self.Kp = 0.3  # Position control gain
        self.Ko = 0.5  # Orientation control gain
        self.distance_threshold = 0.02
        self.orientation_threshold = 0.1
        self.min_velocity = 0.01
        self.max_velocity = 0.3
        self.max_angular_velocity = 0.5
        
        # Current end effector orientation
        self.current_orientation = None
        
        # Timers
        self.control_timer = self.create_timer(1/60, self.control_loop)
        self.display_timer = self.create_timer(0.1, self.display_image)
    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def calculate_distance(self, position):
        return np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)

    def find_next_target(self):
        """Find the next target based on z-height priority"""
        if not self.marker_positions:
            return None

        # Filter out visited markers
        unvisited_markers = [pos for i, pos in enumerate(self.marker_positions) 
                           if tuple(pos) not in self.visited_markers]
        
        if not unvisited_markers:
            self.visited_markers.clear()  # Reset visited markers if all have been visited
            return None

        # Sort markers by z-coordinate (height) in descending order
        sorted_markers = sorted(unvisited_markers, key=lambda x: x[2], reverse=True)
        return sorted_markers[0]

    def transform_to_robot_frame(self, tvec):
        camera_to_robot_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        position = np.append(tvec, 1)
        transformed = np.dot(camera_to_robot_transform, position)
        return transformed[:3]

    def check_movement_timeout(self):
        """Check if we've been trying to reach the target for too long"""
        if self.last_control_time is None:
            self.last_control_time = self.get_clock().now()
            return False

        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_control_time).nanoseconds / 1e9

        if time_diff > self.movement_timeout:
            self.get_logger().warn('Movement timeout reached. Selecting new target.')
            return True
        return False

    def calculate_target_orientation(self, target_position):
        """Calculate desired end effector orientation to face the target"""
        # Calculate direction vector to target
        direction = target_position / np.linalg.norm(target_position)
        
        # Calculate the rotation that aligns the end effector with the target
        # We want the end effector to point towards the target
        z_axis = -direction  # End effector points towards target
        
        # Choose an up vector (y-axis)
        world_up = np.array([0, 0, 1])
        x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Complete the right-handed coordinate system
        y_axis = np.cross(z_axis, x_axis)
        
        # Create rotation matrix
        rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T
        
        # Convert to quaternion
        quaternion = tf_transformations.quaternion_from_matrix(
            np.vstack((np.hstack((rotation_matrix, np.zeros((3, 1)))),
                      [0, 0, 0, 1]))
        )
        
        return quaternion

    def orientation_error(self, current_quat, target_quat):
        """Calculate the orientation error between current and target quaternions"""
        # Convert quaternions to rotation matrices
        current_mat = tf_transformations.quaternion_matrix(current_quat)[:3, :3]
        target_mat = tf_transformations.quaternion_matrix(target_quat)[:3, :3]
        
        # Calculate the error rotation matrix
        error_mat = np.dot(target_mat, current_mat.T)
        
        # Convert to axis-angle representation
        angle = np.arccos((np.trace(error_mat) - 1) / 2)
        if angle < 1e-10:
            return np.zeros(3)
        
        axis = np.array([error_mat[2, 1] - error_mat[1, 2],
                        error_mat[0, 2] - error_mat[2, 0],
                        error_mat[1, 0] - error_mat[0, 1]])
        
        axis = axis / (2 * np.sin(angle))
        
        return axis * angle

    def find_next_target(self):
        """Find the next target based on distance priority"""
        if not self.marker_positions:
            return None, None

        # Filter out visited markers
        unvisited_markers = [(i, pos) for i, pos in enumerate(self.marker_positions) 
                           if tuple(pos) not in self.visited_markers]
        
        if not unvisited_markers:
            self.visited_markers.clear()
            return None, None

        # First, separate markers into near and far groups based on XY distance
        near_markers = []
        far_markers = []
        
        for i, pos in unvisited_markers:
            xy_distance = np.sqrt(pos[0]**2 + pos[1]**2)  # Distance in XY plane
            if xy_distance < 0.5:  # Threshold for near/far classification
                near_markers.append((i, pos))
            else:
                far_markers.append((i, pos))

        # If we have near markers, sort them by total distance
        if near_markers:
            sorted_markers = sorted(near_markers, key=lambda x: self.calculate_distance(x[1]))
            selected_position = sorted_markers[0][1]
        # If no near markers, sort far markers by total distance
        elif far_markers:
            sorted_markers = sorted(far_markers, key=lambda x: self.calculate_distance(x[1]))
            selected_position = sorted_markers[0][1]
        else:
            return None, None
            
        # Calculate desired orientation for the selected position
        desired_orientation = self.calculate_target_orientation(selected_position)
        
        return selected_position, desired_orientation

    def control_loop(self):
        if self.cv_image is None:
            return

        # Detect ArUco markers
        _, ids, tvecs, quaternions = detect_aruco(self.cv_image)
        
        # Update marker positions
        self.marker_positions = []
        if tvecs:
            for tvec in tvecs:
                robot_frame_position = self.transform_to_robot_frame(tvec)
                self.marker_positions.append(robot_frame_position)

        # Target selection and orientation calculation
        if self.current_target is None or self.target_reached or self.check_movement_timeout():
            target_position, target_orientation = self.find_next_target()
            self.current_target = target_position
            self.target_orientation = target_orientation
            self.target_reached = False
            self.last_control_time = self.get_clock().now()
            
            if self.current_target is not None:
                self.get_logger().info(f'New target selected at position: {self.current_target}')

        # Control loop
        if self.current_target is not None and self.target_orientation is not None:
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()

            # Position control
            position_error = self.current_target
            distance = np.linalg.norm(position_error)

            # Orientation control
            if self.current_orientation is not None:
                angular_error = self.orientation_error(self.current_orientation, self.target_orientation)
                orientation_error_magnitude = np.linalg.norm(angular_error)
            else:
                angular_error = np.zeros(3)
                orientation_error_magnitude = 0.0

            if distance < self.distance_threshold and orientation_error_magnitude < self.orientation_threshold:
                self.target_reached = True
                self.visited_markers.add(tuple(self.current_target))
                self.current_target = None
                # Stop all motion
                twist_msg.twist.linear.x = 0.0
                twist_msg.twist.linear.y = 0.0
                twist_msg.twist.linear.z = 0.0
                twist_msg.twist.angular.x = 0.0
                twist_msg.twist.angular.y = 0.0
                twist_msg.twist.angular.z = 0.0
            else:
                # Position control
                velocities = [self.Kp * e for e in position_error]
                velocities = [np.clip(v, -self.max_velocity, self.max_velocity) for v in velocities]
                velocities = [v if abs(v) > self.min_velocity else 0.0 for v in velocities]
                
                # Angular velocity control
                angular_velocities = [self.Ko * e for e in angular_error]
                angular_velocities = [np.clip(v, -self.max_angular_velocity, self.max_angular_velocity) 
                                    for v in angular_velocities]
                
                # Apply controls
                twist_msg.twist.linear.x = velocities[0]
                twist_msg.twist.linear.y = velocities[1]
                twist_msg.twist.linear.z = velocities[2]
                
                twist_msg.twist.angular.x = angular_velocities[0]
                twist_msg.twist.angular.y = angular_velocities[1]
                twist_msg.twist.angular.z = angular_velocities[2]

            # Publish twist command
            self.twist_pub.publish(twist_msg)

    def display_image(self):
        if self.cv_image is not None:
            display_image = self.cv_image.copy()
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, pos in enumerate(self.marker_positions):
                distance = self.calculate_distance(pos)
                xy_distance = np.sqrt(pos[0]**2 + pos[1]**2)
                visited = "✓" if tuple(pos) in self.visited_markers else " "
                text = f"Marker {i}{visited}: D_xy={xy_distance:.2f}m, D_total={distance:.2f}m"
                cv2.putText(display_image, text, (10, 30 + i*30), font, 0.7, (0, 255, 0), 2)

            if self.current_target is not None:
                text = "TARGET: ({:.2f}, {:.2f}, {:.2f})".format(*self.current_target)
                cv2.putText(display_image, text, (10, display_image.shape[0] - 20), font, 0.7, (0, 0, 255), 2)

            cv2.imshow('ArUco Detection', display_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    aruco_twist_publisher = ArucoTwistPublisher()
    rclpy.spin(aruco_twist_publisher)
    aruco_twist_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()