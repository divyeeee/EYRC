#!/usr/bin/env python3
'''
# Team ID:          1118
# Theme:            Logistic coBot
# Author List:      Saeesh , Sambhav, Anshul , Robin
# Filename:         joint_jog_service.py
# Functions:        start_servo, start_servo_callback, joint_state_callback, 
#                   attach_gripper, detach_gripper, execute_callback, 
#                   perform_joint_jog_task, get_wrist_pose, move_wrist_to_pose, 
#                   move_wrist_to_pose2, go_to_state, is_within_tolerance, main
# Global variables: None
'''

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import TransformListener, Buffer
import numpy as np
from linkattacher_msgs.srv import AttachLink, DetachLink
import threading

class JointJogService(Node):
    '''
    Purpose:
    ---
    This class implements a ROS2 node to perform joint jog commands and
    wrist manipulations for a UR5 robot. It handles publishing commands,
    starting services, and moving to specific poses.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None
    '''
    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the JointJogService node, sets up various publishers, subscriptions, 
        service clients, and the transform listener. This function also waits for the 
        start servo service to be available and initializes joint states.

        Returns:
        ---
        None

        Example call:
        ---
        joint_jog_service = JointJogService()
        '''
        super().__init__('joint_jog_service_3')

        # Callback groups to separate execution
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()

        # Publishers and subscriptions
        self.joint_jog_pub = self.create_publisher(JointJog, '/servo_node/delta_joint_cmds', 10)
        self.twist_jog_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Service clients
        self.start_servo_client = self.create_client(Trigger, '/servo_node/start_servo')
        self.attach_client = self.create_client(AttachLink, '/GripperMagnetON')
        self.detach_client = self.create_client(DetachLink, '/GripperMagnetOFF')

        # Transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service to trigger the action
        self.service = self.create_service(
            Trigger, 
            'execute_third_joint_jog', 
            self.execute_callback, 
            callback_group=self.service_callback_group
        )

        # Wait for the start servo service to be available
        while not self.start_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /servo_node/start_servo service...')
        self.desired_joint_angles_deg = [0, -111, 143, 265, -80]
        self.desired_joint_angles_rad = [np.deg2rad(angle) for angle in self.desired_joint_angles_deg]
        self.kp_joint = 10
        self.current_joint_positions = [0.0] * 5
        self.start_servo()

    def start_servo(self):
        while not self.start_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /servo_node/start_servo service...')

        request = Trigger.Request()
        future = self.start_servo_client.call_async(request)
        future.add_done_callback(self.start_servo_callback)

    def start_servo_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Servo started successfully.')
            else:
                self.get_logger().warning('Failed to start servo: ' + response.message)
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def joint_state_callback(self, msg):
        self.current_joint_positions = msg.position

    def attach_gripper(self, model1_name):
        # Create client for the AttachLink service to attach the object
        gripper_control = self.create_client(AttachLink, '/GripperMagnetON')

        # Wait for the service to be available
        while not gripper_control.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gripper attach service not available, waiting again...')

        # Create the request for attaching the gripper
        req = AttachLink.Request()
        req.model1_name = model1_name  # Specify the box/object name
        req.link1_name = 'link'        # Box link name
        req.model2_name = 'ur5'        # UR5 model name
        req.link2_name = 'wrist_3_link'  # UR5 end effector link

        # Call the service asynchronously
        future = gripper_control.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Attached {model1_name} to UR5 gripper.")

    def detach_gripper(self, model1_name):
        # Create client for the DetachLink service to detach the object
        gripper_control = self.create_client(DetachLink, '/GripperMagnetOFF')

        # Wait for the service to be available
        while not gripper_control.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gripper detach service not available, waiting again...')

        # Create the request for detaching the gripper
        req = DetachLink.Request()
        req.model1_name = model1_name  # Specify the box/object name
        req.link1_name = 'link'        # Box link name
        req.model2_name = 'ur5'        # UR5 model name
        req.link2_name = 'wrist_3_link'  # UR5 end effector link

        # Call the service asynchronously
        future = gripper_control.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Detached {model1_name} from UR5 gripper.")

    def execute_callback(self, request, response):
        self.get_logger().info('Executing joint jog operation 3.')

        # Create a thread for the service task
        task_thread = threading.Thread(target=self.perform_joint_jog_task)
        task_thread.start()

        # Provide an immediate response to avoid blocking the service
        response.success = True
        response.message = 'Joint jog operation started successfully.'
        return response

    def perform_joint_jog_task(self):
        try:
            # Step 1: Move the wrist to top position
            target_pose2 = [0.163, 0.093, 0.535]
            self.move_wrist_to_pose2(target_pose2)
            
            target_pose1 = [0.178, 0.0500, 0.417]
            self.move_wrist_to_pose(target_pose1)

            # Step 3: Move the wrist to the second target pose
            target_pose2 = [0.046590, -0.424872, 0.229360]
            self.move_wrist_to_pose2(target_pose2)

            # Step 4: Attach the gripper to the object
            transform = self.tf_buffer.lookup_transform('base_link', 'obj_3', rclpy.time.Time())
            target_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            self.move_wrist_to_pose(target_pose)
            self.attach_gripper('box3')

            # Step 5: Move the wrist to the drop location and detach the gripper
            target_pose2 = [0.163, 0.093, 0.535]
            self.move_wrist_to_pose2(target_pose2)
            target_pose2 = [0.5034, 0.0211, 0.0985]
            self.move_wrist_to_pose2(target_pose2)
            

            self.get_logger().info('Completed Manupulation Task.')
        except Exception as e:
            self.get_logger().error(f'Error during Manupulation Task: {e}')

    def get_wrist_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            position = transform.transform.translation
            rotation = transform.transform.rotation
            self.get_logger().info(f'Wrist pose - Position: ({position.x}, {position.y}, {position.z}), '
                                   f'Orientation: ({rotation.x}, {rotation.y}, {rotation.z}, {rotation.w})')
            return position, rotation
        except Exception as e:
            self.get_logger().error(f'Failed to get wrist_3_link pose: {e}')
            return None, None

    def move_wrist_to_pose(self, target_pose):
        kp_linear = 45 # Linear proportional gain
        tolerance = 0.09  # Tolerance in meters

        while rclpy.ok():
            position, _ = self.get_wrist_pose()
            if not position:
                break

            error_x = target_pose[0] - position.x
            error_y = target_pose[1] - position.y
            error_z = target_pose[2] - position.z
            distance_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)

            if distance_error <= tolerance:
                self.get_logger().info('Target pose reached.')
                break

            msg = TwistStamped()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            msg.twist.linear.x = kp_linear * error_x
            msg.twist.linear.y = kp_linear * error_y
            msg.twist.linear.z = kp_linear * error_z

            self.twist_jog_pub.publish(msg)
            self.get_logger().info(f'Moving wrist - Linear velocities: ({msg.twist.linear.x}, '
                                   f'{msg.twist.linear.y}, {msg.twist.linear.z})')
            rclpy.spin_once(self, timeout_sec=0.1)

    def move_wrist_to_pose2(self, target_pose2):
        kp_linear = 45# Linear proportional gain
        tolerance = 0.08  # Tolerance in meters

        while rclpy.ok():
            position, _ = self.get_wrist_pose()
            if not position:
                break

            error_x = target_pose2[0] - position.x
            error_y = target_pose2[1] - position.y
            error_z = target_pose2[2] - position.z
            distance_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)

            if distance_error <= tolerance:
                self.get_logger().info('Target pose reached.')
                break

            msg = TwistStamped()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            msg.twist.linear.x = kp_linear * error_x
            msg.twist.linear.y = kp_linear * error_y
            msg.twist.linear.z = kp_linear * error_z

            self.twist_jog_pub.publish(msg)
            self.get_logger().info(f'Moving wrist - Linear velocities: ({msg.twist.linear.x}, '
                                   f'{msg.twist.linear.y}, {msg.twist.linear.z})')
            rclpy.spin_once(self, timeout_sec=0.1)

    def go_to_state(self, target_joint_angles_deg):
        target_joint_angles_rad = [np.deg2rad(angle) for angle in target_joint_angles_deg]
        tolerance_deg = 10

        while rclpy.ok():
            if self.is_within_tolerance(target_joint_angles_rad, tolerance_deg):
                self.get_logger().info('Desired joint positions reached.')
                break

            msg = JointJog()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_frame"
            msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                               'wrist_1_joint', 'wrist_2_joint']
            msg.displacements = []
            msg.velocities = []

            for target, current in zip(target_joint_angles_rad, self.current_joint_positions):
                error = (target - current + np.pi) % (2 * np.pi) - np.pi
                error_deg = np.rad2deg(error)
                self.get_logger().info(f'Error for joint: {error_deg} degrees')

                if abs(error_deg) > tolerance_deg or (error_deg<-170 and error_deg>-180):
                    displacement = error * self.kp_joint
                    msg.displacements.append(displacement)
                    velocity = 0.8 + (abs(error_deg) / 180.0) * 20.0
                    msg.velocities.append(velocity)
                else:
                    msg.velocities.append(0.0)

            msg.duration = 0.1
            self.joint_jog_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

    def is_within_tolerance(self, target_joint_angles_rad, tolerance_deg):
        for target, current in zip(target_joint_angles_rad, self.current_joint_positions):
            error = (target - current + np.pi) % (2 * np.pi) - np.pi
            error_deg = np.rad2deg(error)
            if abs(error_deg) > tolerance_deg:
                return False
        return True


def main(args=None):
    rclpy.init(args=args)

    # Use a MultiThreadedExecutor to handle multiple threads
    executor = MultiThreadedExecutor()
    node = JointJogService()

    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Joint Jog Service.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()