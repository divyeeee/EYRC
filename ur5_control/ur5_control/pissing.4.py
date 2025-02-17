#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import numpy as np
import tf2_ros
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header, Int32MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, SetBool

from payload_service.srv import PayloadSW, PicknPlace
from linkattacher_msgs.srv import AttachLink, DetachLink

import threading
import concurrent.futures
import queue

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        self._callback_handler = ReentrantCallbackGroup()
        
        self._joint_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        
        self._joint_callback_group = MutuallyExclusiveCallbackGroup()
        self._marker_callback_group = MutuallyExclusiveCallbackGroup()
        self._service_callback_group = ReentrantCallbackGroup()
        
        # Task queue for handling multiple requests
        self._task_queue = queue.Queue()
        self._is_processing = False
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        self.initialize_variables()
        
        self.setup_publishers_and_subscribers(qos_profile)
        self.setup_services()
        self.setup_transform_listener()
        
        self.initialize_servo()
        
        self._object_manipulation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._movement_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Start the task processing thread
        self._task_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self._task_thread.start()
        
        self.get_logger().info('UR5 Robot Controller Initialized')

    def initialize_variables(self):
        with self._joint_lock:
            self.should_it_start = False
            self.marker_data = None
            self.joint_positions = [0.0] * 6
            self.joint_gain = 0.1
        
        self.target_positions = {
            'top_pos': [0.43, 0.1, 0.46],
            'ebot_pos': [0.65, 0.01, -0.1],
            'init_top': [0.16, 0.11, 0.47],
            'second_box': [-0.007, -0.42, 0.23],
            'return_top': [0.16, 0.09, 0.53],
            'last_pos': [-0.11, 0.25, 0.25]
        }
        
        self.init_joint_config = [0, -137, 138, -82, -90, 180]
        self.target_deg=[39,-121,135,-100,-93]

    def setup_publishers_and_subscribers(self, qos_profile):
        self.twist_publisher = self.create_publisher(
            TwistStamped, 
            '/servo_node/delta_twist_cmds', 
            10
        )
        
        self.joint_publisher = self.create_publisher(
            JointJog, 
            '/servo_node/delta_joint_cmds', 
            10
        )
        
        self.marker_subscriber = self.create_subscription(
            Int32MultiArray, 
            '/detected_aruco_ids', 
            self.update_markers, 
            10, 
            callback_group=self._marker_callback_group
        )
        
        self.joint_subscriber = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.update_joints, 
            qos_profile,
            callback_group=self._joint_callback_group
        )

    def setup_services(self):
        self.servo_client = self.create_client(
            Trigger, 
            '/servo_node/start_servo'
        )
        #SetBool
        self.passing_service = self.create_service(
            PicknPlace, 
            'picknplace', 
            self.handle_passing_request, 
            callback_group=self._callback_handler
        )

    def setup_transform_listener(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def update_joints(self, msg):
        try:
            joint_names = [
                'shoulder_pan_joint', 
                'shoulder_lift_joint', 
                'elbow_joint', 
                'wrist_1_joint', 
                'wrist_2_joint', 
                'wrist_3_joint'
            ]
            
            joint_map = dict(zip(msg.name, msg.position))
            
            with self._joint_lock:
                self.joint_positions = [
                    joint_map.get(name, 0.0) 
                    for name in joint_names
                ]
        
        except Exception as e:
            self.get_logger().error(f'Error updating joint positions: {e}')

    def get_current_joint_positions(self):
        with self._joint_lock:
            return self.joint_positions.copy()

    def update_markers(self, msg):
        self.marker_data = msg.data
        self.get_logger().info(f'Detected Aruco markers: {self.marker_data}')

    def check_position_reached(self, target_angles_rad, tolerance_deg):
        for target, current in zip(target_angles_rad, self.joint_positions):
            error = (target - current + np.pi) % (2 * np.pi) - np.pi
            if abs(np.rad2deg(error)) > tolerance_deg:
                return False
        return True

    def move_joints(self, target_angles_deg):
        target_rad = [np.deg2rad(angle) for angle in target_angles_deg]
        tolerance_deg = 10
        
        while rclpy.ok():
            if self.check_position_reached(target_rad, tolerance_deg):
                self.get_logger().info('Position reached')
                break
            
            msg = JointJog()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_frame"
            msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint']
            msg.velocities = []
            msg.displacements = []
            
            n = 0

            for target, current in zip(target_rad, self.joint_positions):
                n += 1
                error = (target - current + np.pi) % (2 * np.pi) - np.pi
                error_deg = np.rad2deg(error)
                self.get_logger().info(f'Error for joint {n}: {error_deg} degrees, current pose:{current}, target pose:{target}')

                if abs(error_deg) > tolerance_deg or (error_deg < -170 and error_deg > -180):
                    displacement = error * self.joint_gain
                    msg.displacements.append(displacement)
                    velocity = 0.5 + (abs(error_deg) / 180.0) * 5.0
                    self.get_logger().info(f'Diff in Joint angle: {error_deg}, velocity is {velocity}')
                    msg.velocities.append(velocity)
                else:
                    msg.velocities.append(0.0)

            msg.duration = 0.1
            self.joint_publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

    def move_to_position(self, target):
        gain = 5
        tolerance = 0.08

        while rclpy.ok():
            pos,_ = self.get_current_position()
            if not pos:
                break
            
            error_x = target[0] - pos.x
            error_y = target[1] - pos.y
            error_z = target[2] - pos.z
            distance = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            if distance <= tolerance:
                self.get_logger().info('Position reached')
                break
            
            msg = TwistStamped()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x = gain * error_x
            msg.twist.linear.y = gain * error_y
            msg.twist.linear.z = gain * error_z

            self.twist_publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

    def get_current_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            pos = transform.transform.translation
            rot = transform.transform.rotation
            return pos, rot
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')
            return None, None

    def initialize_servo(self):
        try:
            if not self.servo_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('Servo service not available after 5 seconds')
                return False
            
            future = self.servo_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            response = future.result()
            if response.success:
                self.get_logger().info('Servo initialized successfully')
                return True
            else:
                self.get_logger().warning(f'Servo initialization failed: {response.message}')
                return False
        
        except Exception as e:
            self.get_logger().error(f'Servo initialization error: {e}')
            return False
        
    def attach_object(self, object_name):
        def attach_task():
            try:
                client = self.create_client(AttachLink, '/GripperMagnetON')
                if not client.wait_for_service(timeout_sec=5.0):
                    self.get_logger().error(f'Attach service for {object_name} not available')
                    return False

                request = AttachLink.Request()
                request.model1_name = object_name
                request.link1_name = 'link'
                request.model2_name = 'ur5'
                request.link2_name = 'wrist_3_link'

                future = client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                
                if future.result() and future.result().success:
                    self.get_logger().info(f'Successfully attached {object_name}')
                    return True
                else:
                    self.get_logger().warning(f'Failed to attach {object_name}')
                    return False
            
            except Exception as e:
                self.get_logger().error(f'Attach object error: {e}')
                return False

        future = self._object_manipulation_executor.submit(attach_task)
        
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            self.get_logger().error(f'Attach object {object_name} timed out')
            return False

    def detach_object(self, object_name):
        def detach_task():
            try:
                client = self.create_client(DetachLink, '/GripperMagnetOFF')
                if not client.wait_for_service(timeout_sec=5.0):
                    self.get_logger().error(f'Detach service for {object_name} not available')
                    return False

                request = DetachLink.Request()
                request.model1_name = object_name
                request.link1_name = 'link'
                request.model2_name = 'ur5'
                request.link2_name = 'wrist_3_link'
                
                future = client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                
                if future.result() and future.result().success:
                    self.get_logger().info(f'Successfully detached {object_name}')
                    return True
                else:
                    self.get_logger().warning(f'Failed to detach {object_name}')
                    return False
            
            except Exception as e:
                self.get_logger().error(f'Detach object error: {e}')
                return False

        future = self._object_manipulation_executor.submit(detach_task)
        
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            self.get_logger().error(f'Detach object {object_name} timed out')
            return False

    def _process_tasks(self):
        while rclpy.ok():
            try:
                task = self._task_queue.get()
                if task is None:
                    break
                    
                box_name = task
                with self._operation_lock:
                    self._is_processing = True
                    self.start_and_move_arm(box_name)
                    self._is_processing = False
                
                self._task_queue.task_done()
            except Exception as e:
                self.get_logger().error(f'Task processing error: {e}')
                with self._operation_lock:
                    self._is_processing = False

    def update_markers(self, msg):
        self.marker_data = msg.data
        self.get_logger().info(f'Detected Aruco markers: {self.marker_data}')

    def handle_passing_request(self, request, response):
        self.get_logger().info(f'Received payload service request for box: {request.box_name}')
        
        try:
            # Add the task to the queue instead of processing immediately
            self._task_queue.put(request.box_name)
            
            response.success = True
            response.message = f"Request for {request.box_name} queued successfully"
        except Exception as e:
            self.get_logger().error(f'Error queuing request: {e}')
            response.success = False
            response.message = f"Error: {str(e)}"
        
        return response

    def go_to_left(self, box_name):
        try:
            if self.marker_data is None:
                self.get_logger().warn('No markers detected yet')
                return

            self.move_joints(self.init_joint_config)

            movements = [
                ('init_top', "Moving to initial top position"),
                ('second_box', "Moving to second box position"),
            ]

            for pos_key, log_msg in movements:
                self.get_logger().info(log_msg)
                self.move_to_position(self.target_positions[pos_key])

            transform = self.tf_buffer.lookup_transform(
                'base_link', 
                f"obj_{box_name[-1]}", 
                rclpy.time.Time()
            )
            
            target_pose = [
                transform.transform.translation.x, 
                transform.transform.translation.y, 
                transform.transform.translation.z
            ]
            
            self.move_to_position(target_pose)
            self.attach_object(box_name)
            
            drop_positions = [
                ('return_top', "Returning to top position"),
                ('ebot_pos', "Moving to drop position")
            ]
            
            for pos_key, log_msg in drop_positions:
                self.get_logger().info(log_msg)
                self.move_to_position(self.target_positions[pos_key])
            
            self.detach_object(box_name)
            self.move_to_position(self.target_positions['init_top'])

        except Exception as e:
            self.get_logger().error(f'Arm movement error go_to_left: {e}')

    def go_to_right(self, box_name):
        try:
            if self.marker_data is None:
                self.get_logger().warn('No markers detected yet')
                return

            self.move_joints(self.target_deg)

            transform = self.tf_buffer.lookup_transform(
                'base_link', 
                f"obj_{box_name[-1]}", 
                rclpy.time.Time()
            )
            
            target_pose = [
                transform.transform.translation.x, 
                transform.transform.translation.y, 
                transform.transform.translation.z
            ]
            
            self.move_to_position(target_pose)
            self.attach_object(box_name)
            
            drop_positions = [
                ('top_pos', "Returning to top position"),
                ('ebot_pos', "Moving to drop position")
            ]
            
            for pos_key, log_msg in drop_positions:
                self.get_logger().info(log_msg)
                self.move_to_position(self.target_positions[pos_key])
            
            self.detach_object(box_name)
            self.move_to_position(self.target_positions['init_top'])

        except Exception as e:
            self.get_logger().error(f'Arm movement error go_to_right: {e}')
                    
    def start_and_move_arm(self, box_name):
        try:
            if self.marker_data is None:
                self.get_logger().warn('No markers detected yet')
                return

            transform = self.tf_buffer.lookup_transform(
                'base_link', 
                f"obj_{box_name[-1]}", 
                rclpy.time.Time()
            )
            
            target_pose = [
                transform.transform.translation.x, 
                transform.transform.translation.y, 
                transform.transform.translation.z
            ]
            
            if target_pose[1] < -2.3:
                self.go_to_right(box_name)
            else:
                self.go_to_left(box_name)

        except Exception as e:
            self.get_logger().error(f'Arm movement error: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RobotController()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard interrupt received, shutting down...')
        finally:
            # Clean shutdown
            node._task_queue.put(None)  # Signal the task thread to stop
            node._task_thread.join()    # Wait for task thread to finish
            node._object_manipulation_executor.shutdown()
            node._movement_executor.shutdown()
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
    
    except Exception as e:
        print(f"Initialization error: {e}")

if __name__ == '__main__':
    main()