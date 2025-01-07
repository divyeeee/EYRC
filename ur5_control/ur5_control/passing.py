'''
# Team ID:          1114
# Theme:            Logistic coBot
# Author List:      Anuj, Yashita, Chirayu, Divye
# Filename:         new_service.5.py
# Functions:        initialize_variablesm, setup_publishers_and_subscribers, 
#                   setup_services, setup_transform_listener,update_joints, 
#                   get_current_joint_positions, update_markers, check_position_reached, 
#                   move_joints, move_to_position, get_current_position, 
#                   initialize_servo, handle_passing_request, attach_object, 
#                   attach_task, detach_object, detach_task, start_and_move_arm
#                   update_markers, main.
# Global variables: None
'''

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

# -----------------------------
# Custom service imports
# -----------------------------
from payload_service.srv import PayloadSW
from linkattacher_msgs.srv import AttachLink, DetachLink

import threading
import concurrent.futures

class RobotController(Node):
    '''
    Purpose:
    ---
    This class implements a ROS2 node to perform robot control commands and
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
        Initializes the RobotController node, sets up various publishers, subscriptions, 
        service clients, and the transform listener. This function also waits for the 
        start servo service to be available and initializes joint states.

        Returns:
        ---
        None

        Example call:
        ---
        node = RobotController()
        '''
        # -----------------------------
        # Initialize ROS 2 Node
        # -----------------------------
        super().__init__('robot_control_node')
        self._callback_handler = ReentrantCallbackGroup()
        
        # -----------------------------
        # Thread-safe variable protection
        # -----------------------------
        self._joint_lock = threading.Lock()
        
        # -----------------------------
        # Create separate callback groups
        # -----------------------------
        self._joint_callback_group = MutuallyExclusiveCallbackGroup()
        self._marker_callback_group = MutuallyExclusiveCallbackGroup()
        self._service_callback_group = ReentrantCallbackGroup()
        
        # -----------------------------
        # Configure QoS for joint states
        # -----------------------------
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # -----------------------------
        # Initialize class variables with thread-safe initialization
        # -----------------------------
        self.initialize_variables()
        
        # -----------------------------
        # Setup communication interfaces with callback groups
        # -----------------------------
        self.setup_publishers_and_subscribers(qos_profile)
        self.setup_services()
        self.setup_transform_listener()
        
        # -----------------------------
        # Initialize robot systems
        # -----------------------------
        self.initialize_servo()
        
        # -----------------------------
        # Create a thread pool executor for object manipulation
        # -----------------------------
        self._object_manipulation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # -----------------------------
        # Setup motion control timer
        # -----------------------------

        # -----------------------------
        # self._motion_control = self.create_timer(0.5, self.start_and_move_arm)
        # -----------------------------
        
        self.get_logger().info('UR5 Robot Controller Initialized')

    def initialize_variables(self):
        
        '''
        Purpose:
        ---
        Initialize all class-level variables with default values.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.initialize_variable()
        '''
        # -----------------------------
        # Use thread-safe initialization
        # -----------------------------
        with self._joint_lock:

            # -----------------------------
            # Control flow variables
            # -----------------------------
            self.should_it_start = False
            self.marker_data = None
            self.joint_positions = [0.0] * 6  # Updated to 6 joints for UR5
            
            # -----------------------------
            # Robot configuration
            # -----------------------------
            self.joint_gain = 0.1  # Reduced gain for smoother movement
        
        # -----------------------------
        # Predefined target positions
        # -----------------------------
        self.target_positions = {
            'top_pos': [0.43, 0.1, 0.46],
            'ebot_pos': [0.67, 0.01, -0.1],
            'init_top': [0.16, 0.11, 0.47],
            'second_box': [-0.007, -0.42, 0.23],
            'return_top': [0.16, 0.09, 0.53],
            'last_pos': [-0.11, 0.25, 0.25]
        }
        
        # -----------------------------
        # Predefined joint configurations (in degrees)
        # -----------------------------
        self.init_joint_config = [0, -137, 138, -82, -90, 180]

    def setup_publishers_and_subscribers(self, qos_profile):
        
        '''
        Purpose:
        ---
        Configure ROS publishers and subscribers with thread safety.

        Input Arguments:
        ---
        `qos_profile` : [ QoSProfile]
            QoSProfile object containing configuration for joint states.

        Returns:
        ---
        None

        Example call:
        ---
        future.add_done_callback(self.start_servo_callback)
        '''
        # -----------------------------
        # Twist (linear motion) publisher
        # -----------------------------
        self.twist_publisher = self.create_publisher(
            TwistStamped, 
            '/servo_node/delta_twist_cmds', 
            10
        )
        
        # -----------------------------
        # Joint movement publisher
        # -----------------------------
        self.joint_publisher = self.create_publisher(
            JointJog, 
            '/servo_node/delta_joint_cmds', 
            10
        )
        
        # -----------------------------
        # Aruco marker subscriber with specific callback group
        # -----------------------------
        self.marker_subscriber = self.create_subscription(
            Int32MultiArray, 
            '/detected_aruco_ids', 
            self.update_markers, 
            10, 
            callback_group=self._marker_callback_group
        )
        
        # -----------------------------
        # Joint state subscriber with QoS and specific callback group
        # -----------------------------
        self.joint_subscriber = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.update_joints, 
            qos_profile,
            callback_group=self._joint_callback_group
        )

    def setup_services(self):
        
        '''
        Purpose:
        ---
        Configure ROS services.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.setup_services()
        '''
        # -----------------------------
        # Servo start service client
        # -----------------------------
        self.servo_client = self.create_client(
            Trigger, 
            '/servo_node/start_servo'
        )
        
        # -----------------------------
        # Payload transfer service
        # -----------------------------
        self.passing_service = self.create_service(
            SetBool, 
            'picknplace', 
            self.handle_passing_request, 
            callback_group=self._callback_handler
        )

    def setup_transform_listener(self):
        
        '''
        Purpose:
        ---
        To set up transform listener for coordinate transformations.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.setup_transform_listener()
        '''
        # -----------------------------
        # Initialize the transform listener and buffer to manage transformations
        # -----------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def update_joints(self, msg):
        '''
        Purpose:
        ---
        For thread-safe joint position update.
        
        Input Arguments:
        ---
        `msg` : [ sensor_msgs.msg.JointState ]
            Message containing the current joint state, including positions, velocities, and efforts.
        
        Returns:
        ---
        None
        
        Example call:
        ---
        self.update_joints
        '''
        try:
            # -----------------------------
            # Define expected joint order
            # -----------------------------
            joint_names = [
                'shoulder_pan_joint', 
                'shoulder_lift_joint', 
                'elbow_joint', 
                'wrist_1_joint', 
                'wrist_2_joint', 
                'wrist_3_joint'
            ]
            
            # -----------------------------
            # Create a mapping of joint names to positions
            # -----------------------------
            joint_map = dict(zip(msg.name, msg.position))
            
            # -----------------------------
            # Safely update joint positions with thread lock
            # -----------------------------
            with self._joint_lock:
                # -----------------------------
                # Extract positions in the correct order, default to 0.0 if not found
                # -----------------------------
                self.joint_positions = [
                    joint_map.get(name, 0.0) 
                    for name in joint_names
                ]
        
        except Exception as e:
            self.get_logger().error(f'Error updating joint positions: {e}')

    def get_current_joint_positions(self):
        
        '''
        Purpose:
        ---
        Thread-safe method to retrieve current joint positions.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.get_current_joint_positions()
        '''
        # -----------------------------
        # Safely return joint positions with thread lock
        # -----------------------------
        with self._joint_lock:
            return self.joint_positions.copy()

    def update_markers(self, msg):

        '''
        Purpose:
        ---
        Method for updating detected Aruco marker data.

        Input Arguments:
        ---
        msg : [ Int32MultiArray ]

        Returns:
        ---
        None

        Example call:
        ---
        self.update_markers(msg)
        '''
        # -----------------------------
        # Update detected markers
        # -----------------------------
        self.marker_data = msg.data
        self.get_logger().info(f'Detected Aruco markers: {self.marker_data}')

    def check_position_reached(self, target_angles_rad, tolerance_deg):
    
        '''
        Purpose:
        ---
        Check if current joint positions are within tolerance of target positions.

        Input Arguments:
        ---
        target_angles_rad : [list]
            Target joint angles in radians

        tolerance_deg : [float]
            Tolerance in degrees

        Returns:
        ---
        bool : [bool]
            True if position is reached, False otherwise

        Example call:
        ---
        self.check_position_reached(target_rad, tolerance_deg)
        '''
        # -----------------------------
        # Check if each joint is within tolerance
        # -----------------------------
        #-- < target >: < stores target angles in radians >
        #-- < current >: < stores current joint positions >

        for target, current in zip(target_angles_rad, self.joint_positions):
            error = (target - current + np.pi) % (2 * np.pi) - np.pi
            if abs(np.rad2deg(error)) > tolerance_deg:
                return False
        return True

    def move_joints(self, target_angles_deg):
        '''
        Purpose:
        ---
        Move robot joints to specified target angles.

        Input Arguments:
        ---
        target_angles_deg : [list]
            Target joint angles in radians

        Returns:
        ---
        None

        Example call:
        ---
        self.move_joints(target_angles_deg)
        '''
        #<target_rad> = <convert target_angles_deg to radians>
        #<tolerance_deg> = <set tolerance to 10 degrees>
        target_rad = [np.deg2rad(angle) for angle in target_angles_deg]
        tolerance_deg = 10
        
        # -----------------------------
        # Loop until position is reached
        # -----------------------------
        while rclpy.ok():
            if self.check_position_reached(target_rad, tolerance_deg):
                self.get_logger().info('Position reached')
                break
            
            # <msg>: <create JointJog message>
            msg = JointJog()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_frame"
            msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint']
            msg.velocities = []
            msg.displacements = []
            
            n = 0

            # -----------------------------
            # Calculate error and displacement for each joint
            # -----------------------------

            #<error> : <calculate error for each joint position>
            #<error_deg> : <convert error to degrees>
            #<displacement> : <calculate displacement for each joint>
            #<velocity> : <calculate velocity for each joint>
            #<msg.velocities.append(velocity)> : <append velocity to message>
            #<msg.displacements.append(displacement)> : <append displacement to message>

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

            # -----------------------------
            # Publish joint movement command
            # -----------------------------
            msg.duration = 0.1
            self.joint_publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

    def move_to_position(self, target):
        '''
        Purpose:
        ---
        Move robot joints to specified target angles.

        Input Arguments:
        ---
        target : [list]
            Target [x, y, z] position

        Returns:
        ---
        None

        Example call:
        ---
        self.move_to_position(target)
        '''

        # -----------------------------
        # Set gain and tolerance for proportional control loop 
        # -----------------------------
        gain = 5
        tolerance = 0.08
        # -----------------------------
        # Loop until position is reached
        # -----------------------------
        while rclpy.ok():
            #<pos> : <get current end-effector position>
            pos,_ = self.get_current_position()
            if not pos:
                break
            
            #<error_x> : <calculate error in x direction>
            #<error_y> : <calculate error in y direction>
            #<error_z> : <calculate error in z direction>
            error_x = target[0] - pos.x
            error_y = target[1] - pos.y
            error_z = target[2] - pos.z
            distance = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            #<distance> : <calculate distance from target>
            if distance <= tolerance:
                self.get_logger().info('Position reached')
                break
            
            #<msg>: <create TwistStamped message>
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

        """
        Get current end-effector position and rotation.
        
        Returns:
            tuple: (position, rotation) or (None, None) if transform fails
        """
        '''
        Purpose:
        ---
        Get current end-effector position and rotation.

        Input Arguments:
        ---
        None

        Returns:
        ---
        tuple - 
        pos : [geometry_msgs.msg.Point]
            Current end-effector position

        rot : [geometry_msgs.msg.Quaternion]
            Current end-effector rotation    
        Example call:
        ---
        self.move_to_position(target)
        '''

        # -----------------------------
        # Get current end-effector position
        # -----------------------------
        try:
            #<transform> : <get transform from base_link to wrist_3_link>
            transform = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            #<pos> : <get translation from transform>
            pos = transform.transform.translation
            #<rot> : <get rotation from transform>
            rot = transform.transform.rotation
            return pos, rot
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')
            return None, None

    def initialize_servo(self):
        
        '''
        Purpose:
        ---
        Initialize servo service with robust error handling.

        Input Arguments:
        ---
        None

        Returns:
        ---
        tuple

        pos : [geometry_msgs.msg.Point]
            Current end-effector position

        rot : [geometry_msgs.msg.Quaternion]
            Current end-effector rotation    
        Example call:
        ---
        self.move_to_position(target)
        '''
        try:
            # -----------------------------
            # Wait for servo service to be available
            # -----------------------------
            if not self.servo_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('Servo service not available after 5 seconds')
                return False
            
            # -----------------------------
            # Call servo start service
            # -----------------------------
            future = self.servo_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            # -----------------------------
            # Check if service was successful
            # -----------------------------
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

    def handle_passing_request(self, request, response):

        '''
        Purpose:
        ---
        Handle payload service request.

        Input Arguments:
        ---
        request : [std_srvs.srv.SetBool.Request]
            Incoming service request
        response : [std_srvs.srv.SetBool.Response]    

        Returns:
        ---

        response : [std_srvs.srv.SetBool.Response]
        
        Example call:
        ---
        self.handle_passing_request
        '''
        self.get_logger().info('Received payload service request')
        
        # -----------------------------
        # Check if request is valid
        # -----------------------------
        if request.data:
            self.should_it_start = True
            self.start_and_move_arm()
            response.success = True
            response.message = "Payload transfer initialized"
        
        else:
            response.success = False
            response.message = "Invalid payload request parameters"

        return response

    def attach_object(self, object_name):
        '''
        Purpose:
        ---
        Threaded method to attach an object with timeout and error handling.

        Input Arguments:
        ---
        object_name : [str]
            Name of the object to attach    

        Returns:
        ---

        bool : [bool]
            True if attachment was successful, False otherwise
        
        Example call:
        ---
        self.attach_object(object_name)
        '''
        def attach_task():

            '''
            Purpose:
            ---
            Nested method to attach an object with timeout and error handling.

            Input Arguments:
            ---
            None

            Returns:
            ---

            bool : [bool]
                True if attachment was successful, False otherwise
            
            Example call:
            ---
            attach_task()
            '''

        # -----------------------------
        # Attempt to attach object
        # -----------------------------
            try:
                #<client> : <create AttachLink service client>
                client = self.create_client(AttachLink, '/GripperMagnetON')
                if not client.wait_for_service(timeout_sec=5.0):
                    self.get_logger().error(f'Attach service for {object_name} not available')
                    return False
                #<request> : <create AttachLink request>
                request = AttachLink.Request()
                request.model1_name = object_name
                request.link1_name = 'link'
                request.model2_name = 'ur5'
                request.link2_name = 'wrist_3_link'
                #<future> : <call attach service asynchronously>
                future = client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                
                # -----------------------------
                # Check if attachment was successful
                # -----------------------------
                if future.result() and future.result().success:
                    self.get_logger().info(f'Successfully attached {object_name}')
                    return True
                else:
                    self.get_logger().warning(f'Failed to attach {object_name}')
                    return False
            
            except Exception as e:
                self.get_logger().error(f'Attach object error: {e}')
                return False

        # -----------------------------
        # Submit the attachment task to the thread pool
        # -----------------------------
        future = self._object_manipulation_executor.submit(attach_task)
        
        # -----------------------------
        # Wait for the result with a timeout
        # -----------------------------
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            self.get_logger().error(f'Attach object {object_name} timed out')
            return False

    def detach_object(self, object_name):
        
        '''
        Purpose:
        ---
        Threaded method to detach an object with timeout and error handling.

        Input Arguments:
        ---
        object_name : [str]
            Name of the object to detach    

        Returns:
        ---

        bool : [bool]
            True if detachment was successful, False otherwise
        
        Example call:
        ---
        self.detach_object(object_name)
        '''
        def detach_task():
            '''
            Purpose:
            ---
            Threaded method to detach an object with timeout and error handling.

            Input Arguments:
            ---
            None  

            Returns:
            ---

            bool : [bool]
                True if detachment was successful, False otherwise
            
            Example call:
            ---
            detach_object
            '''
        # -----------------------------
        # Attempt to detach object
        # -----------------------------
            try:
                #<client> : <create DetachLink service client>
                client = self.create_client(DetachLink, '/GripperMagnetOFF')
                if not client.wait_for_service(timeout_sec=5.0):
                    self.get_logger().error(f'Detach service for {object_name} not available')
                    return False

                #<request> : <create DetachLink request>
                request = DetachLink.Request()
                request.model1_name = object_name
                request.link1_name = 'link'
                request.model2_name = 'ur5'
                request.link2_name = 'wrist_3_link'
                
                #<future> : <call detach service asynchronously>
                future = client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                
                # -----------------------------
                # Check if detachment was successful

                if future.result() and future.result().success:
                    self.get_logger().info(f'Successfully detached {object_name}')
                    return True
                else:
                    self.get_logger().warning(f'Failed to detach {object_name}')
                    return False
            
            except Exception as e:
                self.get_logger().error(f'Detach object error: {e}')
                return False

        # -----------------------------
        # Submit the detachment task to the thread pool
        # -----------------------------
        future = self._object_manipulation_executor.submit(detach_task)
        
        # -----------------------------
        # Wait for the result with a timeout
        # -----------------------------
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            self.get_logger().error(f'Detach object {object_name} timed out')
            return False
        
    def start_and_move_arm(self):
        '''
        Purpose:
        ---
        Method for main movement sequence for robot arm.

        Input Arguments:
        ---
        None    

        Returns:
        ---
        None

        Example call:
        ---
        self.start_and_move_arm()
        '''
        # -----------------------------
        # Skip if not triggered by service
        # -----------------------------
        if not self.should_it_start:
            return

        try:
            # -----------------------------
            # Wait for marker detection
            # -----------------------------
            if self.marker_data is None:
                self.get_logger().warn('No markers detected yet')
                return

            # -----------------------------
            # Log detected marker
            # -----------------------------
            marker_id = self.marker_data[0]
            self.get_logger().info(f'Processing marker ID: {marker_id}')

            # -----------------------------
            # Move to initial configuration
            # -----------------------------
            self.move_joints(self.init_joint_config)

            # -----------------------------
            # Sequential movement steps
            # -----------------------------
            movements = [
                ('init_top', "Moving to initial top position"),
                ('second_box', "Moving to second box position"),
            ]

            for pos_key, log_msg in movements:
                self.get_logger().info(log_msg)
                self.move_to_position(self.target_positions[pos_key])

            # -----------------------------
            # Get object transform
            # -----------------------------
            transform = self.tf_buffer.lookup_transform(
                'base_link', 
                f'obj_{marker_id}', 
                rclpy.time.Time()
            )
            
            # -----------------------------
            # Convert transform to target pose
            # -----------------------------
            target_pose = [
                transform.transform.translation.x, 
                transform.transform.translation.y, 
                transform.transform.translation.z
            ]
            
            # -----------------------------
            # Move to object and manipulate
            # -----------------------------
            self.move_to_position(target_pose)
            self.attach_object(f'box{marker_id}')
            
            # -----------------------------
            # Return and drop sequence
            # -----------------------------
            drop_positions = [
                ('return_top', "Returning to top position"),
                ('ebot_pos', "Moving to drop position")
            ]
            
            for pos_key, log_msg in drop_positions:
                self.get_logger().info(log_msg)
                self.move_to_position(self.target_positions[pos_key])
            
            # -----------------------------
            # Detach and return to initial position
            # -----------------------------
            self.detach_object(f'box{marker_id}')
            self.move_to_position(self.target_positions['init_top'])
            
            # Reset flag
            self.should_it_start = False

        except Exception as e:
            self.get_logger().error(f'Arm movement error: {e}')
            self.should_it_start = False

    def update_markers(self, msg):

        '''
        Purpose:
        ---
        Update detected Aruco marker data.

        Input Arguments:
        ---
        msg : [ Int32MultiArray ]    

        Returns:
        ---
        None

        Example call:
        ---
        self.update_markers(msg)
        '''
        self.marker_data = msg.data
        self.get_logger().info(f'Detected Aruco markers: {self.marker_data}')


def main(args=None):
    '''
        Purpose:
        ---
        This is the entry point for the robot arm manipulation process. It initializes the ROS 2 node, starts the robot's arm movement, 
        and then shuts down the node once  the process is complete.

        Input Arguments:
        ---
        args (list, optional): 
        Command-line arguments to initialize ROS 2 (default is None). This is typically used to pass arguments to nodes
        if necessary, but in this case, it's left as None.

        Returns:
        ---
        None. This function is the entry point for the robot arm manipulation process.

        Example call:
        ---
        main()
        This function is automatically called when the script is executed.
        '''
    rclpy.init(args=args)
    
    try:
        node = RobotController()
        
        # -----------------------------
        # Create executor with increased thread pool
        # -----------------------------
        # <executor> : <create MultiThreadedExecutor with 4 threads>
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard interrupt received, shutting down...')
        finally:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
    
    except Exception as e:
        print(f"Initialization error: {e}")

if __name__ == '__main__':
    main()
