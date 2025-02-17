'''
# Team ID:          1118
# Theme:            Logistic coBot
# Author List:      Saeesh , Sambhav, Anshul , Robin
# Filename:         ebot_docking.py
# Functions:        force_stop odometry_callback, ultrasonic_rl_callback, ultrasonic_rr_callback ,
#                   check_distance, dock_control_callback, controller_loop, main
# Global variables: None
'''
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf_transformations import euler_from_quaternion
from ebot_docking.srv import DockSw 
import math

class MyRobotDockingController(Node):

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the MyRobotDockingController node, sets up ROS 2 publishers, 
        subscribers, service servers, and control timers. Defines initial parameters 
        for robot docking behavior and motion control.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''
        super().__init__('my_robot_docking_controller')
        
        # Log initialization message
        self.get_logger().info('Initializing MyRobotDocking Controller...')

        # Callback group for managing concurrency in service callbacks
        self.callback_group = ReentrantCallbackGroup()

        # -----------------------------
        # Subscribers: Receive sensor and odometry data
        # -----------------------------
        # Subscription to Odometry data for robot position and orientation
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odometry_callback, 10)
        
        # Subscription to Left Ultrasonic Sensor data
        self.ultrasonic_rl_sub = self.create_subscription(
            Range, '/ultrasonic_rl/scan', self.ultrasonic_rl_callback, 10)
        
        # Subscription to Right Ultrasonic Sensor data
        self.ultrasonic_rr_sub = self.create_subscription(
            Range, '/ultrasonic_rr/scan', self.ultrasonic_rr_callback, 10)
        
        self.get_logger().info('All subscribers started')

        # -----------------------------
        # Publishers: Send commands to control the robot
        # -----------------------------
        # Publisher to send velocity commands to the robot
        self.cmd_vel_pub = self.create_publisher(
            Twist, 'cmd_vel', 10)

        # -----------------------------
        # Service Server: Control docking process
        # -----------------------------
        # Service to start/stop docking based on a request
        self.dock_control_srv = self.create_service(
            DockSw, 'dock_control', self.dock_control_callback,
            callback_group=self.callback_group)

        # -----------------------------
        # Docking Parameters
        # -----------------------------
        # Flag to enable/disable docking procedure
        self.is_docking = False

        # Flags to control linear and orientation phases of docking
        self.linear_dock = False
        self.orientation_dock = False

        # Target docking parameters
        self.target_distance = 0.0  # Desired distance to docking point
        self.target_orientation = 0.0  # Desired orientation (angle) for docking
        self.rack_no = None  # Identifier for docking rack/position

        # -----------------------------
        # Robot State
        # -----------------------------
        # Current pose of the robot [x, y, yaw]
        self.robot_pose = [0.0, 0.0, 0.0]  

        # Ultrasonic sensor values
        self.usrleft_value = None  # Left sensor reading
        self.usrright_value = None  # Right sensor reading

        # -----------------------------
        # Control Parameters
        # -----------------------------
        # Distance thresholds for stopping during docking
        self.STOP_DISTANCE = 0.03  # Desired stop distance (3cm)
        self.DISTANCE_TOLERANCE = 0.01  # Allowable error (1cm)

        # Motion constraints (speed limits)
        self.MAX_LINEAR_SPEED = 0.4  # Maximum linear speed (m/s)
        self.MAX_ANGULAR_SPEED = 0.4  # Maximum angular speed (rad/s)

        # Emergency stopping parameters
        self.EMERGENCY_STOP_DISTANCE = 0.03  # Safety threshold for emergency stop
        self.has_stopped = False  # Flag to indicate if emergency stop occurred

        # -----------------------------
        # Timer: Control Loop
        # -----------------------------
        # Timer to run the controller loop periodically at 10Hz (0.1s)
        self.controller_timer = self.create_timer(0.1, self.controller_loop)

        # Log controller initialization completion
        self.get_logger().info('Controller initialized')


    def force_stop(self):
        '''
        Purpose:
        ---
        Implements an emergency stop mechanism to immediately halt the robot's motion.
        Publishes a zero-velocity command to stop linear and angular movement.
        Resets docking-related flags to ensure the docking process is terminated safely.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.force_stop()
        '''

        # -----------------------------
        # Publish Zero-Velocity Command
        # -----------------------------
        # Create a Twist message to stop the robot
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0  # Stop linear motion
        stop_cmd.angular.z = 0.0  # Stop angular motion

        # Publish the stop command to the robot's velocity topic
        self.cmd_vel_pub.publish(stop_cmd)

        # -----------------------------
        # Update State Flags
        # -----------------------------
        # Mark that an emergency stop has occurred
        self.has_stopped = True

        # Reset docking flags to terminate any ongoing docking processes
        self.linear_dock = False
        self.is_docking = False

        # -----------------------------
        # Log Emergency Stop Activation
        # -----------------------------
        # Log a warning message for visibility
        self.get_logger().warn('Emergency stop activated!')


    def odometry_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function to update the robot's current pose (position and orientation)
        based on odometry data. The position is extracted as x and y coordinates, and 
        the orientation is converted from quaternion to yaw angle.

        Input Arguments:
        ---
        `msg` : [ nav_msgs.msg.Odometry ]
            Odometry message containing the robot's position (x, y) and orientation 
            in quaternion format.

        Returns:
        ---
        None

        Example call:
        ---
        This function is automatically triggered when a new odometry message is received:
            self.odom_sub = self.create_subscription(
                Odometry, 'odom', self.odometry_callback, 10)
        '''

        # -----------------------------
        # Update Robot Position (x, y)
        # -----------------------------
        # Extract the x-coordinate and y-coordinate of the robot's position
        self.robot_pose[0] = msg.pose.pose.position.x  # X position
        self.robot_pose[1] = msg.pose.pose.position.y  # Y position

        # -----------------------------
        # Update Robot Orientation (Yaw)
        # -----------------------------
        # Extract the orientation quaternion from the odometry message
        quaternion = msg.pose.pose.orientation
        
        # Convert quaternion to Euler angles to obtain the yaw (rotation around Z-axis)
        _, _, yaw = euler_from_quaternion([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        ])
        self.robot_pose[2] = yaw  # Update robot's yaw angle in radians


    def ultrasonic_rl_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function to update the distance reading from the left ultrasonic sensor.
        The distance value is extracted from the `Range` message, stored in a class 
        variable, and logged for monitoring. It also triggers a check to validate 
        the current distance for docking control.

        Input Arguments:
        ---
        `msg` : [ sensor_msgs.msg.Range ]
            Range message containing the distance measured by the left ultrasonic sensor.

        Returns:
        ---
        None

        Example call:
        ---
        This function is automatically triggered when a new ultrasonic sensor reading is received:
            self.ultrasonic_rl_sub = self.create_subscription(
                Range, '/ultrasonic_rl/scan', self.ultrasonic_rl_callback, 10)
        '''

        # -----------------------------
        # Update Left Ultrasonic Value
        # -----------------------------
        # Store the current distance value from the left ultrasonic sensor
        self.usrleft_value = msg.range

        # -----------------------------
        # Log the Distance Value
        # -----------------------------
        # Log the distance for monitoring purposes with a precision of two decimal places
        self.get_logger().info(f'Left ultrasonic distance: {self.usrleft_value:.2f}m')

        # -----------------------------
        # Trigger Distance Validation
        # -----------------------------
        # Call a function to validate the distance and take necessary docking actions
        self.check_distance()


    def ultrasonic_rr_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function to update the distance reading from the right ultrasonic sensor.
        The distance value is extracted from the `Range` message, stored in a class 
        variable, and logged for monitoring. It also triggers a check to validate 
        the current distance for docking control.

        Input Arguments:
        ---
        `msg` : [ sensor_msgs.msg.Range ]
            Range message containing the distance measured by the right ultrasonic sensor.

        Returns:
        ---
        None

        Example call:
        ---
        This function is automatically triggered when a new ultrasonic sensor reading is received:
            self.ultrasonic_rr_sub = self.create_subscription(
                Range, '/ultrasonic_rr/scan', self.ultrasonic_rr_callback, 10)
        '''

        # -----------------------------
        # Update Right Ultrasonic Value
        # -----------------------------
        # Store the current distance value from the right ultrasonic sensor
        self.usrright_value = msg.range

        # -----------------------------
        # Log the Distance Value
        # -----------------------------
        # Log the distance for monitoring purposes with a precision of two decimal places
        self.get_logger().info(f'Right ultrasonic distance: {self.usrright_value:.2f}m')

        # -----------------------------
        # Trigger Distance Validation
        # -----------------------------
        # Call a function to validate the distance and take necessary docking actions
        self.check_distance()


    def check_distance(self):
        '''
        Purpose:
        ---
        This function checks the average distance measured by the left and right ultrasonic 
        sensors. Based on the distance, it determines whether to trigger an emergency stop 
        (if the robot is too close) or a normal stop when the target distance is reached.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        This function is internally called from the ultrasonic sensor callbacks:
            self.check_distance()
        '''

        # -----------------------------
        # Validate Sensor Readings
        # -----------------------------
        # Check if both left and right ultrasonic values are available
        if self.usrleft_value is not None and self.usrright_value is not None:
            
            # -----------------------------
            # Calculate Average Distance
            # -----------------------------
            # Compute the average distance based on left and right ultrasonic values
            avg_distance = (self.usrleft_value + self.usrright_value) / 2.0
            
            # Log the computed average distance
            self.get_logger().info(f'Average distance: {avg_distance:.2f}m')

            # -----------------------------
            # Emergency Stop Condition
            # -----------------------------
            # If the average distance is below the emergency stop threshold, 
            # immediately stop the robot
            if avg_distance <= self.EMERGENCY_STOP_DISTANCE:
                self.get_logger().warn(f'Emergency stop - distance {avg_distance:.2f}m')
                self.force_stop()

            # -----------------------------
            # Normal Stop at Target Distance
            # -----------------------------
            # If the robot is within the acceptable tolerance of the target distance,
            # stop the robot gracefully
            elif abs(avg_distance - self.STOP_DISTANCE) <= self.DISTANCE_TOLERANCE:
                self.get_logger().info(f'Target distance reached: {avg_distance:.2f}m')
                self.force_stop()

    def dock_control_callback(self, request, response):
        '''
        Purpose:
        ---
        Handles the service request for docking and undocking operations. Based on the input 
        from the request, it initiates or stops docking. Parameters such as linear docking, 
        orientation docking, target distance, and target orientation are configured.

        Input Arguments:
        ---
        `request` :  [ DockSw.Request ]
            Service request containing:
            - startcmd: Bool flag to initiate docking
            - linear_dock: Bool flag to enable linear docking
            - orientation_dock: Bool flag to enable orientation docking
            - distance: Target distance for docking
            - orientation: Target orientation for docking
            - rack_no: Identifier for the docking rack
            - undocking: Bool flag to initiate undocking

        `response` :  [ DockSw.Response ]
            Service response containing:
            - success: Indicates if the operation was successful
            - message: A string message describing the status of the operation

        Returns:
        ---
        `response` :  [ DockSw.Response ]
            The updated response object with success status and message.

        Example call:
        ---
        This function is triggered by the service server:
            dock_control = self.create_service(DockSw, 'dock_control', self.dock_control_callback)
        '''

        # -----------------------------
        # Docking Start Command
        # -----------------------------
        if request.startcmd:
            # Set docking-related parameters from the request
            self.is_docking = True
            self.linear_dock = request.linear_dock
            self.orientation_dock = request.orientation_dock
            self.target_distance = request.distance
            self.target_orientation = request.orientation
            self.rack_no = request.rack_no
            self.has_stopped = False
            
            # Prepare success response for docking initiation
            response.success = True
            response.message = f"Docking initiated for rack {self.rack_no}"
            self.get_logger().info(response.message)

        else:
            # Failure response if docking start command not issued
            response.success = False
            response.message = "Docking start command not issued"
            self.get_logger().warn(response.message)
            return response

        # -----------------------------
        # Undocking Command
        # -----------------------------
        if request.undocking:
            # Stop docking process and reset parameters
            self.is_docking = False
            self.force_stop()

            # Prepare success response for undocking
            response.success = True
            response.message = f"Undocking initiated from rack {self.rack_no}"
            self.get_logger().info(response.message)
            return response

        # -----------------------------
        # Return Final Response
        # -----------------------------
        return response

    def controller_loop(self):
        '''
        Purpose:
        ---
        Control loop for the docking operation. This function handles linear docking, 
        orientation adjustments, and ensures safety using sensor data. The function 
        calculates the required corrections in linear and angular velocities to achieve 
        the target docking position.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        This function is periodically executed by the timer created in the constructor:
            self.controller_timer = self.create_timer(0.1, self.controller_loop)
        '''

        # -----------------------------
        # Pre-checks before Control Loop Execution
        # -----------------------------
        if not self.is_docking or self.has_stopped:
            # Exit if docking is not active or emergency stop has been triggered
            return

        if self.usrleft_value is None or self.usrright_value is None:
            # Exit and warn if sensor data is unavailable
            self.get_logger().warn('Waiting for ultrasonic sensor readings...')
            return

        # Initialize command message
        twist = Twist()

        # Calculate average distance from ultrasonic sensors
        avg_distance = (self.usrleft_value + self.usrright_value) / 2.0

        # -----------------------------
        # Emergency Stop Check
        # -----------------------------
        if avg_distance <= self.EMERGENCY_STOP_DISTANCE:
            # Trigger emergency stop if the robot is too close to the target
            self.force_stop()
            return

        # -----------------------------
        # Linear Docking Control
        # -----------------------------
        if self.linear_dock:
            # Calculate the error in distance
            distance_error = self.STOP_DISTANCE - avg_distance
            self.get_logger().info(f'Distance error: {distance_error:.2f}m')

            # Check if the robot is within the acceptable distance tolerance
            if abs(distance_error) <= self.DISTANCE_TOLERANCE:
                self.get_logger().info('Target distance reached. Stopping...')
                self.force_stop()
                return
            else:
                # Proportional control for linear speed
                linear_speed = 1.0 * distance_error  # Proportional gain = 1.0
                twist.linear.x = max(-self.MAX_LINEAR_SPEED, min(self.MAX_LINEAR_SPEED, linear_speed))
                self.get_logger().info(f'Calculated Linear Speed: {twist.linear.x:.2f} m/s')

        # -----------------------------
        # Orientation Control
        # -----------------------------
        if self.orientation_dock:
            # Calculate orientation error
            orientation_error = self.target_orientation - self.robot_pose[2]
            orientation_error = math.atan2(math.sin(orientation_error), math.cos(orientation_error))
            self.get_logger().info(f'Orientation error: {orientation_error:.2f} rad')

            # Check if the robot is within the acceptable orientation tolerance
            if abs(orientation_error) < 0.05:  # 0.05 radians tolerance (~2.8 degrees)
                self.get_logger().info('Target orientation reached. Stopping orientation control...')
                self.orientation_dock = False
            else:
                # Proportional control for angular speed
                angular_speed = 1.0 * orientation_error  # Proportional gain = 1.0
                twist.angular.z = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular_speed))
                self.get_logger().info(f'Calculated Angular Speed: {twist.angular.z:.2f} rad/s')

        # -----------------------------
        # Command Publication
        # -----------------------------
        if not self.has_stopped:
            # Publish the calculated Twist command only if no emergency stop has been triggered
            self.cmd_vel_pub.publish(twist)

def main(args=None):
    '''
    Purpose:
    ---
    This is the entry point for the ROS 2 node. It initializes the ROS 2 client library, 
    creates and spins the docking controller node, and handles shutdown procedures.

    Input Arguments:
    ---
    args: Optional list of command-line arguments passed to the program.

    Returns:
    ---
    None

    Example call:
    ---
    main()  # Typically called when the script is executed directly.
    '''
    
    # -----------------------------
    # Initialize ROS 2 Client Library
    # -----------------------------
    rclpy.init(args=args)  # Initialize ROS 2 client library with optional arguments

    try:
        # -----------------------------
        # Create Node Instance
        # -----------------------------
        node = MyRobotDockingController()  # Instantiate the docking controller node

        # -----------------------------
        # Set Up Executor and Add Node
        # -----------------------------
        executor = MultiThreadedExecutor()  # Executor for handling callbacks in parallel
        executor.add_node(node)  # Add the created node to the executor

        # -----------------------------
        # Spin the Executor to Handle Callbacks
        # -----------------------------
        try:
            executor.spin()  # Start spinning the executor to process incoming callbacks
        finally:
            # -----------------------------
            # Shutdown Sequence
            # -----------------------------
            executor.shutdown()  # Shut down the executor once the node stops spinning
            node.destroy_node()  # Destroy the node to release resources
            rclpy.shutdown()  # Cleanly shut down the ROS 2 client library

    except Exception as e:
        # -----------------------------
        # Error Handling
        # -----------------------------
        print(f"An error occurred: {str(e)}")  # Log any error encountered during execution

# -----------------------------
# Entry Point Check
# -----------------------------
if __name__ == '__main__':
    main()  # Execute the main function when the script is run directly
