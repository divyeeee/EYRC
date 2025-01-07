'''
# Team ID:          1114
# Theme:            Logistic coBot
# Author List:      Anuj, Yashita, Chirayu, Divye
# Filename:         new_service.5.py
# Functions:        _configure_parameters, _initialize_state, 
#                   _setup_services, _setup_communication,_emergency_halt, 
#                   _process_odometry, _process_left_proximity, 
#                   _process_left_proximity, _evaluate_proximity,  
#                   _handle_docking_request, _execute_control_loop, main
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

class AutonomousDockingSystem(Node):
    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the AutonomousDockingSystem node, sets up ROS 2 publishers, 
        subscribers, service servers, and control timers. Defines initial parameters 
        for robot docking behavior and motion control.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        node = AutonomousDockingSystem()

        '''
        super().__init__('autonomous_docking_system')
        
        # -----------------------------
        # Configuration parameters
        # -----------------------------
        self._configure_parameters()
        
        # -----------------------------
        # Initialize state variables
        # -----------------------------
        self._initialize_state()
        
        # -----------------------------
        # Set up ROS2 communication
        # -----------------------------
        self._setup_communication()
        
        # -----------------------------
        # Initialize control loop
        # -----------------------------
        #Adding timer to give functionality of callback in timer after certain defined interval
        self._motion_control = self.create_timer(0.1, self._execute_control_loop)
        self.get_logger().info('Autonomous Docking System Initialized')

    def _configure_parameters(self):
    
        '''
        Purpose:
        ---
        Method for configuring system parameters such as proximity thresholds, 
        velocity limits, and orientation tolerances.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self._configure_parameters()

        '''
        self.PROXIMITY_THRESHOLD = 0.05  # 5cm emergency threshold
        self.TARGET_PROXIMITY = 0.05     # 5cm target distance
        self.PROXIMITY_MARGIN = 0.01     # 1cm tolerance
        self.VELOCITY_LINEAR_MAX = 0.4   # Maximum linear velocity
        self.VELOCITY_ANGULAR_MAX = 0.4  # Maximum angular velocity
        self.ORIENTATION_THRESHOLD = 0.05 # Orientation tolerance

    def _initialize_state(self):

        '''
        Purpose:
        ---
        Method for initialize internal state variables such as position data,
        proximity sensor readings, and docking status.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self._initialize_state()
        '''
        self.position_data = [0.0, 0.0, 0.0]  # x, y, yaw
        self.proximity_left = None
        self.proximity_right = None
        self.docking_active = False
        self.motion_halted = False
        self.approach_enabled = False
        self.rotation_enabled = False
        self.target_pose = {
            'distance': 0.0,
            'orientation': 0.0,
            'station_id': None
        }

    def _setup_communication(self):

        '''
        Purpose:
        ---
        Method for setting up ROS2 publishers, subscribers and services for
        communication with the robot's sensors, actuators, and external
        control interfaces. 

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self._setup_communication()
        
        '''
        #Creating a callback handler attribute for Allowing callbacks to be run parallely without restriction
        self._callback_handler = ReentrantCallbackGroup()
        
        # -----------------------------
        # Subscribers
        # -----------------------------
        self.create_subscription(
            Odometry, 
            'odom', 
            self._process_odometry, 
            10
        )
        self.create_subscription(
            Range, 
            '/ultrasonic_rl/scan', 
            self._process_left_proximity, 
            10
        )
        self.create_subscription(
            Range, 
            '/ultrasonic_rr/scan', 
            self._process_right_proximity, 
            10
        )
        
        # -----------------------------
        # Publisher
        # -----------------------------
        self.movement_publisher = self.create_publisher(
            Twist, 
            'cmd_vel', 
            10
        )
        
        # -----------------------------
        # Service
        # -----------------------------
        self.docking_service = self.create_service(
            DockSw, 
            'dock_control', 
            self._handle_docking_request,
            callback_group=self._callback_handler
        )

    def _emergency_halt(self):
       
        '''
        Purpose:
        ---
        Method for executing an emergency stop procedure to halt robot motion 
        and disable further docking operations. 

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self._emergency_halt()
        
        '''
        #Creating A Twist object for storing linear and angular velocities
        halt_command = Twist()
        self.movement_publisher.publish(halt_command)
        #Variable that Tells if motion is stopped or not 
        self.motion_halted = True
        self.approach_enabled = False
        #Boolean variable telling if Docking service is active or not 
        self.docking_active = False
        #Logging information if the Movement is halted in the form of a warning message 
        self.get_logger().warn('Emergency halt initiated!')

    def _process_odometry(self, data):

        '''
        Purpose:
        ---
        Method for processing odometry data updates from the robot's onboard
        localization system. Extracts position and orientation data from the
        odometry message and updates the internal state variables.

        Input Arguments:
        ---
        data : [Odometry]
            [Odometry] message containing robot pose data.

        Returns:
        ---
        None

        Example call:
        ---
        self._process_odometry(data)
        
        '''
        #Setting position values using data parameter on function
        self.position_data[0] = data.pose.pose.position.x
        self.position_data[1] = data.pose.pose.position.y
        quaternion = data.pose.pose.orientation
        #Setting Yaw value using quaternion function
        _, _, yaw = euler_from_quaternion([
            quaternion.x, 
            quaternion.y, 
            quaternion.z, 
            quaternion.w
        ])
        #Putting yaw value into postion_data
        self.position_data[2] = yaw

    def _process_left_proximity(self, data):
        '''
        Purpose:
        ---
        Method for processing left proximity sensor data updates. Extracts
        range data from the sensor message and updates the internal state
        variable for left proximity.

        Input Arguments:
        ---
        data : [Range]
            [Range] message containing left proximity sensor data.

        Returns:
        ---
        None

        Example call:
        ---
        self._process_left_proximity(data)
        
        '''
        
        self.proximity_left = data.range
        self.get_logger().info(f'Left proximity: {self.proximity_left:.2f}m')
        self._evaluate_proximity()

    def _process_right_proximity(self, data):

        '''
        Purpose:
        ---
        Method for processing right proximity sensor data updates. Extracts
        range data from the sensor message and updates the internal state
        variable for right proximity.

        Input Arguments:
        ---
        data : [Range]
            [Range] message containing right proximity sensor data.

        Returns:
        ---
        None

        Example call:
        ---
        self._setup_communication()
        
        '''
        #Adding variable for right proximity using data parameter
        self.proximity_right = data.range
        #Logging information to terminal 
        self.get_logger().info(f'Right proximity: {self.proximity_right:.2f}m')
        self._evaluate_proximity()

    def _evaluate_proximity(self):
        """Evaluate proximity sensor data and take action if needed"""
        '''
        Purpose:
        ---
        Method for evaluating the current proximity sensor data and taking
        appropriate action based on the proximity threshold and target distance.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self._evaluate_proximity()
        
        '''
        # If any of the proximity sensor data is missing, return without action 
        if None in (self.proximity_left, self.proximity_right):
            return
         #<current_proximity> is the average of the left and right proximity sensor readings   
        current_proximity = (self.proximity_left + self.proximity_right) / 2.0
        #Logging information to terminal
        self.get_logger().info(f'Average proximity: {current_proximity:.2f}m')
        #If the current proximity is less than the threshold, halt the robot motion
        if current_proximity <= self.PROXIMITY_THRESHOLD:
            self.get_logger().warn(f'Emergency stop - proximity {current_proximity:.2f}m')
            self._emergency_halt()
        elif abs(current_proximity - self.TARGET_PROXIMITY) <= self.PROXIMITY_MARGIN:
            self.get_logger().info(f'Target proximity achieved: {current_proximity:.2f}m')
            self._emergency_halt()

    def _handle_docking_request(self, request, response):
        """Handle incoming docking service requests"""
        '''
        Purpose:
        ---
        Method for handling incoming docking service requests. Parses the request
        data to determine the docking operation to be performed and sets the internal
        state variables accordingly.

        Input Arguments:
        ---
        request : [DockSw.Request]
            [DockSw.Request] message containing docking request data.
        response : [DockSw.Response]
            [DockSw.Response] message containing response data.


        Returns:
        ---
        response : [DockSw.Response]
            [DockSw.Response] message containing the response to the request.


        Example call:
        ---
        response = self._handle_docking_request(request, response)
        
        '''
        #If the startcmd is true, then the docking service is active and the robot is moving towards the station
        if request.startcmd:
            self.docking_active = True
            self.approach_enabled = request.linear_dock
            self.rotation_enabled = request.orientation_dock
            self.target_pose['distance'] = request.distance
            self.target_pose['orientation'] = request.orientation
            self.target_pose['station_id'] = request.rack_no
            self.motion_halted = False
            
            #<response> is the response message to be sent back to the client
            response.success = True
            response.message = f"Docking initiated for station {self.target_pose['station_id']}"
            self.get_logger().info(response.message)
        #If the undocking command is received, the robot is undocking from the station    
        elif request.undocking:
            self.docking_active = False
            self._emergency_halt()
            response.success = True
            response.message = f"Undocking from station {self.target_pose['station_id']}"
            self.get_logger().info(response.message)
        # If the docking command is invalid, set the response message to indicate an error    
        else:
            response.success = False
            response.message = "Invalid docking command"
            self.get_logger().warn(response.message)
            
        return response

    def _execute_control_loop(self):
        """Execute the main control loop"""
        '''
        Purpose:
        ---
        Method for executing the main control loop for the autonomous docking system.
        Evaluates the current proximity sensor data and computes the control commands
        for linear and angular velocity based on the target proximity and orientation.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self._execute_control_loop()
        
        '''
        #If the docking service is not active or the robot motion is halted, return without action
        if not self.docking_active or self.motion_halted:
            return

        #If the proximity sensor data is missing, log a warning message and return without action
        if None in (self.proximity_left, self.proximity_right):
            self.get_logger().warn('Awaiting proximity sensor data...')
            return

        #Create a Twist object for storing linear and angular velocity commands
        #<movement_command> is the Twist message to be published to the robot's motion controller
        movement_command = Twist()

        #<current_proximity> is the average of the left and right proximity sensor readings
        current_proximity = (self.proximity_left + self.proximity_right) / 2.0

        #If the current proximity is less than the threshold, halt the robot motion
        if current_proximity <= self.PROXIMITY_THRESHOLD:
            self._emergency_halt()
            return
        
        #If the approach behavior is enabled, compute the linear velocity command
        if self.approach_enabled:

            #<proximity_error> is the difference between the target proximity and the current proximity
            proximity_error = self.TARGET_PROXIMITY - current_proximity
            self.get_logger().info(f'Proximity error: {proximity_error:.2f}m')

            #If the proximity error is within the tolerance, halt the robot motion
            if abs(proximity_error) <= self.PROXIMITY_MARGIN:
                self._emergency_halt()

            #Otherwise, compute the linear velocity command based on the proximity error    
            else:
                #<linear_velocity> is the proportional control term for linear velocity
                linear_velocity = 1.0 * proximity_error
                movement_command.linear.x = max(
                    -self.VELOCITY_LINEAR_MAX, 
                    min(self.VELOCITY_LINEAR_MAX, linear_velocity)
                )
                self.get_logger().info(f'Linear velocity: {movement_command.linear.x:.2f}')
        
        #If the rotation behavior is enabled, compute the angular velocity command
        if self.rotation_enabled:
            #<orientation_error> is the difference between the target orientation and the current orientation
            orientation_error = self.target_pose['orientation'] - self.position_data[2]
            orientation_error = math.atan2(
                math.sin(orientation_error), 
                math.cos(orientation_error)
            )

            #If the orientation error is within the tolerance, disable the rotation behavior
            if abs(orientation_error) < self.ORIENTATION_THRESHOLD:
                self.rotation_enabled = False

            #Otherwise, compute the angular velocity command based on the orientation error    
            else:
                angular_velocity = 1.0 * orientation_error
                movement_command.angular.z = max(
                    -self.VELOCITY_ANGULAR_MAX, 
                    min(self.VELOCITY_ANGULAR_MAX, angular_velocity)
                )

        #Publish the computed movement command to the robot's motion controller
        if not self.motion_halted:
            self.movement_publisher.publish(movement_command)

def main(args=None):
    '''
    Purpose:
    ---
    Main method for initializing the AutonomousDockingSystem node and starting the
    ROS 2 executor for handling communication and control tasks.

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
    rclpy.init(args=args)
    
    try:
        #<node> is the AutonomousDockingSystem node object
        node = AutonomousDockingSystem()
        #<executor> is the MultiThreadedExecutor object for running the ROS 2 node
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        #Run the ROS 2 node until shutdown
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()