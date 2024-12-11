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
        super().__init__('autonomous_docking_system')
        
        # Configuration parameters
        self._configure_parameters()
        
        # Initialize state variables
        self._initialize_state()
        
        # Set up ROS2 communication
        self._setup_communication()
        
        # Initialize control loop
        self._motion_control = self.create_timer(0.1, self._execute_control_loop)
        self.get_logger().info('Autonomous Docking System Initialized')

    def _configure_parameters(self):
        """Configure system parameters"""
        self.PROXIMITY_THRESHOLD = 0.05  # 5cm emergency threshold
        self.TARGET_PROXIMITY = 0.05     # 5cm target distance
        self.PROXIMITY_MARGIN = 0.01     # 1cm tolerance
        self.VELOCITY_LINEAR_MAX = 0.4   # Maximum linear velocity
        self.VELOCITY_ANGULAR_MAX = 0.4  # Maximum angular velocity
        self.ORIENTATION_THRESHOLD = 0.05 # Orientation tolerance

    def _initialize_state(self):
        """Initialize internal state variables"""
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
        """Setup ROS2 publishers, subscribers and services"""
        self._callback_handler = ReentrantCallbackGroup()
        
        # Subscribers
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
        
        # Publisher
        self.movement_publisher = self.create_publisher(
            Twist, 
            'cmd_vel', 
            10
        )
        
        # Service
        self.docking_service = self.create_service(
            DockSw, 
            'dock_control', 
            self._handle_docking_request,
            callback_group=self._callback_handler
        )

    def _emergency_halt(self):
        """Execute emergency stop procedure"""
        halt_command = Twist()
        self.movement_publisher.publish(halt_command)
        self.motion_halted = True
        self.approach_enabled = False
        self.docking_active = False
        self.get_logger().warn('Emergency halt initiated!')

    def _process_odometry(self, data):
        """Process odometry data updates"""
        self.position_data[0] = data.pose.pose.position.x
        self.position_data[1] = data.pose.pose.position.y
        quaternion = data.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            quaternion.x, 
            quaternion.y, 
            quaternion.z, 
            quaternion.w
        ])
        self.position_data[2] = yaw

    def _process_left_proximity(self, data):
        """Process left proximity sensor data"""
        self.proximity_left = data.range
        self.get_logger().info(f'Left proximity: {self.proximity_left:.2f}m')
        self._evaluate_proximity()

    def _process_right_proximity(self, data):
        """Process right proximity sensor data"""
        self.proximity_right = data.range
        self.get_logger().info(f'Right proximity: {self.proximity_right:.2f}m')
        self._evaluate_proximity()

    def _evaluate_proximity(self):
        """Evaluate proximity sensor data and take action if needed"""
        if None in (self.proximity_left, self.proximity_right):
            return
            
        current_proximity = (self.proximity_left + self.proximity_right) / 2.0
        self.get_logger().info(f'Average proximity: {current_proximity:.2f}m')
        
        if current_proximity <= self.PROXIMITY_THRESHOLD:
            self.get_logger().warn(f'Emergency stop - proximity {current_proximity:.2f}m')
            self._emergency_halt()
        elif abs(current_proximity - self.TARGET_PROXIMITY) <= self.PROXIMITY_MARGIN:
            self.get_logger().info(f'Target proximity achieved: {current_proximity:.2f}m')
            self._emergency_halt()

    def _handle_docking_request(self, request, response):
        """Handle incoming docking service requests"""
        if request.startcmd:
            self.docking_active = True
            self.approach_enabled = request.linear_dock
            self.rotation_enabled = request.orientation_dock
            self.target_pose['distance'] = request.distance
            self.target_pose['orientation'] = request.orientation
            self.target_pose['station_id'] = request.rack_no
            self.motion_halted = False
            
            response.success = True
            response.message = f"Docking initiated for station {self.target_pose['station_id']}"
            self.get_logger().info(response.message)
        elif request.undocking:
            self.docking_active = False
            self._emergency_halt()
            response.success = True
            response.message = f"Undocking from station {self.target_pose['station_id']}"
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = "Invalid docking command"
            self.get_logger().warn(response.message)
            
        return response

    def _execute_control_loop(self):
        """Execute the main control loop"""
        if not self.docking_active or self.motion_halted:
            return

        if None in (self.proximity_left, self.proximity_right):
            self.get_logger().warn('Awaiting proximity sensor data...')
            return

        movement_command = Twist()
        current_proximity = (self.proximity_left + self.proximity_right) / 2.0

        if current_proximity <= self.PROXIMITY_THRESHOLD:
            self._emergency_halt()
            return
        
        if self.approach_enabled:
            proximity_error = self.TARGET_PROXIMITY - current_proximity
            self.get_logger().info(f'Proximity error: {proximity_error:.2f}m')

            if abs(proximity_error) <= self.PROXIMITY_MARGIN:
                self._emergency_halt()
            else:
                linear_velocity = 1.0 * proximity_error
                movement_command.linear.x = max(
                    -self.VELOCITY_LINEAR_MAX, 
                    min(self.VELOCITY_LINEAR_MAX, linear_velocity)
                )
                self.get_logger().info(f'Linear velocity: {movement_command.linear.x:.2f}')

        if self.rotation_enabled:
            orientation_error = self.target_pose['orientation'] - self.position_data[2]
            orientation_error = math.atan2(
                math.sin(orientation_error), 
                math.cos(orientation_error)
            )

            if abs(orientation_error) < self.ORIENTATION_THRESHOLD:
                self.rotation_enabled = False
            else:
                angular_velocity = 1.0 * orientation_error
                movement_command.angular.z = max(
                    -self.VELOCITY_ANGULAR_MAX, 
                    min(self.VELOCITY_ANGULAR_MAX, angular_velocity)
                )

        if not self.motion_halted:
            self.movement_publisher.publish(movement_command)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = AutonomousDockingSystem()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
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