import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import tf2_ros
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header, Int32MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger 
from payload_service.srv import PayloadSW

# Custom service imports
from payload_service.srv import PayloadSW
from linkattacher_msgs.srv import AttachLink, DetachLink

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        
        # Setup configuration and services
        self._callback_handler = ReentrantCallbackGroup()
        
        # Initialize class variables
        self.initialize_variables()
        
        # Setup communication interfaces
        self.setup_publishers_and_subscribers()
        self.setup_services()
        self.setup_transform_listener()
        
        # Initialize robot systems
        self.initialize_servo()
        
        # Setup motion control timer
        self._motion_control = self.create_timer(0.5, self.start_and_move_arm)
        
        self.get_logger().info('UR5 Robot Controller Initialized')

    def initialize_variables(self):
        """Initialize all class-level variables with default values."""
        # Control flow variables
        self.should_it_start = False
        self.marker_data = None
        self.joint_positions = [0.0] * 6  # Updated to 6 joints for UR5
        
        # Robot configuration
        self.joint_gain = 0.1  # Reduced gain for smoother movement
        
        # Predefined target positions (in meters)
        self.target_positions = {
            'top_pos': [0.43, 0.1, 0.46],
            'ebot_pos': [0.5, 0.01, -0.1],
            'init_top': [0.16, 0.11, 0.47],
            'second_box': [-0.007, -0.42, 0.23],
            'return_top': [0.16, 0.09, 0.53],
            'last_pos': [-0.11, 0.25, 0.25]
        }
        
        # Predefined joint configurations (in degrees)
        self.init_joint_config = [0, -137, 138, -82, -90, 180]

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
            n=0
            for target, current in zip(target_rad, self.joint_positions):
                n=n+1
                error = (target - current + np.pi) % (2 * np.pi) - np.pi
                error_deg = np.rad2deg(error)
                self.get_logger().info(f'Error for joint {n}: {error_deg} degrees, current pose:{current}, target pose:{target}')

                if abs(error_deg) > tolerance_deg or (error_deg<-170 and error_deg>-180):
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
            pos, _ = self.get_current_position()
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

    def setup_publishers_and_subscribers(self):
        """Configure ROS publishers and subscribers."""
        # Twist (linear motion) publisher
        self.twist_publisher = self.create_publisher(
            TwistStamped, 
            '/servo_node/delta_twist_cmds', 
            10
        )
        
        # Joint movement publisher
        self.joint_publisher = self.create_publisher(
            JointJog, 
            '/servo_node/delta_joint_cmds', 
            10
        )
        
        # Aruco marker subscriber
        self.marker_subscriber = self.create_subscription(
            Int32MultiArray, 
            '/detected_aruco_ids', 
            self.update_markers, 
            10
        )
        
        # Joint state subscriber
        self.joint_subscriber = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.update_joints, 
            10
        )

    def setup_services(self):
        """Configure ROS services."""
        # Servo start service client
        self.servo_client = self.create_client(
            Trigger, 
            '/servo_node/start_servo'
        )
        
        # Payload transfer service
        self.passing_service = self.create_service(
            PayloadSW, 
            'picknplace', 
            self.handle_passing_request, 
            callback_group=self._callback_handler
        )

    def update_markers(self, msg):
        """Update detected Aruco marker data."""
        self.marker_data = msg.data
        self.get_logger().info(f'Detected Aruco markers: {self.marker_data}')

    def update_joints(self, msg):
        """Update current joint positions."""
        self.joint_positions = msg.position

    def setup_transform_listener(self):
        """Set up transform listener for coordinate transformations."""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def handle_passing_request(self, request, response):
        """Handle payload service request."""
        self.get_logger().info('Received payload service request')
        
        if request.receive and request.drop:
            self.should_it_start = True
            self.start_and_move_arm()
            response.success = True
            response.message = "Payload transfer initialized"
        else:
            response.success = False
            response.message = "Invalid payload request parameters"

        return response

    def initialize_servo(self):
        """Initialize servo service with robust error handling."""
        try:
            # Wait for servo service to be available
            if not self.servo_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('Servo service not available after 5 seconds')
                return False
            
            # Call servo start service
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
        client = self.create_client(AttachLink, '/GripperMagnetON')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for attach service...')

        request = AttachLink.Request()
        request.model1_name = object_name
        request.link1_name = 'link'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def detach_object(self, object_name):
        client = self.create_client(DetachLink, '/GripperMagnetOFF')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for detach service...')

        request = DetachLink.Request()
        request.model1_name = object_name
        request.link1_name = 'link'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
    def start_and_move_arm(self):
        """Main movement sequence for robot arm."""
        # Skip if not triggered by service
        if not self.should_it_start:
            return

        try:
            # Wait for marker detection
            if self.marker_data is None:
                self.get_logger().warn('No markers detected yet')
                return

            # Log detected marker
            marker_id = self.marker_data[0]
            self.get_logger().info(f'Processing marker ID: {marker_id}')

            # Move to initial configuration
            self.move_joints(self.init_joint_config)

            # Sequential movement steps
            movements = [
                ('init_top', "Moving to initial top position"),
                ('second_box', "Moving to second box position"),
            ]

            for pos_key, log_msg in movements:
                self.get_logger().info(log_msg)
                self.move_to_position(self.target_positions[pos_key])

            # Get object transform
            transform = self.tf_buffer.lookup_transform(
                'base_link', 
                f'obj_{marker_id}', 
                rclpy.time.Time()
            )
            
            # Convert transform to target pose
            target_pose = [
                transform.transform.translation.x, 
                transform.transform.translation.y, 
                transform.transform.translation.z
            ]
            
            # Move to object and manipulate
            self.move_to_position(target_pose)
            self.attach_object(f'box{marker_id}')
            
            # Return and drop sequence
            drop_positions = [
                ('return_top', "Returning to top position"),
                ('ebot_pos', "Moving to drop position")
            ]
            
            for pos_key, log_msg in drop_positions:
                self.get_logger().info(log_msg)
                self.move_to_position(self.target_positions[pos_key])
            
            # Detach and return to initial position
            self.detach_object(f'box{marker_id}')
            self.move_to_position(self.target_positions['init_top'])
            
            # Reset flag
            self.should_it_start = False

        except Exception as e:
            self.get_logger().error(f'Arm movement error: {e}')
            self.should_it_start = False


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RobotController()
        executor = MultiThreadedExecutor()
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