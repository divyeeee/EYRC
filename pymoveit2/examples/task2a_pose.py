import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5
import time
import math

class ServoControlNode(Node):
    def __init__(self):
        super().__init__('servo_control_node')

        # Create a service client to call '/servo_node/start_servo'
        self.servo_start_client = self.create_client(Trigger, '/servo_node/start_servo')

        # Wait for the service to be available
        self.get_logger().info('Waiting for /servo_node/start_servo service...')
        while not self.servo_start_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /servo_node/start_servo service...')

        # Create a publisher for the '/servo_node/delta_twist_cmds' topic
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.get_logger().info('Publisher for /servo_node/delta_twist_cmds initialized.')

        # Initialize MoveIt2
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur5.joint_names(),
            base_link_name=ur5.base_link_name(),
            end_effector_name=ur5.end_effector_name(),
            group_name=ur5.MOVE_GROUP_ARM,
        )

        # Service client to get forward kinematics (end-effector pose)
        self.fk_service_client = self.create_client(GetPositionFK, '/compute_fk')

        # Subscribe to the joint states topic
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Store the current joint positions
        self.current_joint_positions = None

        self.start_servo()

    def joint_state_callback(self, msg):
        """Callback function to store the current joint positions."""
        joint_positions = {}
        for name, position in zip(msg.name, msg.position):
            joint_positions[name] = position
        self.current_joint_positions = [joint_positions[joint_name] for joint_name in ur5.joint_names()]

    def wait_for_joint_states(self, timeout=5.0):
        """Wait until joint states are received or timeout."""
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        while self.current_joint_positions is None:
            if (self.get_clock().now().seconds_nanoseconds()[0] - start_time) > timeout:
                self.get_logger().error("Timeout waiting for joint states.")
                return False
            self.get_logger().info('Waiting for joint states...')
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def start_servo(self):
        # Create a request to call the Trigger service
        request = Trigger.Request()

        # Call the service asynchronously and wait for result
        future = self.servo_start_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().success:
            self.get_logger().info('Servo started successfully.')
        else:
            self.get_logger().error('Failed to start servo. Check service.')

    def get_end_effector_pose(self):
        """Function to get the current pose of the end effector using FK."""
        if not self.wait_for_joint_states():
            return [0.0, 0.0, 0.0]  # Default value if no joint states received

        request = GetPositionFK.Request()
        request.header.frame_id = 'base_link'
        request.fk_link_names = [ur5.end_effector_name()]
        request.robot_state.joint_state.name = ur5.joint_names()
        request.robot_state.joint_state.position = self.current_joint_positions

        # Call the FK service and get the pose
        future = self.fk_service_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            pose = future.result().pose_stamped[0].pose
            return [pose.position.x, pose.position.y, pose.position.z]
        else:
            self.get_logger().error('Failed to get end-effector pose.')
            return [0.0, 0.0, 0.0]  # Return a default value in case of failure

    def p_controller(self, final_pose):
        # Get the initial position of the end effector
        initial_pose = self.get_end_effector_pose()
        self.get_logger().info(f"Initial pose of end effector: {initial_pose}")

        # Create a TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.header = Header()
        twist_msg.header.frame_id = 'base_link'  # Ensure this is correct for your robot
        
        # Define proportional gain (Kp)
        Kp = 0.6  # Adjust the gain as necessary

        rate_hz = 125
        rate = self.create_rate(rate_hz)

        # Threshold for stopping the loop
        position_threshold = 0.01

        # Publish twist commands continuously
        self.get_logger().info('Publishing twist commands at {} Hz'.format(rate_hz))
        while rclpy.ok():
            # Get the current end effector position
            current_pose = self.get_end_effector_pose()

            # Calculate the difference between current and final positions
            error_x = final_pose[0] - current_pose[0]
            error_y = final_pose[1] - current_pose[1]
            error_z = final_pose[2] - current_pose[2]

            # Compute the control signal (velocity)
            vel_x = Kp * error_x
            vel_y = Kp * error_y
            vel_z = Kp * error_z

            # Limit the velocity to a maximum value
            max_velocity = 0.6
            vel_x = max(min(vel_x, max_velocity), -max_velocity)
            vel_y = max(min(vel_y, max_velocity), -max_velocity)
            vel_z = max(min(vel_z, max_velocity), -max_velocity)

            # Set the linear velocities
            twist_msg.twist.linear.x = vel_x
            twist_msg.twist.linear.y = vel_y
            twist_msg.twist.linear.z = vel_z
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.0

            # Update the timestamp
            twist_msg.header.stamp = self.get_clock().now().to_msg()

            # Publish the message
            self.twist_pub.publish(twist_msg)

            # Log the command
            self.get_logger().info(f"Current Pose: {current_pose}")
            self.get_logger().info(f"Final Pose: {final_pose}")
            self.get_logger().info(f"Twist command: {twist_msg.twist}")

            # Check if the position error is small enough to stop
            if abs(error_x) < position_threshold and abs(error_y) < position_threshold and abs(error_z) < position_threshold:
                self.get_logger().info("Reached target position. Stopping the controller.")
                break

            # Sleep to maintain the loop rate
            time.sleep(1/rate_hz)

def main(args=None):
    rclpy.init(args=args)

    # Initialize the node
    node = ServoControlNode()

    try:
        # Call the service to start the servo
        node.start_servo()

        # Define final target pose for the P controller
        final_pose = [0.110446, 0.608, 0.279]  # Update with your actual target pose

        # Start P-controller to move to final pose
        node.p_controller(final_pose)

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()