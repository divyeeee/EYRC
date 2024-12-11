import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped

class ServoTwistPublisher(Node):
    def __init__(self):
        super().__init__('servo_twist_publisher') 
        
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
        
        # Set the publishing rate to 125 Hz
        self.timer = self.create_timer(1/125, self.publish_twist)
    
    def publish_twist(self):
        # Create a TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set the linear velocity (z = 0.2 m/s)
        twist_msg.twist.linear.x = 0.0
        twist_msg.twist.linear.y = 0.2
        twist_msg.twist.linear.z = 0.2
        
        # Set the angular velocity (all 0.0)
        twist_msg.twist.angular.x = 0.2
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = 0.0
        
        # Publish the twist message
        self.twist_pub.publish(twist_msg)
        self.get_logger().info('Published twist command.')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create the node and spin
        node = ServoTwistPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
