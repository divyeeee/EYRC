import rclpy
from rclpy.node import Node
from payload_service.srv import PayloadSW

class PayloadServiceCaller(Node):
    def __init__(self):
        super().__init__('payload_service_caller')
        
        # Log node initialization
        self.get_logger().info('Initializing PayloadServiceCaller node...')

        # Create a client for the payload service
        self.client = self.create_client(PayloadSW, 'picknplace')

        # Log the attempt to connect to the service
        self.get_logger().info('Waiting for the "picknplace" service to be available...')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('"picknplace" service is not available. Retrying...')

        self.get_logger().info('"picknplace" service is now available.')

        # Send the request
        self.send_request()

    def send_request(self):
        """Send a service request to the robot controller."""
        # Log the creation of a service request
        self.get_logger().info('Creating a service request for "picknplace"...')

        request = PayloadSW.Request()

        # Set request parameters
        request.receive = True
        request.drop = True

        # Log the request details
        self.get_logger().info(f'Request parameters set: receive={request.receive}, drop={request.drop}')

        # Call the service asynchronously
        future = self.client.call_async(request)

        # Log that the service call has been made
        self.get_logger().info('Service call made. Waiting for the response...')

        # Add a callback to handle the response
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        """Handle the service response."""
        try:
            # Get the response
            response = future.result()

            # Log the response
            if response.success:
                self.get_logger().info(f'Service call successful: {response.message}')
            else:
                self.get_logger().warning(f'Service call failed: {response.message}')
        except Exception as e:
            # Log the exception
            self.get_logger().error(f'An exception occurred during the service call: {e}')

        # Shut down the node after processing the response
        self.get_logger().info('Shutting down PayloadServiceCaller node...')
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # Log the initialization of the program
    print('Starting PayloadServiceCaller node...')

    try:
        node = PayloadServiceCaller()
        rclpy.spin(node)
    except Exception as e:
        # Log any errors that occur during execution
        print(f"An error occurred: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
            print('PayloadServiceCaller node destroyed.')
        rclpy.shutdown()
        print('ROS2 shutdown complete.')

if __name__ == '__main__':
    main()
