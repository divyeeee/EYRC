#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from payload_service.srv import PicknPlace
from rclpy.callback_groups import ReentrantCallbackGroup
import time

class PickAndPlaceClient(Node):
    def __init__(self):
        super().__init__('pick_and_place_client')
        self.callback_group = ReentrantCallbackGroup()
        self.client = self.create_client(
            PicknPlace, 
            'picknplace',
            callback_group=self.callback_group
        )

    def wait_for_service(self):
        self.get_logger().info('Waiting for service...')
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service not available')
            return False
        return True

    def send_request(self, box_name):
        if not self.wait_for_service():
            return False

        request = PicknPlace.Request()
        request.data = True
        request.box_name = box_name
        
        self.get_logger().info(f'Processing {box_name}...')
        
        try:
            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                response = future.result()
                self.get_logger().info(f'Finished {box_name}: {response.message}')
                return True
            else:
                self.get_logger().error(f'Service call failed for {box_name}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Service call failed for {box_name}: {str(e)}')
            return False

def main(args=None):
    rclpy.init(args=args)
    
    client = PickAndPlaceClient()
    boxes = ['box1', 'box2', 'box3']
    
    try:
        for box in boxes:
            # Process each box
            success = client.send_request(box)
            
            if success:
                client.get_logger().info(f'Completed {box}, waiting 20 seconds...')
            else:
                client.get_logger().error(f'Failed {box}, waiting 20 seconds...')
            
            # Wait between boxes
            time.sleep(20)
            
    except KeyboardInterrupt:
        client.get_logger().info('Operation interrupted by user')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()