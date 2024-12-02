#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf_transformations import quaternion_from_euler
from payload_service.srv import PayloadSW
from ebot_docking.srv import DockSw
import time

class ChalJaBhai(Node):
    def __init__(self):
        super().__init__('chalega_bhai')
        self.nav = BasicNavigator()
        self.payload_ka_client = self.create_client(PayloadSW, '/payload_sw')
        self.dock_ka_client = self.create_client(DockSw, '/dock_control')

    def request_docker(self, start=True, linear=True, orientation=True, 
                    distance=0.5, orient_value=-1.57, rack="1", undock=False):
        while not self.dock_ka_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Docking service, Please pickup the Phone.............')

        req = DockSw.Request()
        req.startcmd = start
        req.linear_dock = linear
        req.orientation_dock = orientation
        req.distance = distance
        req.orientation = orient_value
        req.rack_no = str(rack)  
        req.undocking = undock

        return self.dock_ka_client.call_async(req)
    
    def request_payload(self, receive, drop):
        while not self.payload_ka_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Payload service, Please Pickup the phone..........')
        req = PayloadSW.Request()
        req.receive = receive
        req.drop = drop        
        return self.payload_ka_client.call_async(req)

    def go_to_pose(self, x, y, yaw):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.nav.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        
        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        self.nav.goToPose(goal_pose)
        i = 0
        while not self.nav.isTaskComplete():
            feedback = self.nav.getFeedback()
            if feedback and i % 5 == 0:
                self.get_logger().info(f'Going to goal: {x}, {y}, {yaw}... ' +
                    f'Distance remaining: {feedback.distance_remaining:.2f} meters.')
            i += 1
            
        result = self.nav.getResult()
        if result == "SUCCEEDED":
            self.get_logger().info('Goal succeeded!')
        elif result == "CANCELED":
            self.get_logger().info('Goal was canceled!')
        else:
            self.get_logger().warn('Goal failed!')

def main(args=None):
    rclpy.init(args=args)
    navigator = ChalJaBhai()
    navigator.nav.lifecycleStartup()
    navigator.get_logger().info('Starting Navigation...')
    navigator.go_to_pose(0.44, -2.4, -3.01)
    navigator.request_payload(receive=True, drop=False)
    time.sleep(1)
    navigator.go_to_pose(2.32, 2.65, -1.71)
    navigator.request_docker(rack=1)
    time.sleep(4)  
    navigator.request_payload(receive=False, drop=True)
    navigator.request_docker(undock=True, rack=1)
    time.sleep(4)
    navigator.go_to_pose(0.43, -2.52, -1.57)
    navigator.request_payload(receive=True, drop=False)
    time.sleep(1)
    navigator.go_to_pose(-4.3, 2.89, -1.6)
    navigator.request_docker(rack=2)
    time.sleep(4)  
    navigator.request_payload(receive=False, drop=True)
    navigator.nav.lifecycleShutdown()
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()