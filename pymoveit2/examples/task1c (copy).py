import rclpy
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5
import time
from linkattacher_msgs.srv import AttachLink, DetachLink 

class UR5moveit(Node):
    def __init__(self):
        super().__init__('ur5_control')

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur5.joint_names(),
            base_link_name=ur5.base_link_name(),
            end_effector_name=ur5.end_effector_name(),
            group_name=ur5.MOVE_GROUP_ARM,
        )


        self.callback_group = ReentrantCallbackGroup()


        self.attach_client = self.create_client(
            AttachLink, '/GripperMagnetON', callback_group=self.callback_group
        )
        self.detach_client = self.create_client(
            DetachLink, '/GripperMagnetOFF', callback_group=self.callback_group
        )


        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gripper Attach service...')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gripper Detach service...')

    def move_to_joint_goal(self, joint_positions):
        """Move UR5 to a specified joint configuration."""
        self.get_logger().info(f'Moving to joint positions: {joint_positions}')
        self.moveit2.move_to_configuration(joint_positions)
        time.sleep(3.2)  # Replace with actual feedback from MoveIt if possible

    def attach_box(self, box_name):
        """Attach the box to the gripper using GripperMagnetON service."""
        self.get_logger().info(f"Attaching box: {box_name}")
        req = AttachLink.Request()
        req.model1_name = box_name
        req.link1_name = 'link'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        future = self.attach_client.call_async(req)

        # Wait for future to complete, with a timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            self.get_logger().info(f"Box {box_name} attached successfully!")
        else:
            self.get_logger().error(f"Failed to attach box {box_name}.")

    def detach_box(self, box_name):
        """Detach the box from the gripper using GripperMagnetOFF service."""
        self.get_logger().info(f"Detaching box: {box_name}")
        req = DetachLink.Request()
        req.model1_name = box_name
        req.link1_name = 'link'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        future = self.detach_client.call_async(req)

        # Wait for future to complete, with a timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            self.get_logger().info(f"Box {box_name} detached successfully!")
        else:
            self.get_logger().error(f"Failed to detach box {box_name}.")

def main():
    rclpy.init()
    ur5_control = UR5moveit()
    
    P1 = [-1.3439, -1.50098, 1.37881, -3.00197, -1.78024, 3.1765]  # Position 1
    P2 = [-0.436332, -0.463786, 1.13446, -2.22402, -1.5708, 2.70526]  # Position 2
    P3 = [0.366519, -0.335066, 0.855211, -2.07694, -1.55334, 3.50811]  # Position 3
    drop = [0.0,-2.28638, -0.785398, -3.21141 , -1.50098, 3.14159]  # Pre-drop position
    drop1 = [0.0174533, -2.33874, -0.506145, -3.4383, -1.50098, 3.14159]
    d3 = [0.0, -1.78024, -1.72788, -2.77507, -1.55334 ,3.14159]
    box_name = 'box1'  # Box model name from ArUco TF
    ur5_control.get_logger().info("Moving to P1")
    ur5_control.move_to_joint_goal(P1)
    ur5_control.attach_box(box_name)
    ur5_control.get_logger().info("Moving to drop position")
    ur5_control.move_to_joint_goal(drop)
    ur5_control.detach_box(box_name)
    box_name = 'box49'
    ur5_control.get_logger().info("Moving to P2")
    ur5_control.move_to_joint_goal(P2)
    ur5_control.attach_box(box_name)
    ur5_control.get_logger().info("Moving to drop position")
    ur5_control.move_to_joint_goal(drop1)
    ur5_control.detach_box(box_name)
    box_name = 'box3'
    ur5_control.get_logger().info("Moving to P3")
    ur5_control.move_to_joint_goal(P3)
    ur5_control.attach_box(box_name)
    ur5_control.get_logger().info("Moving to drop position")
    ur5_control.move_to_joint_goal(d3)
    ur5_control.detach_box(box_name)
    ur5_control.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()