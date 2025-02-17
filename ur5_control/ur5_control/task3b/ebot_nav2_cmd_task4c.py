#!/usr/bin/env python3
'''
# Team ID:          1118
# Theme:            Logistic coBot
# Author List:      Saeesh , Sambhav, Anshul , Robin
# Filename:         ebot_nav.py
# Functions:        send_request,response_callback,start_navigation,request_payload_service,
#                   detach_gripper,request_docking,navigate_to_pose,main
# Global variables: None
'''
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf_transformations import quaternion_from_euler
from payload_service.srv import PayloadSW
from ebot_docking.srv import DockSw
from std_srvs.srv import Trigger
import time 
from linkattacher_msgs.srv import DetachLink


class ThirdJointJogClient(Node):
    '''
        Purpose:
        ---
        Initializes the ROS 2 client node for the passing3. It creates a service client 
        to interact with the "execute_third_joint_jog" service and waits for the service to become available.
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        node = ThirdJointJogClient()  # Called when an instance of the class is created.
        '''
    def __init__(self):
        super().__init__('third_joint_jog_client')
        self.client = self.create_client(Trigger, 'execute_third_joint_jog')
        
        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Third Joint Jog service to become available...')
    
    def send_request(self):
        self.get_logger().info('Sending request to Third Joint Jog service')
        request = Trigger.Request()
        # Send the request asynchronously
        future = self.client.call_async(request)
        return future
    
class JointJogClient(Node):
    '''
        Purpose:
        ---
        Initializes the ROS 2 client node for the joint jog service. It creates a service client 
        to interact with the "execute_second_joint_jog" service and waits for the service to become available.
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        node = JointJogClient()  # Called when an instance of the class is created.
        '''
    def __init__(self):
        super().__init__('joint_jog_client')
        self.client = self.create_client(Trigger, 'execute_second_joint_jog')
        
        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Second Joint Jog service to become available...')
    
    def send_request(self):
        self.get_logger().info('Sending request to Second Joint Jog service')
        request = Trigger.Request()
        future = self.client.call_async(request)
        return future

class JointJogServiceClient(Node):
    '''
        Purpose:
        ---
        Initializes the ROS 2 client node for the joint jog service. It creates a service client 
        to interact with the "execute_joint_jog" service and waits for the service to become available.
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        node = JointJogServiceClient()  # Called when an instance of the class is created.
        '''
    def __init__(self):
        super().__init__('joint_jog_service_client')

        # Create a service client for the "execute_joint_jog" service
        self.client = self.create_client(Trigger, 'execute_joint_jog')

        # Wait for the service to become available
        self.get_logger().info('Waiting for joint jog service...')
        self.client.wait_for_service()

        self.get_logger().info('Joint jog service available. Sending request.')

    def send_request(self):
        # Create and send a Trigger request
        request = Trigger.Request()

        # Asynchronous call to the service
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            # Get the response and log the results
            response = future.result()
            if response.success:
                self.get_logger().info(f"Service call succeeded: {response.message}")
            else:
                self.get_logger().warning(f"Service call failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed with exception: {e}")

class EBotNavigator(Node):
    '''
    Purpose:
    ---
    The `EBotNavigator` class is a ROS 2 Node that handles the navigation and manipulation of the EBot robot. 
    It integrates services for payload handling, docking, and gripper operations, and includes functionality 
    for joint jogging.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None
    
    Example call:
    ---
    This class is initialized when creating an instance of `EBotNavigator`, which will automatically initialize
    the associated services and clients.
    '''


    def __init__(self):
        '''
        Purpose:
        ---
        The constructor initializes the `EBotNavigator` node, sets up necessary service clients for payload control, 
        docking, gripper operations, and joint jogging, and prepares the robot's navigation system.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''
        super().__init__('ebot_navigator')
        self.navigator = BasicNavigator() 
        
        # Initialize the payload service client
        self.payload_req_cli = self.create_client(PayloadSW, '/payload_sw')
        self.dock_client = self.create_client(DockSw, '/dock_control')
        self.detach_client = self.create_client(DetachLink, '/GripperMagnetOFF')
        
        # Create both Joint Jog Service Clients
        self.joint_jog_client = JointJogServiceClient()
        self.second_joint_jog_client = JointJogClient()
        self.third_joint_jog_client = ThirdJointJogClient()



    def start_navigation(self):
        '''
        Purpose:
        ---
        The `start_navigation` method is used to initiate the robot’s navigation sequence, including movement to predefined poses,
        docking to specified racks, and performing arm manipulation tasks. It also integrates joint jogging for precise movements.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        
        Example call:
        ---
        To start the navigation sequence, call `start_navigation()` on an instance of the `EBotNavigator` class:
        `ebot_navigator.start_navigation()`
        '''
        self.navigator.lifecycleStartup()
        self.get_logger().info('Starting Navigation...')
        self.joint_jog_client.send_request()
        # Navigate to receive
        self.navigate_to_pose(0.3, -2.50, 3.14)
        self.request_docking(rack=1, orient_value=3.14, distance=0.01)    
        time.sleep(8)
        
        # Call arm manipulation service and joint jog service
        self.detach_gripper('box1')
        self.get_logger().info('Undocking....')
        self.request_docking(undock=True)
        time.sleep(15)

        # Call the second joint jog service
        self.second_joint_jog_client.send_request()
        # Wait for the service to complete

        self.navigate_to_pose(2.32, 2.65, -1.57)
        self.request_docking(rack=1, orient_value=-1.57, distance=0.06)
        time.sleep(8)
        self.request_payload_service(receive=False, drop=True, box_name='box1')
        time.sleep(1)
        self.get_logger().info('Undocking....')
        self.request_docking(undock=True)
        time.sleep(5)

        self.navigate_to_pose(-0.2, -2.43, 3.14)
        self.request_docking(rack=1, orient_value=3.14, distance=0.01)    
        time.sleep(8)
        self.detach_gripper('box2')

        #call the third joint jog service 
        self.third_joint_jog_client.send_request()


        self.navigate_to_pose(-4.5, 2.89, -1.57)
        # Request docking at conv 1
        self.request_docking(rack=1, orient_value=-1.57, distance=0.06)
        time.sleep(6)  
        self.request_payload_service(receive=False, drop=True, box_name='box2')

        self.get_logger().info('Undocking....')
        self.request_docking(undock=True)
        time.sleep(5)

        self.navigate_to_pose(0.5, -2.43, 3.14)
        self.request_docking(rack=1, orient_value=3.14, distance=0.01)    
        time.sleep(8)
        self.detach_gripper('box3')


        self.navigate_to_pose(2.32, 2.55, -1.57)
        self.request_docking(rack=1, orient_value=-1.57, distance=0.06)
        time.sleep(10)
        self.request_payload_service(receive=False, drop=True, box_name='box3')
        self.get_logger().info('Undocking....')
        self.request_docking(undock=True)
        time.sleep(5)

        self.navigator.lifecycleShutdown()

    def request_payload_service(self, receive, drop, box_name):
        """
        Purpose:
        ---
        This function handles requests to the payload service. It allows the robot to either receive or drop a payload based on
        the provided boolean flags. The function waits for the payload service to become available and then sends the appropriate
        request to the service.

        Input Arguments:
        ---
        receive (bool): 
            - If True, the robot requests to receive a payload.
            - If False, the robot will not request a payload to be received.
        
        drop (bool): 
            - If True, the robot requests to drop a payload.
            - If False, the robot will not request the payload to be dropped.

        Returns:
        ---
        Future: 
            The result of the asynchronous service call. This can be used to handle the service response once it's complete.

        Example call:
        ---
        To request a payload service (e.g., to drop a payload), call:
        `ebot_navigator.request_payload_service(receive=False, drop=True)`
        """
        # Validate box name
        valid_boxes = ['box1', 'box2', 'box3']
        if box_name not in valid_boxes:
            raise ValueError(f"box_name must be one of {valid_boxes}")
            
        # Wait for service availability
        while not self.payload_req_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Payload service not available, waiting again...')
        
        # Create and populate request
        req = PayloadSW.Request()
        req.receive = receive
        req.drop = drop
        req.box_name = box_name
        
        return self.payload_req_cli.call_async(req)
    
    def detach_gripper(self, model1_name):
        """
        Purpose:
        ---
        This function sends a request to the gripper control service to detach a specified model from the UR5 robotic arm's gripper.
        The function waits for the service to become available and then sends a request to detach the model from the gripper.

        Input Arguments:
        ---
        model1_name (str): 
            The name of the model to detach from the gripper.

        Returns:
        ---
        None: 
            The function doesn't return any value, but it logs the success of the detachment.

        Example call:
        ---
        To detach a model from the gripper, call:
        `ebot_navigator.detach_gripper("box1")`
        """
        gripper_control = self.create_client(DetachLink, '/GripperMagnetOFF')

        while not gripper_control.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gripper detach service not available, waiting again...')

        req = DetachLink.Request()
        req.model1_name = model1_name 
        req.link1_name = 'link'
        req.model2_name = 'ur5' 
        req.link2_name = 'wrist_3_link'  

        future = gripper_control.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Detached {model1_name} from UR5 gripper.")

    def request_docking(self, start=True, linear=True, orientation=True, 
                    distance=0.5, orient_value=-1.57, rack="1", undock=False):
        """
        Purpose:
        ---
        This function sends a request to the docking service to start or stop the docking procedure.
        It allows setting various parameters such as linear and orientation docking, target distance, orientation, and the rack number.

        Input Arguments:
        ---
        start (bool): 
            If True, the docking process is initiated.
        linear (bool): 
            If True, linear docking will be enabled.
        orientation (bool): 
            If True, orientation docking will be enabled.
        distance (float): 
            The target distance to reach during docking (default is 0.5 meters).
        orient_value (float): 
            The target orientation to achieve during docking (default is -1.57 radians).
        rack (str): 
            The rack number to dock to (default is "1").
        undock (bool): 
            If True, it initiates undocking instead of docking (default is False).

        Returns:
        ---
        Future: 
            The return value is a Future object, which represents the result of the asynchronous call to the docking service.

        Example call:
        ---
        To initiate docking, call:
        `ebot_navigator.request_docking(start=True, linear=True, orientation=True, rack="2")`
        """
        while not self.dock_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Docking service not available, waiting again...')

        req = DockSw.Request()
        req.startcmd = start
        req.linear_dock = linear
        req.orientation_dock = orientation
        req.distance = distance
        req.orientation = orient_value
        req.rack_no = str(rack)  # Ensure rack number is a string
        req.undocking = undock

        return self.dock_client.call_async(req)

    def navigate_to_pose(self, x, y, yaw):
        """
        Purpose:
        ---
        This function sends a goal pose to the robot and monitors its progress until the goal is either completed, canceled, or failed.

        Input Arguments:
        ---
        x (float): 
            The target x-coordinate in the map frame.
        y (float): 
            The target y-coordinate in the map frame.
        yaw (float): 
            The target orientation (yaw) in radians.

        Returns:
        ---
        None. The function performs navigation and logs the status of the task.

        Example call:
        ---
        `navigate_to_pose(1.0, 2.0, 1.57)`
        This would navigate the robot to position (1.0, 2.0) with a yaw of 1.57 radians (about 90 degrees).

        Workflow:
        ---
        - Creates a goal pose with the specified x, y, and yaw values.
        - Converts yaw to a quaternion for proper orientation.
        - Sends the goal pose to the navigator.
        - Monitors the progress, logging feedback on distance remaining.
        - Logs the result once the navigation task is complete.
        """
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        
        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        self.navigator.goToPose(goal_pose)
        i = 0
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback and i % 5 == 0:
                self.get_logger().info(f'Navigating to goal: {x}, {y}, {yaw}... ' +
                    f'Distance remaining: {feedback.distance_remaining:.2f} meters.')
            i += 1
            
        result = self.navigator.getResult()
        if result == "SUCCEEDED":
            self.get_logger().info('Goal succeeded!')
        elif result == "CANCELED":
            self.get_logger().info('Goal was canceled!')
        else:
            self.get_logger().warn('Goal failed!')

def main(args=None):

    """
    Purpose:
    ---
    This is the entry point for the robot's navigation process. It initializes the ROS 2 node, starts the robot's navigation, 
    and then shuts down the node once navigation is complete.

    Input Arguments:
    ---
    args (list, optional): 
        Command-line arguments to initialize ROS 2 (default is None). This is typically used to pass arguments to nodes
        if necessary, but in this case, it's left as None.

    Returns:
    ---
    None. This function is a standard entry point for ROS 2 nodes.

    Example call:
    ---
    `main()` 
    This function is automatically called when the script is executed.

    
    """
    rclpy.init(args=args)
    navigator = EBotNavigator()
    navigator.start_navigation()
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
