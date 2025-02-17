'''
# Team ID:          1114
# Theme:            Logistic coBot
# Author List:      Anuj, Yashita, Chirayu, Divye
# Filename:         new_client.py
# Functions:        go_to_pose,request_payload_service_chirayu,request_payload,
#                   request_docker, main
#                   
# Global variables: None
'''
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf_transformations import quaternion_from_euler
from payload_service.srv import PayloadSW, PickNplaceSW
from ebot_docking.srv import DockSw
from std_srvs.srv import SetBool
import time

class PayloadAndNavigation(Node):

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the ROS 2 client node for payload and navigation. It creates a service clients 
        to interact with the "picknplace", "dock_control" and "payload_sw" service and waits for the service to become available.
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        driver = PayloadAndNavigation()  # Called when an instance of the class is created.
        '''
        # -----------------------------
        # Initialize ROS 2 Node
        # -----------------------------
        # Logging an initialization message
        super().__init__("payload_and_navigation_node")
        #Initialising Navigation functionality of node using Basicnavigator()
        self.nav = BasicNavigator()

        # -----------------------------
        # Create Service Clients
        # -----------------------------
        # Service clients
        self.payload_client = self.create_client(PickNplaceSW, 'picknplace')
        self.dock_client = self.create_client(DockSw, '/dock_control')
        self.payload_ka_client = self.create_client(PayloadSW, '/payload_sw')

    def go_to_pose(self, x, y, yaw):
        '''
        Purpose:
        ---
        Creates and sends a Trigger request to the "execute_joint_jog" service. 
        The function makes an asynchronous call to the service and sets up a callback to handle the response.
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        self.send_request()  # This is called to send the request to the service.
        '''
        #<goal_pose>:<A PoseStamped message for the location of the goal for ebot>
        goal_pose = PoseStamped() 
        goal_pose.header.frame_id = 'map'#Coordinates are defined with respect to map frame
        goal_pose.header.stamp = self.nav.get_clock().now().to_msg()
        goal_pose.pose.position.x = x 
        goal_pose.pose.position.y = y

        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])#Final goal pose orientation in Quaternion form

        self.nav.goToPose(goal_pose)
        #While loop for logging information about the distance from goal and goal location given , using nav2 stack functions
        i = 0
        while not self.nav.isTaskComplete():
            feedback = self.nav.getFeedback()
            if feedback and i % 5 == 0:
                self.get_logger().info(f'Going to goal: {x}, {y}, {yaw}... ' +
                                       f'Distance remaining: {feedback.distance_remaining:.2f} meters.')
            i += 1
        #Handling success, Cancellation or failure of Task and logging on to terminal
        result = self.nav.getResult()
        if result == "SUCCEEDED":
            self.get_logger().info('Goal succeeded!')
        elif result == "CANCELED":
            self.get_logger().info('Goal was canceled!')
        else:
            self.get_logger().warn('Goal failed!')
    '''
    Purpose:
    ---
    <Sending a Setbool standard request to interact with the picknplace service>

    Input Arguments:
    ---
    < data >     :  [<Bool>]
      <setting a true value for the request to the picknplace service>
    
    Returns:
    ---

    Example call:
    ---
    <request_payload_service_chirayu()>
    '''
    def request_payload_service_chirayu(self,boxname,data=True):
        #Handling of service-client interaction
        while not self.payload_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for payload service to be available...')
        #<request> : <A Setbool Request type request consisting of Bool value Data>
        request = PickNplaceSW.Request()
        request.data = data
        request.box_name = boxname
        # request.drop = drop
        #Creation of Future for asynchronous communication
        future = self.payload_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        # Handling of Successful or Failed response to service call and logging of information to terminal
        if future.result():
            response = future.result()
            if response.success:
                self.get_logger().info(f'Payload service call succeeded: {response.message}')
            else:
                self.get_logger().warning(f'Payload service call failed: {response.message}')
        else:
            self.get_logger().error('Failed to call payload service.')
    '''
    Purpose:
    ---
    <This is function that sends a request for Drop to the Payload Service>

    Input Arguments:
    ---
    <receive> : [<Bool>]
     <Tells the Service if to receive  or not using Boolean Value, default value is True>
    <drop> : [<Bool>]
     <Tells the Service either to drop or not using a Boolean Value,default value is False>

    Example Call:
    ---
    <request_payload(receive=True,drop=False)>
    '''

    def request_payload(self,boxname, receive=False, drop=True):
        print("hi")
        #Handling of Client-service interaction and logging 
        while not self.payload_ka_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Payload service, Please Pickup the phone..........')
        #<reqi> : <PayloadSW request type object consisting of attributes to tell service to perform accordingly>
        reqi = PayloadSW.Request()
        reqi.receive = receive
        reqi.drop = drop 
        reqi.box_name = boxname
        #Printing for Error Handling     
        print(reqi)
        #Creation of future for Asynchronous communication
        future = self.payload_ka_client.call_async(reqi)
        # Running Node and blocking execution further until future produces result accordingly
        rclpy.spin_until_future_complete(self, future)
        # Handling of result and logging of information to terminal , depending on success and failure 
        if future.result():
            response = future.result()
            if response.success:
                self.get_logger().info(f'Payload service call succeeded: {response.message}')
            else:
                self.get_logger().warning(f'Payload service call failed: {response.message}')
        else:
            self.get_logger().error('Failed to call payload service.')

    '''
    Purpose:
    ---
    <A function to send a request to the docking service>
    
    Input Arguments:
    ---
    <start> : [<Bool>]
       <initiates communication with the Docking service,with a default True value>
    <linear> : [<Bool>]
        <Tells the service to use a linear docking approach>
    <orientation> : [<Bool>]
        <Tells the service to adjust orientation or not while docking>
    <distance> : [<Float>]
        <Minimum Docking distance for docking service to begin>
    <orient_value> : [<Float>]
        <Gives an idea of the yaw orientation>
    <undock> : [<Bool>]
        <Tells whether to undock or not> 
    
    Example call:
    ---
    client_name.request_docker(orient_value=3.10,rack=1)
    '''

    def request_docker(self, start=True, linear=True, orientation=True,
                       distance=0.65, orient_value=-1.57, rack="1", undock=False):
        #Handling of Service CLient interaction and logging 
        while not self.dock_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for docking service to be available...')
        #<req> : <DockSw request object containing essential information on whether to start , minimum distance for docking , rack to be docked to
        #and orienation values as well as the docking approach to be used
        req = DockSw.Request()
        req.startcmd = start
        req.linear_dock = linear
        req.orientation_dock = orientation
        req.distance = distance
        req.orientation = orient_value
        req.rack_no = str(rack)
        req.undocking = undock
        #Creation of future object for asynchronous and smooth communication
        future = self.dock_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        #Handling of future result and Logging to terminal correspondingly for Error Handling and analysis
        if future.result():
            response = future.result()
            if response.success:
                self.get_logger().info('Docking service call succeeded.')
            else:
                self.get_logger().warning('Docking service call failed.')
        else:
            self.get_logger().error('Failed to call docking service.')

def main(args=None):
    #Initialistion of Node
    rclpy.init(args=args)
    #Creation of PayloadandNavigation node to give driver all functionalities of said node
    #<driver> : <Child node of PayloadAndNavigation Node type for handling all functionalities>
    driver = PayloadAndNavigation()
    #Using lifecycleStartup functionality from BasicNavigator Class for adding lifecycle node functionality to driver
    driver.nav.lifecycleStartup()

    try:
        # Move to the first location
    #     driver.get_logger().info("Moving to the first location...")
    #     driver.go_to_pose(0.85, -2.40, 3.14)
    #     driver.request_docker(orient_value=3.10,rack=1)
    #     time.sleep(1.5)

        # Call the payload service
        driver.get_logger().info("Calling the payload service...")
        driver.request_payload_service_chirayu(boxname="1")

    #     # Move to the second location
    #     driver.get_logger().info("Moving to the second location...")
    #     driver.go_to_pose(2.32, 2.55, -1.70)
    #     driver.request_docker(orient_value=-1.70,rack=1)
    #     #Using sleep for smoother communication
    #     time.sleep(5)
    #     driver.request_payload(boxname="Box1")
    #     time.sleep(5)

    #     # Move to the third location
    #     driver.get_logger().info("Moving to the first location...")
    #     driver.go_to_pose(1.13, -2.40, 3.14)
    #     driver.request_docker(orient_value=3.10,rack=1)
    #     time.sleep(1.5)

    #     # Call the payload service
    #     driver.get_logger().info("Calling the payload service...")
    #     driver.request_payload_service_chirayu(boxname="2")

    #     # Move to the fourth location
    #     driver.get_logger().info("Moving to the second location...")
    #     driver.go_to_pose(-4.4,  2.89, -1.57)
    #     driver.request_docker(orient_value=-1.70,rack=1)
    #     #Using sleep for smoother communication
    #     time.sleep(5)
    #     driver.request_payload("Box2")
    #     time.sleep(5)

    #     # Move to the fifth location
    #     driver.get_logger().info("Moving to the first location...")
    #     driver.go_to_pose(1.13, -2.40, 3.14)
    #     driver.request_docker(orient_value=3.10,rack=1)
    #     time.sleep(1.5)

    #     # Call the payload service
    #     driver.get_logger().info("Calling the payload service...")
    #     driver.request_payload_service_chirayu(boxname="3")

    #     # Move to the sixth location
    #     driver.get_logger().info("Moving to the second location...")
    #     driver.go_to_pose(2.32, 2.55, -1.70)
    #     driver.request_docker(orient_value=-1.70,rack=1)
    #     #Using sleep for smoother communication
    #     time.sleep(5)
    #     driver.request_payload("Box3")
    #     time.sleep(5)
    # #Exception and error Handling 
    # except Exception as e:
    #     driver.get_logger().error(f"An error occurred: {e}")
    # #Shutting down of lifecycle and destroying node to stop further communication
    finally:
        driver.nav.lifecycleShutdown()
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
