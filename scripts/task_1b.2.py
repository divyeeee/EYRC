#!/usr/bin/python3
import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
import geometry_msgs.msg
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32MultiArray#, Int32MultiArrayLayout  # Added for publishing IDs and centers
from cv2 import aruco

def calculate_rectangle_area(coordinates):
    '''
    Description:    Function to calculate area or detected aruco

    Args:
        coordinates (list):     coordinates of detected aruco (4 set of (x,y) coordinates)

    Returns:
        area        (float):    area of detected aruco
        width       (float):    width of detected aruco
    '''
    area = None
    width = None
    x0=coordinates[0][0]
    x1=coordinates[1][0]
    x2=coordinates[2][0]
    x3=coordinates[3][0]
    y0=coordinates[0][1]
    y1=coordinates[1][1]
    y2=coordinates[2][1]
    y3=coordinates[3][1]
    print(f"coordinates in function, calculate rectangle area: {coordinates}")
    width=math.sqrt(((x0-x1)**2)+((y0-y1)**2))
    length=math.sqrt(((x1-x2)**2)+((y1-y2)**2))
    area=width*length
    print(f'area in calculate rectangle area function: {area}')
    return area, width


def detect_aruco(image):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''
    aruco_area_threshold = 1500
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])
    size_of_aruco_m = 0.15

    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []
    tvecs=[]
    rvecs=[]

    grayimg=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    aruco_dict  = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    corners,ids,reject= aruco.detectMarkers(grayimg,aruco_dict,parameters = parameters)
    print(f"corners in detect aruco function: {corners}")
    print(f"ids in detect aruco function: {ids}")
    if ids is not None:
        for corner, id in zip(corners, ids):
            i=1
            corner=np.reshape(corner,(4,2))
            corner=corner.astype(int)
            area,width=calculate_rectangle_area(corner)
            if area>=aruco_area_threshold:
                cx=(corner[0][0]+corner[1][0]+corner[2][0]+corner[3][0])/4
                cy=(corner[0][1]+corner[1][1]+corner[2][1]+corner[3][1])/4
                center=(cx,cy)
                print(f"one center from detect_aruco function: {center}")
                center_aruco_list.append(center)
                print(f'corner for pnp: {corner}, and its length: {len(corner)}')
                corner=[corner]
                print(f"new corner: {corner}")
                corner = np.array(corner, dtype=np.float32)
                print(f'after np.array(corner, dtype=np.float32): {corner}')
                rvec,tvec,rejected=cv2.aruco.estimatePoseSingleMarkers(corner, size_of_aruco_m, cam_mat, dist_mat)
                rot_mat = cv2.Rodrigues(np.array([0, np.pi, 0]))[0]
                cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 1)
                if rvec is not None:
            # Ensure rvec is in the expected shape
                    if rvec.shape == (1, 1, 3):
                        # Calculate the distance from the RGB camera to the ArUco marker
                        distance_from_rgb = np.linalg.norm(tvec)
                        tvecs.append(tvec)
                        rvecs.append(rvec)

                        # Calculate the angle of the pose estimated for the ArUco marker
                        roll, pitch, yaw = R.from_rotvec(rvec[0, 0]).as_euler('xyz',degrees=True)
                        
                        angle_aruco = yaw

                        # Append the calculated values to the respective lists
                        distance_from_rgb_list.append(distance_from_rgb)
                        angle_aruco_list.append(angle_aruco)
                        width_aruco_list.append(size_of_aruco_m)
                        print(f"type of ids: {type(ids)} and ids {ids} and id: {type(id)} and id {id}")
                        #sometimes i hate you numpy
                        # Heaven Knows I'm Miserable Now
                        i+=1
                        print(f"i :{i}")
                        # ids.append(id)

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids,tvecs,rvecs

# [Previous functions remain unchanged until class definition]

class aruco_tf(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')

        # Previous subscriptions
        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        # New publishers for ArUco IDs and centers
        self.id_publisher = self.create_publisher(Int32MultiArray, '/detected_aruco_ids', 10)
        self.center_publisher = self.create_publisher(Int32MultiArray, '/detected_aruco_centers', 10)

        # Previous initializations
        image_processing_rate = 0.2
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(image_processing_rate, self.process_image)
        
        self.cv_image = None
        self.depth_image = None

    # [Previous callback methods remain unchanged]

    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic. 
                        Use this function to receive image depth data and convert to CV2 image

        Args:
            data (Image):    Input depth image frame received from aligned depth camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############
        cv_image_depth = self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')
        self.depth_image= cv_image_depth

        # INSTRUCTIONS & HELP : 

        #	->  Use data variable to convert ROS Image message to CV2 Image type

        #   ->  HINT: You may use CvBridge to do the same

        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.
                        Use this function to receive raw image data and convert to CV2 image

        Args:
            data (Image):    Input coloured raw image frame received from image_raw camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############
        cv_image_color = self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')
        self.cv_image = cv_image_color
        # INSTRUCTIONS & HELP : 

        #	->  Use data variable to convert ROS Image message to CV2 Image type

        #   ->  HINT:   You may use CvBridge to do the same
        #               Check if you need any rotation or flipping image as input data maybe different than what you expect to be.
        #               You may use cv2 functions such as 'flip' and 'rotate' to do the same

        ############################################


    def process_image(self):
        try:
            if self.cv_image is None:
                self.get_logger().warn('No image received yet')
                return

            # Camera parameters
            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 640 
            centerCamY = 360
            focalX = 931.1829833984375
            focalY = 931.1829833984375

            # Detect ArUco markers
            center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids, tvecs, rvecs = detect_aruco(self.cv_image)

            # Sort the centers and IDs based on the x-coordinate of the centers
            sorted_centers_and_ids = sorted(zip(center_aruco_list, [id_[0] for id_ in ids]), key=lambda x: x[0][0])
            sorted_center_aruco_list, sorted_ids = zip(*sorted_centers_and_ids)

            # Publish detected IDs and centers
            if ids is not None and len(ids) > 0:
                id_msg = Int32MultiArray()
                center_msg = Int32MultiArray()
                id_msg.data = [int(id_) for id_ in sorted_ids]
                center_msg.data = [int(cx) for cx, cy in sorted_center_aruco_list] + [int(cy) for cx, cy in sorted_center_aruco_list]
                self.id_publisher.publish(id_msg)
                self.center_publisher.publish(center_msg)
                self.get_logger().info(f'Published ArUco IDs: {id_msg.data}')
                self.get_logger().info(f'Published ArUco Centers: {center_msg.data}')

            if len(center_aruco_list) != 0:
                for i, marker_id in enumerate(ids):
                    try:
                        # Get center coordinates
                        cx, cy = center_aruco_list[i]
                        distance_from_rgb = distance_from_rgb_list[i]

                        tvec = tvecs[i]
                        rvec = rvecs[i]

                        # Publish camera to marker transform
                        t = geometry_msgs.msg.TransformStamped()
                        t.header.stamp = self.get_clock().now().to_msg()
                        t.header.frame_id = 'camera_color_optical_frame'
                        t.child_frame_id = f'cam_{marker_id}'
                        
                        if marker_id==12:
                            t.transform.translation.x = float(tvec[0][0][0]) #0.01
                            t.transform.translation.y = float(tvec[0][0][1]-0.1) #-0.12
                            t.transform.translation.z = float(tvec[0][0][2]-0.3) #0.25
                        else:
                            t.transform.translation.x = float(tvec[0][0][0])
                            t.transform.translation.y = float(tvec[0][0][1])
                            t.transform.translation.z = float(tvec[0][0][2])


                        rot_mat = R.from_rotvec(rvec[0, 0])
                        rot_180_y = R.from_euler('xyz', [0, 180, 0], degrees=True)
                        q = (rot_mat*rot_180_y).as_quat()
                        t.transform.rotation.x = float(q[0])
                        t.transform.rotation.y = float(q[1])
                        t.transform.rotation.z = float(q[2])
                        t.transform.rotation.w = float(q[3])

                        self.br.sendTransform(t)

                        # Calculate corrected angle
                        angle_aruco = (0.788 * angle_aruco_list[i]) - ((angle_aruco_list[i] ** 2) / 3160)
                        yaw = angle_aruco

                        # Calculate position
                        depth = distance_from_rgb / 1000.0
                        x = distance_from_rgb * (sizeCamX - cx - centerCamX) / focalX
                        y = distance_from_rgb * (sizeCamY - cy - centerCamY) / focalY
                        z = distance_from_rgb

                        # Draw on image
                        cv2.circle(self.cv_image, (int(cx), int(cy)), 5, (0, 0, 255), 5)
                        cv2.putText(self.cv_image, f'ID= {marker_id[0]}', (int(cx), int(cy) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(self.cv_image, f'Center', (int(cx), int(cy) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        # Publish base to object transform
                        base_frame_id = 'base_link'
                        obj_frame_id = f'obj_{marker_id[0]}'

                        if (self.tf_buffer.can_transform('base_link', f'cam_{marker_id}', rclpy.time.Time())):
                            t = self.tf_buffer.lookup_transform('base_link', f'cam_{marker_id}', rclpy.time.Time())
                            t.header.stamp = self.get_clock().now().to_msg()
                            t.header.frame_id = base_frame_id
                            t.child_frame_id = obj_frame_id
                            self.br.sendTransform(t)
                            self.get_logger().info(f'Successfully received data for marker {marker_id}')

                    except Exception as e:
                        self.get_logger().error(f'Error processing marker {marker_id}: {str(e)}')
                        continue

            cv2.imshow("Aruco Detection", self.cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in process_image: {str(e)}')

# [Rest of the code remains unchanged]

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
                    You can find more on this here -> https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/
    '''

    main()