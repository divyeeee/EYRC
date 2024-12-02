import rclpy
from rclpy.node import Node
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header, Int32MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import TransformListener, Buffer
import tf2_ros
import numpy as np
from linkattacher_msgs.srv import AttachLink, DetachLink
from servo_msgs.srv import ServoLink

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        self.twist_publisher = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.joint_publisher = self.create_publisher(JointJog, '/servo_node/delta_joint_cmds', 10)
        self.marker_subscriber = self.create_subscription(Int32MultiArray, '/detected_aruco_ids', self.update_markers, 10)
        self.joint_subscriber = self.create_subscription(JointState, '/joint_states', self.update_joints, 10)
        self.servo_client = self.create_client(Trigger, '/servo_node/start_servo')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_data = None
        self.joint_positions = [0.0] * 5
        self.target_angles_deg = [-263, -111, 143, 265, -80]
        self.target_angles_rad = [np.deg2rad(angle) for angle in self.target_angles_deg]
        self.joint_gain = 1.0
        self.target_positions = {
            'top_pos': [0.43, 0.1, 0.46],
            'ebot_pos': [0.5, 0.01, -0.1],
            'init_top': [0.16, 0.11, 0.47],
            'second_box': [-0.007, -0.42, 0.23],
            'return_top': [0.16, 0.09, 0.53],
            'last_pos': [-0.11, 0.25, 0.25]
        }
        self.initialize_servo()

    def initialize_servo(self):
        while not self.servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for servo service...')
        future = self.servo_client.call_async(Trigger.Request())
        future.add_done_callback(self.handle_servo_response)

    def handle_servo_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Servo initialized successfully')
            else:
                self.get_logger().warning('Servo initialization failed: ' + response.message)
        except Exception as e:
            self.get_logger().error(f'Service error: {e}')

    def update_markers(self, msg):
        self.marker_data = msg.data

    def update_joints(self, msg):
        self.joint_positions = msg.position

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

    def remove_object(self, name):
        client = self.create_client(ServoLink, '/SERVOLINK')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for servo link service...')

        request = ServoLink.Request()
        request.box_name = name
        request.box_link = 'link'
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

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
            for target, current in zip(target_rad, self.joint_positions):
                error = (target - current + np.pi) % (2 * np.pi) - np.pi
                error_deg = np.rad2deg(error)
                self.get_logger().info(f'Error for joint: {error_deg} degrees')

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

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    while controller.marker_data is None:
        rclpy.spin_once(controller)
    id = controller.marker_data
    print(f'THIS IS THE DETECTED ARUCO ID {controller.marker_data}')
    
    controller.move_joints(controller.target_angles_deg)
    
    transform = controller.tf_buffer.lookup_transform('base_link', 'obj_'+str(id[-1]), rclpy.time.Time())
    target_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
    controller.move_to_position(target_pose)
    controller.attach_object('box'+str(id[-1]))

    controller.move_to_position(controller.target_positions['top_pos'])
    controller.move_to_position(controller.target_positions['ebot_pos'])
    controller.detach_object('box'+str(id[-1]))
    controller.remove_object('box'+str(id[-1]))
    controller.move_to_position(controller.target_positions['init_top'])
    
    controller.move_to_position(controller.target_positions['second_box'])
    
    transform = controller.tf_buffer.lookup_transform('base_link', 'obj_'+str(id[0]), rclpy.time.Time())
    target_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
    controller.move_to_position(target_pose)
    controller.attach_object('box'+str(id[0]))

    controller.move_to_position(controller.target_positions['return_top'])
    controller.move_to_position(controller.target_positions['ebot_pos'])
    controller.detach_object('box'+str(id[0]))
    controller.remove_object('box'+str(id[0]))

    controller.move_to_position(controller.target_positions['top_pos'])
    controller.move_to_position(controller.target_positions['last_pos'])

    while controller.marker_data is None:
        rclpy.spin_once(controller)
    id = controller.marker_data
    object_num = str(id[-1])
    object_name = 'obj_'+str(object_num)
    print(f'THIS IS THE DETECTED ARUCO ID {controller.marker_data}')

    transform = controller.tf_buffer.lookup_transform('base_link', object_name, rclpy.time.Time())
    target_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
    controller.move_to_position(target_pose)
            
    controller.attach_object(f'box{object_num}')
    controller.move_to_position(controller.target_positions['top_pos'])
    controller.move_to_position(controller.target_positions['ebot_pos'])
    controller.detach_object(f'box{object_num}')
    controller.remove_object(f'box{object_num}')
    controller.move_to_position(controller.target_positions['top_pos'])

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()