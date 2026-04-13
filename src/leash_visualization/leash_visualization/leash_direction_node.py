"""Compute leash pull direction in URDF base frame, publish as RViz arrow marker.

Subscribes to two Vicon PoseStamped topics (robot + leash), computes
the direction vector from the URDF base origin to the leash in the
robot's body frame (with 180°-Z flip and offset correction).
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3Stamped
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_rotate(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[:3]


class LeashDirectionNode(Node):

    def __init__(self):
        super().__init__('leash_direction_node')

        self.declare_parameter('go2_topic', '/vicon/go2_1/go2_1')
        self.declare_parameter('leash_topic', '/vicon/string/string')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('marker_rate_hz', 30.0)
        self.declare_parameter('arrow_scale', 1.0)
        self.declare_parameter('offset_x', 0.04)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_z', -0.10)
        go2_topic = self.get_parameter('go2_topic').get_parameter_value().string_value
        leash_topic = self.get_parameter('leash_topic').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        rate = self.get_parameter('marker_rate_hz').get_parameter_value().double_value
        self.arrow_scale = self.get_parameter('arrow_scale').get_parameter_value().double_value
        self.offset_obj = np.array([
            self.get_parameter('offset_x').get_parameter_value().double_value,
            self.get_parameter('offset_y').get_parameter_value().double_value,
            self.get_parameter('offset_z').get_parameter_value().double_value,
        ])

        self.latest_robot = None
        self.latest_leash = None
        self.debug_counter = 0

        self.tf_broadcaster = TransformBroadcaster(self)
        self.sub_robot = self.create_subscription(PoseStamped, go2_topic, self.robot_cb, 10)
        self.sub_leash = self.create_subscription(PoseStamped, leash_topic, self.leash_cb, 10)
        self.pub_marker = self.create_publisher(Marker, '/leash_arrow', 10)
        self.pub_dir = self.create_publisher(PoseStamped, '/leash_direction', 10)
        self.pub_vec = self.create_publisher(Vector3Stamped, '/leash_vector', 10)
        self.timer = self.create_timer(1.0 / rate, self.timer_cb)

        self.get_logger().info(
            f'Leash direction: {go2_topic} + {leash_topic} → /leash_arrow'
        )

    def robot_cb(self, msg):
        self.latest_robot = msg

    def leash_cb(self, msg):
        self.latest_leash = msg

    def timer_cb(self):
        self.debug_counter += 1

        if self.latest_robot is None or self.latest_leash is None:
            if self.debug_counter % 90 == 0:
                self.get_logger().warn(
                    f'Waiting for data: robot={"OK" if self.latest_robot else "NONE"}, '
                    f'leash={"OK" if self.latest_leash else "NONE"}'
                )
            return

        # Extract positions and robot orientation
        pr = self.latest_robot.pose.position
        pl = self.latest_leash.pose.position
        qr = self.latest_robot.pose.orientation

        p_robot = np.array([pr.x, pr.y, pr.z])
        p_leash = np.array([pl.x, pl.y, pl.z])
        q_obj = np.array([qr.x, qr.y, qr.z, qr.w])

        # Publish leash position as TF so it's visible in RViz
        t_leash = TransformStamped()
        t_leash.header.stamp = self.get_clock().now().to_msg()
        t_leash.header.frame_id = 'map'
        t_leash.child_frame_id = 'leash_vicon'
        t_leash.transform.translation.x = p_leash[0]
        t_leash.transform.translation.y = p_leash[1]
        t_leash.transform.translation.z = p_leash[2]
        ql = self.latest_leash.pose.orientation
        t_leash.transform.rotation.x = ql.x
        t_leash.transform.rotation.y = ql.y
        t_leash.transform.rotation.z = ql.z
        t_leash.transform.rotation.w = ql.w
        self.tf_broadcaster.sendTransform(t_leash)

        # Vector from mocap robot origin to leash, in map frame
        v_map = p_leash - p_robot

        # Rotate into go2_obj frame
        v_obj = quat_rotate(quat_conjugate(q_obj), v_map)

        # Subtract offset (mocap origin → URDF base, in obj frame)
        # This is the leash vector in the go2_1_adjusted frame
        v_obj_from_base = v_obj - self.offset_obj

        vec_msg = Vector3Stamped()
        vec_msg.header.stamp = self.get_clock().now().to_msg()
        vec_msg.header.frame_id = 'go2_1_adjusted'
        vec_msg.vector.x = float(v_obj_from_base[0])
        vec_msg.vector.y = float(v_obj_from_base[1])
        vec_msg.vector.z = float(v_obj_from_base[2])
        self.pub_vec.publish(vec_msg)

        # 180°-Z flip: obj → URDF base frame
        v_base = np.array([-v_obj_from_base[0], -v_obj_from_base[1], v_obj_from_base[2]])

        length = np.linalg.norm(v_base)
        if length < 1e-4:
            return

        # Scale arrow
        v_arrow = v_base * self.arrow_scale

        if self.debug_counter % 90 == 0:
            self.get_logger().info(
                f'Arrow: base_frame len={length:.3f} '
                f'dir=[{v_base[0]/length:.2f}, {v_base[1]/length:.2f}, {v_base[2]/length:.2f}]'
            )

        # Publish arrow marker
        from geometry_msgs.msg import Point
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'leash'
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD

        start = Point(x=0.0, y=0.0, z=0.0)
        end = Point(x=float(v_arrow[0]), y=float(v_arrow[1]), z=float(v_arrow[2]))
        m.points = [start, end]

        m.scale.x = 0.02   # shaft diameter
        m.scale.y = 0.04   # head diameter
        m.scale.z = 0.05   # head length

        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.2
        m.color.a = 1.0

        m.lifetime.sec = 0
        m.lifetime.nanosec = int(2 * 10**8)  # 200ms

        self.pub_marker.publish(m)

        # Also publish as PoseStamped for logging
        dir_msg = PoseStamped()
        dir_msg.header = m.header
        unit = v_base / length
        dir_msg.pose.position.x = float(unit[0])
        dir_msg.pose.position.y = float(unit[1])
        dir_msg.pose.position.z = float(unit[2])
        self.pub_dir.publish(dir_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LeashDirectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
