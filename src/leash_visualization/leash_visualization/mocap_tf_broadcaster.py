"""Broadcast TF map → base from Vicon go2_obj pose.

Applies:
  1. 180°-Z rotation (mocap obj frame has x/y flipped vs URDF base)
  2. Translation offset from mocap rigid-body origin to URDF base origin
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2.  Format: (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q.  q format: (x, y, z, w)."""
    qv = np.array([v[0], v[1], v[2], 0.0])
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    rotated = quat_multiply(quat_multiply(q, qv), q_conj)
    return rotated[:3]


# 180° about Z: (x,y,z,w) = (0, 0, sin(90°), cos(90°)) = (0, 0, 1, 0)
Q_180Z = np.array([0.0, 0.0, 1.0, 0.0])


class MocapTfBroadcaster(Node):

    def __init__(self):
        super().__init__('mocap_tf_broadcaster')

        self.declare_parameter('go2_topic', '/vicon/go2_1/go2_1')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('offset_x', 0.04)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_z', -0.10)

        go2_topic = self.get_parameter('go2_topic').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.offset_obj = np.array([
            self.get_parameter('offset_x').get_parameter_value().double_value,
            self.get_parameter('offset_y').get_parameter_value().double_value,
            self.get_parameter('offset_z').get_parameter_value().double_value,
        ])

        self.tf_broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(PoseStamped, go2_topic, self.pose_cb, 10)
        self.logged_frame = False

        self.get_logger().info(
            f'Broadcasting TF map → {self.base_frame} from {go2_topic} '
            f'(offset in obj: {self.offset_obj})'
        )

    def pose_cb(self, msg):
        if not self.logged_frame:
            self.get_logger().info(
                f'Vicon msg frame_id: "{msg.header.frame_id}" (overriding to "map")'
            )
            self.logged_frame = True

        stamp = msg.header.stamp
        p = msg.pose.position
        q = msg.pose.orientation
        q_obj = np.array([q.x, q.y, q.z, q.w])

        # Orientation: base = obj rotated 180° about Z
        q_base = quat_multiply(q_obj, Q_180Z)

        # Position: base origin = obj origin + R(q_obj) * offset_in_obj
        p_obj = np.array([p.x, p.y, p.z])
        p_base = p_obj + quat_rotate(q_obj, self.offset_obj)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = self.base_frame
        t.transform.translation.x = p_base[0]
        t.transform.translation.y = p_base[1]
        t.transform.translation.z = p_base[2]
        t.transform.rotation.x = q_base[0]
        t.transform.rotation.y = q_base[1]
        t.transform.rotation.z = q_base[2]
        t.transform.rotation.w = q_base[3]

        # Also publish raw Vicon frame so the offset is visible in RViz
        t_raw = TransformStamped()
        t_raw.header.stamp = stamp
        t_raw.header.frame_id = 'map'
        t_raw.child_frame_id = 'go2_vicon'
        t_raw.transform.translation.x = p_obj[0]
        t_raw.transform.translation.y = p_obj[1]
        t_raw.transform.translation.z = p_obj[2]
        t_raw.transform.rotation.x = q_obj[0]
        t_raw.transform.rotation.y = q_obj[1]
        t_raw.transform.rotation.z = q_obj[2]
        t_raw.transform.rotation.w = q_obj[3]

        # Adjusted frame at offset position with original Vicon orientation
        t_adj = TransformStamped()
        t_adj.header.stamp = stamp
        t_adj.header.frame_id = 'map'
        t_adj.child_frame_id = 'go2_1_adjusted'
        t_adj.transform.translation.x = p_base[0]
        t_adj.transform.translation.y = p_base[1]
        t_adj.transform.translation.z = p_base[2]
        t_adj.transform.rotation.x = q_obj[0]
        t_adj.transform.rotation.y = q_obj[1]
        t_adj.transform.rotation.z = q_obj[2]
        t_adj.transform.rotation.w = q_obj[3]

        self.tf_broadcaster.sendTransform([t, t_raw, t_adj])


def main(args=None):
    rclpy.init(args=args)
    node = MocapTfBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
