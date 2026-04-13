"""Publish fixed joint states for Go2 standing pose.

Matches UNITREE_GO2_CFG.init_state from the simulation config.
All non-leg joints (rotors, head, etc.) are set to 0.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

STANDING_POSE = {
    'FL_hip_joint': 0.1,  'FR_hip_joint': -0.1,
    'RL_hip_joint': 0.1,  'RR_hip_joint': -0.1,
    'FL_thigh_joint': 0.8,  'FR_thigh_joint': 0.8,
    'RL_thigh_joint': 1.0,  'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5,  'FR_calf_joint': -1.5,
    'RL_calf_joint': -1.5,  'RR_calf_joint': -1.5,
}

ZERO_JOINTS = [
    'Head_upper_joint', 'Head_lower_joint',
    'FL_hip_rotor_joint', 'FL_thigh_rotor_joint', 'FL_calf_rotor_joint',
    'FL_calflower_joint', 'FL_calflower1_joint', 'FL_foot_joint',
    'FR_hip_rotor_joint', 'FR_thigh_rotor_joint', 'FR_calf_rotor_joint',
    'FR_calflower_joint', 'FR_calflower1_joint', 'FR_foot_joint',
    'RL_hip_rotor_joint', 'RL_thigh_rotor_joint', 'RL_calf_rotor_joint',
    'RL_calflower_joint', 'RL_calflower1_joint', 'RL_foot_joint',
    'RR_hip_rotor_joint', 'RR_thigh_rotor_joint', 'RR_calf_rotor_joint',
    'RR_calflower_joint', 'RR_calflower1_joint', 'RR_foot_joint',
    'imu_joint', 'radar_joint', 'front_camera_joint',
]


class FixedJointPublisher(Node):

    def __init__(self):
        super().__init__('fixed_joint_publisher')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(1.0 / 50.0, self.publish)

        self.names = list(STANDING_POSE.keys()) + ZERO_JOINTS
        self.positions = [STANDING_POSE.get(n, 0.0) for n in self.names]

        self.get_logger().info(f'Publishing fixed joint states ({len(self.names)} joints)')

    def publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = self.positions
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FixedJointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
