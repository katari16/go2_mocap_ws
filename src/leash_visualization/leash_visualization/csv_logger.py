"""Subscribe to /leash_vector and log to CSV.

Usage:
  ros2 run leash_visualization csv_logger
  ros2 run leash_visualization csv_logger --ros-args -p output_file:=/tmp/leash_data.csv
"""

import csv
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped


class CsvLogger(Node):

    def __init__(self):
        super().__init__('csv_logger')

        self.declare_parameter('output_file', f'leash_log_{int(time.time())}.csv')
        self.declare_parameter('topic', '/leash_vector')

        output_file = self.get_parameter('output_file').get_parameter_value().string_value
        topic = self.get_parameter('topic').get_parameter_value().string_value

        self.csv_file = open(output_file, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'x', 'y', 'z', 'magnitude'])
        self.count = 0

        self.sub = self.create_subscription(Vector3Stamped, topic, self.cb, 10)
        self.get_logger().info(f'Logging {topic} → {output_file}')

    def cb(self, msg):
        t = msg.header.stamp
        v = msg.vector
        mag = (v.x**2 + v.y**2 + v.z**2) ** 0.5
        self.writer.writerow([t.sec, t.nanosec, f'{v.x:.6f}', f'{v.y:.6f}', f'{v.z:.6f}', f'{mag:.6f}'])
        self.count += 1
        if self.count % 100 == 0:
            self.csv_file.flush()
            self.get_logger().info(f'{self.count} samples logged')

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f'Closed CSV ({self.count} samples)')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CsvLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
