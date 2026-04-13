"""Subscribe to /leash_vector and log to CSV.

Usage:
  ros2 run leash_visualization csv_logger
  ros2 run leash_visualization csv_logger --ros-args -p output_file:=/tmp/leash_data.csv
"""

import csv
import os
import time
import numpy as np
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

        self.output_file = output_file
        self.csv_file = open(output_file, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'x', 'y', 'z', 'magnitude'])
        self.count = 0
        self.samples = []

        self.sub = self.create_subscription(Vector3Stamped, topic, self.cb, 10)
        self.get_logger().info(f'Logging {topic} → {output_file}')

    def cb(self, msg):
        t = msg.header.stamp
        v = msg.vector
        mag = (v.x**2 + v.y**2 + v.z**2) ** 0.5
        self.writer.writerow([t.sec, t.nanosec, f'{v.x:.6f}', f'{v.y:.6f}', f'{v.z:.6f}', f'{mag:.6f}'])
        self.samples.append([v.x, v.y, v.z])
        self.count += 1
        if self.count % 100 == 0:
            self.csv_file.flush()
            self.get_logger().info(f'{self.count} samples logged')

    def destroy_node(self):
        self.csv_file.close()

        if self.samples:
            data = np.array(self.samples)
            median = np.median(data, axis=0)
            mag_median = np.linalg.norm(median)

            base, ext = os.path.splitext(self.output_file)
            summary_path = f'{base}_median{ext}'
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'z', 'magnitude'])
                writer.writerow([f'{median[0]:.6f}', f'{median[1]:.6f}',
                                 f'{median[2]:.6f}', f'{mag_median:.6f}'])

            self.get_logger().info(
                f'Median: x={median[0]:.4f} y={median[1]:.4f} z={median[2]:.4f} '
                f'mag={mag_median:.4f} → {summary_path}'
            )

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
