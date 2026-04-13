import os
from glob import glob
from setuptools import setup

package_name = 'leash_visualization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'mocap_tf_broadcaster = leash_visualization.mocap_tf_broadcaster:main',
            'leash_direction_node = leash_visualization.leash_direction_node:main',
            'fixed_joint_publisher = leash_visualization.fixed_joint_publisher:main',
            'csv_logger = leash_visualization.csv_logger:main',
        ],
    },
)
