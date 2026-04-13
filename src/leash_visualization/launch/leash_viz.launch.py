"""Launch Go2 RViz visualization with leash direction arrow from Vicon mocap."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    go2_topic_arg = DeclareLaunchArgument(
        'go2_topic', default_value='/vicon/go2/go2',
        description='Vicon PoseStamped topic for Go2 rigid body',
    )
    leash_topic_arg = DeclareLaunchArgument(
        'leash_topic', default_value='/vicon/leash/leash',
        description='Vicon PoseStamped topic for leash rigid body',
    )

    go2_desc_dir = get_package_share_directory('go2_description')
    urdf_file = os.path.join(go2_desc_dir, 'urdf', 'go2_description.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    leash_viz_dir = get_package_share_directory('leash_visualization')
    rviz_config = os.path.join(leash_viz_dir, 'rviz', 'leash_viz.rviz')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
    )

    fixed_joint_publisher = Node(
        package='leash_visualization',
        executable='fixed_joint_publisher',
    )

    mocap_tf_broadcaster = Node(
        package='leash_visualization',
        executable='mocap_tf_broadcaster',
        parameters=[{
            'go2_topic': LaunchConfiguration('go2_topic'),
            'base_frame': 'base',
            'offset_x': 0.04,
            'offset_y': 0.0,
            'offset_z': -0.10,
        }],
    )

    leash_direction_node = Node(
        package='leash_visualization',
        executable='leash_direction_node',
        parameters=[{
            'go2_topic': LaunchConfiguration('go2_topic'),
            'leash_topic': LaunchConfiguration('leash_topic'),
            'base_frame': 'base',
            'offset_x': 0.04,
            'offset_y': 0.0,
            'offset_z': -0.10,
        }],
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        go2_topic_arg,
        leash_topic_arg,
        robot_state_publisher,
        fixed_joint_publisher,
        mocap_tf_broadcaster,
        leash_direction_node,
        rviz2,
    ])
