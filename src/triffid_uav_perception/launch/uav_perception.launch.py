"""
Launch file for UAV perception node (skeleton).
Usage (inside Docker container):
    ros2 launch triffid_uav_perception uav_perception.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.35'),
        DeclareLaunchArgument('rgb_topic', default_value='/uav/camera/image_raw'),
        DeclareLaunchArgument('gps_topic', default_value='/uav/gps/fix'),

        Node(
            package='triffid_uav_perception',
            executable='uav_node',
            name='uav_perception_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'gps_topic': LaunchConfiguration('gps_topic'),
            }],
        ),
    ])
