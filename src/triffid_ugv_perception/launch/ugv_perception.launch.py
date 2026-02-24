"""
Launch file for UGV perception node.
Usage (inside Docker container):
    ros2 launch triffid_ugv_perception ugv_perception.launch.py

Or with parameters:
    ros2 launch triffid_ugv_perception ugv_perception.launch.py \
        model_path:=yolo11n.pt \
        confidence_threshold:=0.4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.35'),
        DeclareLaunchArgument('target_frame', default_value='map'),
        DeclareLaunchArgument('api_url', default_value='https://crispres.com/wp-json/map-manager/v1/features'),
        DeclareLaunchArgument('publish_to_api', default_value='false'),
        DeclareLaunchArgument('gps_origin_lat', default_value='0.0'),
        DeclareLaunchArgument('gps_origin_lon', default_value='0.0'),

        Node(
            package='triffid_ugv_perception',
            executable='ugv_node',
            name='ugv_perception_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'target_frame': LaunchConfiguration('target_frame'),
            }],
        ),

        # GeoJSON bridge: converts detections → GeoJSON → TRIFFID API
        Node(
            package='triffid_ugv_perception',
            executable='geojson_bridge',
            name='geojson_bridge',
            output='screen',
            parameters=[{
                'api_url': LaunchConfiguration('api_url'),
                'publish_to_api': LaunchConfiguration('publish_to_api'),
                'gps_origin_lat': LaunchConfiguration('gps_origin_lat'),
                'gps_origin_lon': LaunchConfiguration('gps_origin_lon'),
            }],
        ),

        # Diagnostics: health monitoring + heartbeat
        Node(
            package='triffid_ugv_perception',
            executable='diagnostics',
            name='triffid_diagnostics',
            output='screen',
            parameters=[{
                'target_frame': LaunchConfiguration('target_frame'),
            }],
        ),
    ])
