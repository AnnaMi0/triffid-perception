"""
Launch file for UGV perception pipeline.

Usage (inside Docker container):
    ros2 launch triffid_ugv_perception ugv_perception.launch.py

With parameters:
    ros2 launch triffid_ugv_perception ugv_perception.launch.py \
        model_path:=yolo11n.pt \
        confidence_threshold:=0.4 \
        depth_grid_step_u:=32 \
        depth_grid_step_v:=24

Frame defaults (from rosbag):
    target_frame  = b2/base_link
    depth_frame   = f_depth_optical_frame
    rgb_frame     = f_oc_link
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # -- Shared arguments --
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.35'),
        DeclareLaunchArgument('target_frame', default_value='b2/base_link'),

        # Depth grid sampling resolution (lower = more points, slower)
        DeclareLaunchArgument('depth_grid_step_u', default_value='64'),
        DeclareLaunchArgument('depth_grid_step_v', default_value='48'),
        DeclareLaunchArgument('use_dummy_detections', default_value='false'),

        # -- Static TF publishers --
        # These transforms come from the robot URDF / tf_static in the
        # rosbag.  The bag records /tf_static with VOLATILE durability,
        # so the perception node often misses them due to DDS discovery
        # timing.  Publishing them here guarantees availability.

        # b2/base_link -> f_oc_link  (front USB RGB camera)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_base_to_rgb',
            arguments=[
                '--x', '0.3993', '--y', '0.0', '--z', '-0.0158',
                '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                '--frame-id', 'b2/base_link',
                '--child-frame-id', 'f_oc_link',
            ],
        ),

        # b2/base_link -> f_dc_link  (depth camera base, tilted ~45 deg down)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_base_to_dc',
            arguments=[
                '--x', '0.4216', '--y', '0.025', '--z', '0.0619',
                '--qx', '0', '--qy', '0.3827', '--qz', '0', '--qw', '0.9239',
                '--frame-id', 'b2/base_link',
                '--child-frame-id', 'f_dc_link',
            ],
        ),

        # f_dc_link -> f_depth_frame  (identity)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_dc_to_depth',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                '--frame-id', 'f_dc_link',
                '--child-frame-id', 'f_depth_frame',
            ],
        ),

        # f_depth_frame -> f_depth_optical_frame  (ROS optical convention)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_depth_to_optical',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5',
                '--frame-id', 'f_depth_frame',
                '--child-frame-id', 'f_depth_optical_frame',
            ],
        ),

        # -- UGV Perception Node --
        Node(
            package='triffid_ugv_perception',
            executable='ugv_node',
            name='ugv_perception_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'target_frame': LaunchConfiguration('target_frame'),
                'depth_grid_step_u': LaunchConfiguration('depth_grid_step_u'),
                'depth_grid_step_v': LaunchConfiguration('depth_grid_step_v'),
                'use_dummy_detections': LaunchConfiguration('use_dummy_detections'),
            }],
        ),
    ])
