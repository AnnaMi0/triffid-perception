from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Shared arguments
        DeclareLaunchArgument('model_path', default_value='/ws/best.pt'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.35'),
        DeclareLaunchArgument('yolo_imgsz', default_value='1280'),
        DeclareLaunchArgument('target_frame', default_value='b2/base_link'),

        # Configurable topic names (RealSense D435i front camera)
        DeclareLaunchArgument('rgb_image_topic',
                              default_value='/camera_front_435i/realsense_front_435i/color/image_raw'),
        DeclareLaunchArgument('depth_image_topic',
                              default_value='/camera_front_435i/realsense_front_435i/depth/image_rect_raw'),
        DeclareLaunchArgument('camera_info_topic',
                              default_value='/camera_front_435i/realsense_front_435i/color/camera_info'),
        DeclareLaunchArgument('depth_camera_info_topic',
                      default_value='/camera_front_435i/realsense_front_435i/depth/camera_info'),
        DeclareLaunchArgument('use_dummy_detections', default_value='false'),
        DeclareLaunchArgument('mqtt_enabled', default_value='true'),
        DeclareLaunchArgument('mqtt_host', default_value='localhost'),
        DeclareLaunchArgument('mqtt_port', default_value='1883'),
        DeclareLaunchArgument('mqtt_topic',
                              default_value='ugv/detections/front/geojson'),

        # UGV Perception Node
        Node(
            package='triffid_ugv_perception',
            executable='ugv_node',
            name='ugv_perception_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'yolo_imgsz': LaunchConfiguration('yolo_imgsz'),
                'target_frame': LaunchConfiguration('target_frame'),
                'rgb_image_topic': LaunchConfiguration('rgb_image_topic'),
                'depth_image_topic': LaunchConfiguration('depth_image_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'depth_camera_info_topic': LaunchConfiguration('depth_camera_info_topic'),
                'use_dummy_detections': LaunchConfiguration('use_dummy_detections'),
            }],
        ),

        # GeoJSON Bridge Node
        Node(
            package='triffid_ugv_perception',
            executable='geojson_bridge',
            name='geojson_bridge',
            output='screen',
            parameters=[{
                'mqtt_enabled': LaunchConfiguration('mqtt_enabled'),
                'mqtt_host': LaunchConfiguration('mqtt_host'),
                'mqtt_port': LaunchConfiguration('mqtt_port'),
                'mqtt_topic': LaunchConfiguration('mqtt_topic'),
            }],
        ),
    ])
