#!/usr/bin/env python3
"""
Launch file to start the full system: RealSense camera, IMU odometry, vision pose estimation, and visualization.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare configurable launch arguments
    declare_enable_realsense = DeclareLaunchArgument('enable_realsense', default_value='true', description='Start RealSense camera')

    declare_depth_topic = DeclareLaunchArgument('depth_topic', default_value='/camera/camera/aligned_depth_to_color/image_raw')
    declare_color_topic = DeclareLaunchArgument('color_topic', default_value='/camera/camera/color/image_raw')
    declare_camera_info_topic = DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/color/camera_info')
    declare_tcp_pose_topic = DeclareLaunchArgument('tcp_pose_topic', default_value='/robot1_transform')

  
    # Include RealSense launch (if available)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'enable_depth': 'true',
            'enable_color': 'true',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_realsense'))
    )

    # Qwen service node
    qwen_node = Node(
        package='qwen_pkg',
        executable='qwen_service',
        name='qwen_service_node',
        output='screen',
        parameters=[
            {'camera1_topic': LaunchConfiguration('color_topic')},
            {'camera2_topic': LaunchConfiguration('color_topic')},
        ],
    )

    img_show_node = Node(
        package='qwen_pkg',
        executable='image_show',
        name='image_show_node',
        output='screen'
    )

    sam3_node = Node(
        package='sam3_pkg',
        executable='sam3_node',
        name='sam3_node',
        output='screen',
        parameters=[
            {'color_topic': LaunchConfiguration('color_topic')},
            {'depth_topic': LaunchConfiguration('depth_topic')},
            {'camera_info_topic': LaunchConfiguration('camera_info_topic')},
            {'tcp_pose_topic': LaunchConfiguration('tcp_pose_topic')},
        ]
    )


    ld = LaunchDescription()

    ld.add_action(declare_enable_realsense)
    ld.add_action(declare_depth_topic)
    ld.add_action(declare_color_topic)
    ld.add_action(declare_camera_info_topic)
    ld.add_action(declare_tcp_pose_topic)
    # ld.add_action(realsense_launch)
    ld.add_action(qwen_node)
    ld.add_action(img_show_node)
    ld.add_action(sam3_node)

    return ld


if __name__ == '__main__':
    generate_launch_description()
