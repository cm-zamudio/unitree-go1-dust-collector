Undergraduate Research project - Fall 2025, University of Illinois Chicago

The project uses a **Unitree GO1 quadruped robot** equipped with a **vacuum attachment** to autonomously clean dirty spots on the floor. To detect areas requiring cleaning, an **Intel RealSense D435i camera** combined with **YOLO-based object detection** is mounted on the GO1.

The system is built using **ROS 2 Humble** and **Ubuntu 22.04**.

The sections below describe the steps required to replicate the project.

## Dependencies / Acknowledgements

This project uses the following external package:

- **realsense-d435-rtab-map-in-ROS2**  
  https://github.com/simonbogh/realsense-d435-rtab-map-in-ROS2  
  Used for RTAB-Map integration with the Intel RealSense D435 in ROS 2.

- **unitree_ros**
  https://github.com/snt-arg/unitree_ros
  Used for ROS2 integration with GO1 robot.

## Run

1) ros2 launch yolov8_ros2 camera_yolo.launch.py

2) ros2 launch rtabmap_launch rtabmap.launch.py  args:="--delete_db_on_start"  depth_topic:=/camera/camera/aligned_depth_to_color/image_raw  rgb_topic:=/camera/camera/color/image_raw  camera_info_topic:=/camera/camera/color/camera_info  approx_sync:=false  frame_id:=camera_link

3) ros2 run tf2_ros static_transform_publisher 0.1 0 0.03 0 0 0 base_link camera_link

4) ros2 launch go1_ekf ekf_launch.py

5) ros2 launch unitree_ros unitree_driver_launch.py wifi:=true

6) ros2 run go1_examples go_goal


