#!/usr/bin/env python3

# Basic ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import ReliabilityPolicy, QoSProfile

# Executor and callback imports
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# ROS2 interfaces
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

# Image msg parser
from cv_bridge import CvBridge

# Vision model - YOLO-World
from ultralytics import YOLOWorld

# Others
import numpy as np
import open3d as o3d
import json
import torch


class Yolov8Node(Node):

    def __init__(self):
        super().__init__("yolov8_world_node")
        rclpy.logging.set_logger_level('yolov8_world_node', rclpy.logging.LoggingSeverity.INFO)

        # =========================
        # Parameters
        # =========================
        # YOLO-World model checkpoint
        self.declare_parameter("model", "/home/christian/colcon_ws/src/yolov8_ros2/yolov8_ros2/models/yolov8s-worldv2.pt")
        model = self.get_parameter("model").get_parameter_value().string_value

        # Device (cpu / cuda)
        self.declare_parameter("device", "cuda")
        self.device = self.get_parameter("device").get_parameter_value().string_value

        # Depth threshold in meters
        self.declare_parameter("depth_threshold", 1.2)
        self.depth_threshold = self.get_parameter("depth_threshold").get_parameter_value().double_value

        # Detection confidence threshold (lower for easier detection)
        self.declare_parameter("threshold", 0.25)
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value

        # Enable/disable YOLO
        self.declare_parameter("enable_yolo", True)
        self.enable_yolo = self.get_parameter("enable_yolo").get_parameter_value().bool_value

        # Open-vocabulary classes for YOLO-World (detailed prompts)
        self.declare_parameter(
            "world_classes",
            [
                 "crumpled white paper ball on the floor",
                 "crumpled paper ball",
                 "crumpled paper trash",
                 "small white trash on the floor",
                # "dust on the ground",
                # "trash on the ground",
                # "dirt on the ground"
                 "dust on ground",
                 "brown dust on ground",
                 "beige object on black ground",
                 "different colored object on black floor",
                 "orange traffic cone"
            ]
        )
        self.world_classes = list(
            self.get_parameter("world_classes").get_parameter_value().string_array_value
        )

        # =========================
        # Transformations (unchanged from your original)
        # =========================
        self.tf_world_to_camera = np.array([
            [-0.000, -1.000,  0.000, -0.017],
            [ 0.559,  0.000,  0.829, -0.272],
            [-0.829,  0.000,  0.559,  0.725],
            [ 0.000,  0.000,  0.000,  1.000]
        ])
        self.tf_camera_to_optical = np.array([
            [-0.003,  0.001,  1.000,  0.000],
            [-1.000, -0.002, -0.003,  0.015],
            [ 0.002, -1.000,  0.001, -0.000],
            [ 0.000,  0.000,  0.000,  1.000]
        ])
        self.tf_world_to_optical = np.matmul(self.tf_world_to_camera, self.tf_camera_to_optical)

        # =========================
        # Other inits
        # =========================
        self.group_1 = MutuallyExclusiveCallbackGroup()  # camera subscribers
        self.group_2 = MutuallyExclusiveCallbackGroup()  # vision timer

        self.cv_bridge = CvBridge()

        # Initialize YOLO-World model
        self.get_logger().info(f"Loading YOLO-World model: {model}")
        self.yolo = YOLOWorld(model)

        try:
            self.yolo.set_classes(self.world_classes)
            self.get_logger().info(f"YOLO-World classes: {self.world_classes}")
        except ModuleNotFoundError:
            self.get_logger().error(
                "YOLO-World text encoder (CLIP) is not installed. "
                "Run: python3 -m pip install 'git+https://github.com/ultralytics/CLIP.git'"
            )
            raise

        # Fuse layers if available (for speed)
        try:
            self.yolo.fuse()
        except AttributeError:
            self.get_logger().warn("Model does not support fuse(), continuing without fusion.")

        self.color_image_msg = None
        self.depth_image_msg = None
        self.camera_intrinsics = None
        self.pred_image_msg = Image()

        # Set clipping distance for background removal
        depth_scale = 0.001
        self.depth_threshold = self.depth_threshold / depth_scale

        # =========================
        # Publishers
        # =========================
        self._item_dict_pub = self.create_publisher(String, "/yolo/prediction/item_dict", 10)
        self._pred_pub = self.create_publisher(Image, "/yolo/prediction/image", 10)

        # =========================
        # Subscribers
        # =========================
        self._color_image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.color_image_callback,
            qos_profile_sensor_data,
            callback_group=self.group_1
        )
        self._depth_image_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_image_callback,
            qos_profile_sensor_data,
            callback_group=self.group_1
        )
        self._camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=self.group_1
        )

        # =========================
        # Timers
        # =========================
        self._vision_timer = self.create_timer(
            0.3,
            self.object_detection,
            callback_group=self.group_2
        )

    # =============================
    # Callbacks
    # =============================
    def color_image_callback(self, msg):
        self.color_image_msg = msg

    def depth_image_callback(self, msg):
        self.depth_image_msg = msg

    def camera_info_callback(self, msg):
        try:
            if self.camera_intrinsics is None:
                # Set intrinsics in o3d object
                self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
                self.camera_intrinsics.set_intrinsics(
                    msg.width,        # width
                    msg.height,       # height
                    msg.k[0],         # fx
                    msg.k[4],         # fy
                    msg.k[2],         # cx
                    msg.k[5]          # cy
                )
                self.get_logger().info('Camera intrinsics have been set!')
        except Exception as e:
            self.get_logger().error(f'camera_info_callback Error: {e}')

    # =============================
    # Background removal
    # =============================
    def bg_removal(self, color_img_msg: Image, depth_img_msg: Image):
        if self.color_image_msg is not None and self.depth_image_msg is not None:

            # Convert color image msg
            cv_color_image = self.cv_bridge.imgmsg_to_cv2(color_img_msg, desired_encoding='bgr8')
            np_color_image = np.array(cv_color_image, dtype=np.uint8)

            # Convert depth image msg
            cv_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
            np_depth_image = np.array(cv_depth_image, dtype=np.uint16)

            # Background removal (still computed for possible future use)
            grey_color = 153
            depth_image_3d = np.dstack((np_depth_image, np_depth_image, np_depth_image))
            bg_removed = np.where(
                (depth_image_3d > self.depth_threshold) | (depth_image_3d != depth_image_3d),
                grey_color,
                np_color_image
            )

            return bg_removed, np_color_image, np_depth_image

        self.get_logger().error("Background removal error, color or depth msg was None")
        return None, None, None

    # =============================
    # Object detection (YOLO-World)
    # =============================
    def object_detection(self):
        if not self.enable_yolo or self.color_image_msg is None or self.depth_image_msg is None or self.camera_intrinsics is None:
            return

        self.get_logger().debug("Successfully acquired color and depth image msgs")

        try:
            # Get images (we will detect on raw color image)
            bg_removed, np_color_image, np_depth_image = self.bg_removal(self.color_image_msg, self.depth_image_msg)
            if bg_removed is None:
                return
        except Exception as e:
            self.get_logger().error(f"Error during background removal: {str(e)}")
            return

        self.get_logger().debug("Successfully removed background")

        # Predict on ORIGINAL color image using YOLO-World
        try:
            results = self.yolo.predict(
                source=np_color_image,   # <--- use raw color image here
                show=False,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device,
                imgsz=640
            )
        except Exception as e:
            self.get_logger().error(f"YOLO-World prediction error: {str(e)}")
            return

        self.get_logger().debug("Successfully predicted")

        # Go through detections in prediction results
        for detection in results:
            # Publish prediction image (with boxes drawn)
            try:
                pred_img = detection.plot()
                self.pred_image_msg = self.cv_bridge.cv2_to_imgmsg(pred_img, encoding='passthrough')
                self._pred_pub.publish(self.pred_image_msg)
            except Exception as e:
                self.get_logger().error(f"Error plotting detection: {str(e)}")
                continue

            try:
                detection_class = detection.boxes.cls.cpu().numpy()
                detection_conf = detection.boxes.conf.cpu().numpy()
                n_objects = len(detection_class)

                # Debug: log raw detections
                if n_objects > 0:
                    class_names_dbg = [detection.names[int(c)] for c in detection_class]
                    self.get_logger().info(
                        f"YOLO-World raw detections: {list(zip(class_names_dbg, detection_conf))}"
                    )
                else:
                    self.get_logger().info("YOLO-World: no detections in this frame.")

                if n_objects == 0:
                    empty_dict_msg = String()
                    empty_dict_msg.data = json.dumps({})
                    self._item_dict_pub.publish(empty_dict_msg)
                    continue

                object_boxes = detection.boxes.xyxy.cpu().numpy()

                fx = self.camera_intrinsics.get_focal_length()[0]
                fy = self.camera_intrinsics.get_focal_length()[1]
                cx = self.camera_intrinsics.get_principal_point()[0]
                cy = self.camera_intrinsics.get_principal_point()[1]

                item_dict = {}
                for n in range(n_objects):
                    x1, y1, x2, y2 = object_boxes[n].astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(np_depth_image.shape[1] - 1, x2)
                    y2 = min(np_depth_image.shape[0] - 1, y2)

                    depth_roi = np_depth_image[y1:y2, x1:x2]
                    valid_depths = depth_roi[depth_roi > 0]
                    if len(valid_depths) == 0:
                        center_xyz = [0, 0, 0]
                        width_m = 0.0
                    else:
                        median_depth = np.median(valid_depths).astype(float)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        depth_meters = median_depth * 0.001

                        yaxis1 = (center_x - cx) * depth_meters / fx
                        zaxis1 = (center_y - cy) * depth_meters / fy
                        x = depth_meters + 0.099  # adjust to arm base

                        ycorr = -yaxis1 + 0.030  # offset adjust to center of object
                        zcorr = -zaxis1 + 0.050  # offset
                        center_xyz = [x, ycorr, zcorr]

                        # Convert width in pixels to meters (approximate)
                        x1_m = (x1 - cx) * depth_meters / fx
                        x2_m = (x2 - cx) * depth_meters / fx
                        width_m = abs(x2_m - x1_m)
                        width_m = width_m / 2 - 0.009  # adjust to gripper width

                    class_idx = int(detection_class[n])
                    class_name = detection.names[class_idx]

                    item_dict[f'item_{n}'] = {
                        'class': class_name,
                        'confidence': round(float(detection_conf[n]), 3),
                        'position_xyz': [round(val, 3) for val in center_xyz],
                        'estimated_width_m': round(width_m, 4)
                    }

                self.item_dict = item_dict
                self.item_dict_str = json.dumps(self.item_dict)
                class_names = [detection.names[int(item)] for item in detection_class]
                self.get_logger().debug(f"Yolo-World detected items: {class_names}")

                item_dict_msg = String()
                item_dict_msg.data = self.item_dict_str
                self._item_dict_pub.publish(item_dict_msg)

                self.get_logger().debug("Item dictionary successfully created and published")

            except Exception as e:
                self.get_logger().error(f"Error processing detection results: {str(e)}")
                continue

    # =============================
    # Shutdown
    # =============================
    def shutdown_callback(self):
        self.get_logger().warn("Shutting down...")


def main(args=None):
    rclpy.init(args=args)

    # Instantiate node class
    vision_node = Yolov8Node()

    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.shutdown_callback()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
