#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import json
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def yaw_from_quaternion(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class Go1MoveForward(Node):
    def __init__(self):
        super().__init__('person_approach_integrated')

        # params
        self.declare_parameter('standoff_distance', 0.2)
        self.declare_parameter('max_speed', 0.15) #changed from 0.20
        self.declare_parameter('max_turn', 0.5) #changed from 0.6
        self.declare_parameter('linear_kp', 0.5) #changed from 0.5
        self.declare_parameter('angular_kp', 1.0)
        self.declare_parameter('max_travel', 3.0)

        self.standoff_distance = self.get_parameter('standoff_distance').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.max_turn = self.get_parameter('max_turn').get_parameter_value().double_value
        self.linear_kp = self.get_parameter('linear_kp').get_parameter_value().double_value
        self.angular_kp = self.get_parameter('angular_kp').get_parameter_value().double_value
        self.max_travel = self.get_parameter('max_travel').get_parameter_value().double_value

        # odom
        self.current_odom_x = None
        self.current_odom_y = None
        self.current_yaw = None

        # safety start
        self.start_odom_x = None
        self.start_odom_y = None

        # goal state
        self.have_goal = False
        self.reached_goal = False

        # latched stuff when YOLO fires
        self.desired_forward_dist = 0.0     # how far we want to walk, from YOLO x - standoff
        self.desired_heading = None         # world heading to face toward the person
        self.forward_integrated = 0.0       # how far we think we've moved (integrated cmd)
        self.dt = 0.1                       # timer period

        # ros
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom/filtered',
            self.odom_callback,
            10
        )

        self.det_sub = self.create_subscription(
            String,
            '/yolo/prediction/item_dict',
            self.detection_callback,
            10
        )

        # 10 Hz
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("person_approach_integrated started (single-shot, side targets, distance by integration).")

    def odom_callback(self, msg: Odometry):
        self.current_odom_x = msg.pose.pose.position.x
        self.current_odom_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if self.start_odom_x is None:
            self.start_odom_x = self.current_odom_x
            self.start_odom_y = self.current_odom_y
            self.get_logger().info("Approach start odom recorded.")

    def detection_callback(self, msg: String):
        # single-shot
        if self.have_goal:
            return

        if self.current_yaw is None:
            self.get_logger().warn("Got detection but no odom yet; skipping.")
            return

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Could not parse detection JSON.")
            return

        for _, item_info in data.items():
            pos_xyz = item_info.get('position_xyz', None)
            if pos_xyz is None or len(pos_xyz) < 2:
                continue

            rel_x = float(pos_xyz[0])  # forward in camera/robot frame
            rel_y = float(pos_xyz[1])  # left/right in robot frame

            # how far forward we want to travel
            desired_forward = rel_x - self.standoff_distance
            if desired_forward < 0.0:
                desired_forward = 0.0

            # angle to the person in the ROBOT frame
            angle_robot = math.atan2(rel_y, rel_x)
            # convert to world: current_yaw + that offset
            desired_heading_world = self.current_yaw + angle_robot

            # normalize desired heading
            while desired_heading_world > math.pi:
                desired_heading_world -= 2.0 * math.pi
            while desired_heading_world < -math.pi:
                desired_heading_world += 2.0 * math.pi

            self.desired_forward_dist = desired_forward
            self.desired_heading = desired_heading_world
            self.forward_integrated = 0.0  # reset counter
            self.have_goal = True

            self.get_logger().info(
                f"Latched person rel=({rel_x:.2f}, {rel_y:.2f}) -> walk {desired_forward:.2f} m toward angle {math.degrees(angle_robot):.1f}° in robot frame."
            )
            break

    def control_loop(self):
        twist = Twist()

        # need odom for safety and heading
        if self.current_odom_x is None or self.current_odom_y is None or self.current_yaw is None:
            self.cmd_pub.publish(twist)
            return

        # safety against runaway
        dx = self.current_odom_x - self.start_odom_x
        dy = self.current_odom_y - self.start_odom_y
        if math.sqrt(dx*dx + dy*dy) > self.max_travel:
            self.get_logger().info("Max travel reached, stopping.")
            self.reached_goal = True
            self.cmd_pub.publish(Twist())
            return

        # no goal yet
        if not self.have_goal:
            self.cmd_pub.publish(twist)
            return

        # already done
        if self.reached_goal:
            self.cmd_pub.publish(twist)
            return

        # how much is left to go based on our own integration
        remain = self.desired_forward_dist - self.forward_integrated

        if remain <= 0.01:   # within 1 cm of what we wanted
            self.get_logger().info("Reached integrated forward distance. Stopping.")
            self.reached_goal = True
            self.cmd_pub.publish(Twist())
            return

        # heading control toward latched heading
        heading_error = self.desired_heading - self.current_yaw
        while heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        while heading_error < -math.pi:
            heading_error += 2.0 * math.pi

        ang_cmd = self.angular_kp * heading_error
        ang_cmd = max(min(ang_cmd, self.max_turn), -self.max_turn)

        # forward command based on how much is left
        fwd_cmd = self.linear_kp * remain
        if fwd_cmd < 0.0:
            fwd_cmd = 0.0
        if fwd_cmd > self.max_speed:
            fwd_cmd = self.max_speed

        twist.linear.x = fwd_cmd
        twist.angular.z = ang_cmd

        # publish first
        self.cmd_pub.publish(twist)

        # NOW integrate what we actually sent
        self.forward_integrated += fwd_cmd * self.dt


def main(args=None):
    rclpy.init(args=args)
    node = Go1MoveForward()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down person_approach_integrated.")
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
