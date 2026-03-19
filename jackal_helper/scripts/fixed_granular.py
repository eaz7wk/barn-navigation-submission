#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction
import actionlib
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header


class TemplateType(object):
    MOVE_STRAIGHT = 0
    STEER_LEFT = 1
    STEER_RIGHT = 2


class RobotGeometry(object):
    def __init__(self):
        self.length = 0.45
        self.width = 0.39
        self.half_length = self.length / 2.0
        self.half_width = self.width / 2.0
        self.radius = math.hypot(self.half_length, self.half_width)


class MotionTube(object):
    def __init__(self, template_type, v, w, T, samples, beam_indices, arc_len):
        self.template_type = template_type
        self.v = v
        self.w = w
        self.T = T
        self.samples = samples
        self.beam_indices = beam_indices
        self.arc_len = arc_len

        self.cost = float("inf")
        self.is_feasible = False
        self.goal_progress = 0.0
        self.min_clearance = float("inf")
        self.heading_after = 0.0
        self.obstacle_penalty = 0.0
        self.recovery_bias = 0.0
        
        self.left_clearance = float("inf")
        self.right_clearance = float("inf")
        self.center_balance = 0.0


class GoalOrientedMotionTubePlanner(object):
    def __init__(self):
        rospy.init_node("goal_oriented_motion_tube_planner_v25", anonymous=True)
        rospy.logwarn("DEBUG: fixed_granular_v25.py loaded – BARN stable + smarter recovery")

        self.latest_scan = None
        self.current_pose = None
        self.recovery_state = "NONE"

        self.sensor_config = {
            "min_angle": -np.pi,
            "max_angle": np.pi,
            "angle_increment": 0.01,
            "range_min": 0.1,
            "range_max": 30.0,
            "num_beams": 0,
        }

        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.planning_enabled = False

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.marker_pub = rospy.Publisher("/motion_tubes", MarkerArray, queue_size=1)

        self.try_laser_topics()
        self.try_odom_topics()

        self.robot = RobotGeometry()

        self.max_v = float(rospy.get_param("~max_v", 1.2))
        self.max_w = float(rospy.get_param("~max_w", 1.2))

        self.velocity_layers = rospy.get_param("~velocity_layers", [0.25, 0.55, 0.9, 1.2])
        self.velocity_layers = [min(float(v), self.max_v) for v in self.velocity_layers]

        n_ang = int(rospy.get_param("~num_angular_samples", 25))
        self.angular_rates = np.linspace(-self.max_w, self.max_w, n_ang)

        self.base_to_laser_yaw = float(rospy.get_param("~base_to_laser_yaw", 0.0))
        self.deadband_w = float(rospy.get_param("~deadband_w", 0.05))

        self.time_horizons = rospy.get_param("~time_horizons", [0.8, 1.4, 2.0])
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.45))

        self.loop_dt = float(rospy.get_param("~loop_dt", 0.15))
        self.scan_timeout = float(rospy.get_param("~scan_timeout", max(0.20, 1.5 * self.loop_dt)))
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.50))

        self.use_fwd_slowdown = bool(rospy.get_param("~use_fwd_slowdown", True))
        self.fwd_slow_half_angle_deg = float(rospy.get_param("~fwd_slow_half_angle_deg", 25.0))
        self.fwd_slow_gain = float(rospy.get_param("~fwd_slow_gain", 0.40))
        self.min_forward_scale = float(rospy.get_param("~min_forward_scale", 0.20))

        self.enable_recovery_scan = bool(rospy.get_param("~enable_recovery_scan", True))

        # pruning
        self.arc_len_merge_tol = float(rospy.get_param("~arc_len_merge_tol", 0.15))
        self.short_arc_threshold = float(rospy.get_param("~short_arc_threshold", 0.50))
        self.short_tube_keep_per_side = int(rospy.get_param("~short_tube_keep_per_side", 5))

        # clearance
        self.w_clearance = float(rospy.get_param("~w_clearance", 10.0))
        self.clearance_safe_dist = float(rospy.get_param("~clearance_safe_dist", 0.22))

        # strong long-tube priority
        self.long_tube_rel_ratio = float(rospy.get_param("~long_tube_rel_ratio", 0.75))
        
        # centering / lateral clearance
        self.w_center_balance = float(rospy.get_param("~w_center_balance", 6.0))
        self.w_side_clearance = float(rospy.get_param("~w_side_clearance", 8.0))
        self.side_clearance_safe_dist = float(rospy.get_param("~side_clearance_safe_dist", 0.14))

        # recovery basic
        self.recovery_backup_dist = float(rospy.get_param("~recovery_backup_dist", 0.20))
        self.recovery_backup_v = float(rospy.get_param("~recovery_backup_v", -0.10))
        self.recovery_backup_timeout = float(rospy.get_param("~recovery_backup_timeout", 3.0))
        self.recovery_backup_min_time = float(rospy.get_param("~recovery_backup_min_time", 1.0))

        self.recovery_scan_step_deg = float(rospy.get_param("~recovery_scan_step_deg", 35.0))
        self.recovery_scan_max_deg = float(rospy.get_param("~recovery_scan_max_deg", 210.0))
        self.recovery_yaw_tol_deg = float(rospy.get_param("~recovery_yaw_tol_deg", 3.0))
        self.recovery_w = float(rospy.get_param("~recovery_w", 0.55))
        self.recovery_min_pause = float(rospy.get_param("~recovery_min_pause", 0.08))

        # recovery goal suppress after exit
        self.recovery_goal_suppress_cycles = int(rospy.get_param("~recovery_goal_suppress_cycles", 8))
        self.recovery_goal_suppress_counter = 0

        # stuck detection
        self.stuck_window_sec = float(rospy.get_param("~stuck_window_sec", 2.0))
        self.stuck_min_progress = float(rospy.get_param("~stuck_min_progress", 0.10))
        self.stuck_min_motion = float(rospy.get_param("~stuck_min_motion", 0.18))
        self.stuck_confirm_count = int(rospy.get_param("~stuck_confirm_count", 3))

        self.progress_history = []
        self.stuck_counter = 0

        # recovery temporary goal
        self.recovery_goal_xy = None
        self.recovery_goal_radius = float(rospy.get_param("~recovery_goal_radius", 0.60))
        self.recovery_goal_max_dist = float(rospy.get_param("~recovery_goal_max_dist", 1.20))
        self.recovery_goal_min_dist = float(rospy.get_param("~recovery_goal_min_dist", 0.50))

        # wall-following bias during recovery
        self.recovery_follow_mode = "NONE"  # NONE / LEFT / RIGHT
        self.wall_follow_ref_dist = float(rospy.get_param("~wall_follow_ref_dist", 0.32))
        self.w_wall_follow = float(rospy.get_param("~w_wall_follow", 4.0))
        self.w_escape_heading = float(rospy.get_param("~w_escape_heading", 6.0))
        self.w_escape_open = float(rospy.get_param("~w_escape_open", 5.0))
        self.w_escape_leave = float(rospy.get_param("~w_escape_leave", 4.0))

        # recovery memory
        self.recovery_history = []
        self.recovery_memory_radius = float(rospy.get_param("~recovery_memory_radius", 1.0))
        self.recovery_memory_decay_sec = float(rospy.get_param("~recovery_memory_decay_sec", 12.0))
        self.w_recovery_memory = float(rospy.get_param("~w_recovery_memory", 4.0))

        # escape lifecycle
        self.escape_follow_timeout = float(rospy.get_param("~escape_follow_timeout", 3.5))
        self.escape_follow_enter_t = 0.0
        self.escape_start_xy = None
        self.escape_min_leave_dist = float(rospy.get_param("~escape_min_leave_dist", 0.60))

        # recovery state vars
        self.recovery_enter_t = 0.0
        self.recovery_backup_start_xy = None
        self.recovery_backup_start_t = 0.0
        self.recovery_base_yaw = 0.0
        self.recovery_target_yaw = 0.0
        self.recovery_scan_direction = 0
        self.recovery_current_offset_deg = 0.0
        self.recovery_last_switch_t = 0.0

        # planner state
        self.current_goal = None
        self.motion_tubes = []
        self.selected_tube = None
        self.min_cost = 0.0
        self.max_cost = 1.0

        self.goal_start_time = None
        self._ranges_clipped = None
        self._fwd_cache = float("inf")
        self.no_feasible = False

        self._tube_configs = self._build_pruned_tube_configs()
        rospy.loginfo(
            "Pruned tube configs: %d (from %d raw)",
            len(self._tube_configs),
            len(self.velocity_layers) * len(self.angular_rates) * len(self.time_horizons),
        )

        self.timer = rospy.Timer(rospy.Duration(self.loop_dt), self.planning_cycle)
        self.diag_timer = rospy.Timer(rospy.Duration(3.0), self.print_diagnostics)

        rospy.loginfo(
            "Planner v2.5 initialized (dt=%.3fs, scan_timeout=%.3fs, half_width=%.3fm)",
            self.loop_dt,
            self.scan_timeout,
            self.robot.half_width,
        )

        self.action_server = actionlib.SimpleActionServer(
            "/move_base", MoveBaseAction, self.goal_callback, False
        )
        self.action_server.start()
        rospy.loginfo("Action server started – ready to accept goals")

    # =====================================================================
    # Tube config generation
    # =====================================================================
    def _build_pruned_tube_configs(self):
        raw = []
        for T in self.time_horizons:
            for v in self.velocity_layers:
                for w in self.angular_rates:
                    arc = abs(v) * T
                    raw.append((v, w, T, arc))

        raw.sort(key=lambda x: x[3])

        groups = []
        for item in raw:
            if not groups or abs(item[3] - groups[-1][0][3]) > self.arc_len_merge_tol:
                groups.append([item])
            else:
                groups[-1].append(item)

        merged = []
        for grp in groups:
            by_w = {}
            for (v, w, T, arc) in grp:
                key = round(w, 4)
                if key not in by_w or T > by_w[key][2]:
                    by_w[key] = (v, w, T, arc)
            merged.extend(by_w.values())

        configs = []
        for (v, w, T, arc) in merged:
            configs.append((v, w, T, arc, arc < self.short_arc_threshold))

        short_configs = [c for c in configs if c[4]]
        long_configs = [(c[0], c[1], c[2]) for c in configs if not c[4]]

        short_by_vt = {}
        for (v, w, T, arc, _) in short_configs:
            key = (round(v, 4), round(T, 4))
            short_by_vt.setdefault(key, []).append((v, w, T))

        N = self.short_tube_keep_per_side
        for key, items in short_by_vt.items():
            items.sort(key=lambda x: x[1])
            right_side = items[:N]
            left_side = items[-N:]
            kept = list(set(right_side + left_side))
            long_configs.extend(kept)

        rospy.loginfo(
            "Tube pruning: %d short kept (from %d), %d long, %d total",
            len(long_configs) - len([c for c in configs if not c[4]]),
            len(short_configs),
            len([c for c in configs if not c[4]]),
            len(long_configs),
        )
        return long_configs

    # --------------------- topic selection ---------------------
    def try_laser_topics(self):
        for topic in ["/front/scan", "/laser/scan", "/base_scan", "/scan"]:
            try:
                rospy.loginfo("Trying laser topic: %s", topic)
                rospy.wait_for_message(topic, LaserScan, timeout=2.0)
                self.scan_sub = rospy.Subscriber(topic, LaserScan, self.scan_callback, queue_size=1)
                rospy.loginfo("Using laser topic: %s", topic)
                return
            except Exception:
                continue
        rospy.logwarn("No LaserScan topic found")

    def try_odom_topics(self):
        for topic in ["/odometry/filtered", "/odom", "/jackal/odom", "/robot/odom"]:
            try:
                rospy.loginfo("Trying odom topic: %s", topic)
                rospy.wait_for_message(topic, Odometry, timeout=2.0)
                self.odom_sub = rospy.Subscriber(topic, Odometry, self.odom_callback, queue_size=1)
                rospy.loginfo("Using odom topic: %s", topic)
                return
            except Exception:
                continue
        rospy.logwarn("No Odometry topic found")

    # --------------------- callbacks ---------------------
    def scan_callback(self, msg):
        self.latest_scan = msg
        self.last_scan_time = rospy.get_time()

        if self.sensor_config["num_beams"] == 0:
            inc = msg.angle_increment if abs(msg.angle_increment) > 1e-9 else 1e-3
            self.sensor_config.update(
                {
                    "min_angle": msg.angle_min,
                    "max_angle": msg.angle_max,
                    "angle_increment": inc,
                    "range_min": msg.range_min,
                    "range_max": msg.range_max,
                    "num_beams": len(msg.ranges),
                }
            )
            self.planning_enabled = True
            rospy.loginfo(
                "Laser: %d beams, FOV %.1f°, inc=%.5f rad, r=[%.2f, %.2f]",
                len(msg.ranges),
                math.degrees(msg.angle_max - msg.angle_min),
                inc,
                msg.range_min,
                msg.range_max,
            )

        arr = np.asarray(msg.ranges, dtype=np.float32)
        if arr.size > 0:
            arr[~np.isfinite(arr)] = float(self.sensor_config["range_max"])
            arr = np.clip(
                arr,
                float(self.sensor_config["range_min"]),
                float(self.sensor_config["range_max"]),
            )
            self._ranges_clipped = arr
        else:
            self._ranges_clipped = None

    def odom_callback(self, msg):
        self.current_pose = PoseStamped()
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose
        self.last_odom_time = rospy.get_time()

    # --------------------- goal/action ---------------------
    def goal_callback(self, goal):
        self.current_goal = goal.target_pose
        self.no_feasible = False
        self.goal_start_time = rospy.get_time()
        self.recovery_goal_suppress_counter = 0
        self.progress_history = []
        self.stuck_counter = 0
        self._exit_recovery()

        rospy.loginfo(
            "New goal: (%.2f, %.2f) frame=%s",
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.header.frame_id,
        )

        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.current_pose is None:
                r.sleep()
                continue
            if self.is_goal_reached():
                self.action_server.set_succeeded()
                self.cmd_pub.publish(Twist())
                rospy.loginfo("Goal reached")
                break
            if self.action_server.is_preempt_requested():
                self.action_server.set_preempted()
                self.cmd_pub.publish(Twist())
                rospy.loginfo("Goal preempted")
                break
            r.sleep()

    def is_goal_reached(self):
        if not self.current_goal or not self.current_pose:
            return False
        dx = self.current_goal.pose.position.x - self.current_pose.pose.position.x
        dy = self.current_goal.pose.position.y - self.current_pose.pose.position.y
        return math.hypot(dx, dy) < self.goal_tolerance

    # --------------------- planning loop ---------------------
    def planning_cycle(self, _):
        now = rospy.get_time()

        if (
            not self.planning_enabled
            or self.current_goal is None
            or self.current_pose is None
            or self.latest_scan is None
        ):
            self.cmd_pub.publish(Twist())
            return

        scan_age = now - self.last_scan_time
        odom_age = now - self.last_odom_time
        if scan_age > self.scan_timeout or odom_age > self.odom_timeout:
            rospy.logwarn_throttle(
                1.0, "Data stale -> STOP (scan=%.3f, odom=%.3f)", scan_age, odom_age
            )
            self.cmd_pub.publish(Twist())
            return

        self._fwd_cache = self._compute_forward_clearance_cached()
        self.update_progress_history(now)

        if self.enable_recovery_scan and self.recovery_state != "NONE":
            self._recovery_step(now)
            return

        self.generate_motion_tubes()
        self.evaluate_tubes()

        feas = [t for t in self.motion_tubes if t.is_feasible]
        stuck_by_progress = self.is_stuck_by_progress()

        if len(feas) > 0 and not stuck_by_progress:
            self.no_feasible = False

            if self.recovery_goal_suppress_counter > 0:
                self.recovery_goal_suppress_counter -= 1
                if self.recovery_goal_suppress_counter == 0:
                    rospy.loginfo("Recovery goal suppression ended, full goal weight restored")

            self.select_best_tube()
            self.publish_commands()
            self._publish_tube_markers()
            return

        self.no_feasible = len(feas) == 0

        if not self.enable_recovery_scan:
            self.selected_tube = None
            self.cmd_pub.publish(Twist())
            self._publish_tube_markers()
            return

        if self.recovery_state == "NONE":
            rospy.logwarn(
                "Entering recovery: no_feasible=%s stuck=%s",
                str(self.no_feasible),
                str(stuck_by_progress),
            )
            self._enter_recovery()

        self._recovery_step(now)

    # =====================================================================
    # Helpers
    # =====================================================================
    def _pose_xy_yaw(self):
        if not self.current_pose:
            return 0.0, 0.0, 0.0
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return x, y, yaw

    def _wrap(self, ang):
        while ang > math.pi:
            ang -= 2 * math.pi
        while ang < -math.pi:
            ang += 2 * math.pi
        return ang

    def _current_xy(self):
        if not self.current_pose:
            return (0.0, 0.0)
        p = self.current_pose.pose.position
        return (p.x, p.y)

    def _dist_xy(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def angle_to_beam_idx(self, angle_in_base):
        if not self.latest_scan or self.sensor_config["num_beams"] == 0:
            return 0
        angle_in_laser = self._wrap(angle_in_base + self.base_to_laser_yaw)
        ang_min = self.sensor_config["min_angle"]
        ang_max = self.sensor_config["max_angle"]
        inc = self.sensor_config["angle_increment"]
        n = self.sensor_config["num_beams"]

        if inc > 0:
            if angle_in_laser <= ang_min:
                return 0
            if angle_in_laser >= ang_max:
                return n - 1
        else:
            if angle_in_laser >= ang_min:
                return 0
            if angle_in_laser <= ang_max:
                return n - 1

        idx = int((angle_in_laser - ang_min) / inc)
        return max(0, min(n - 1, idx))

    def _compute_forward_clearance_cached(self):
        if self._ranges_clipped is None or self.sensor_config["num_beams"] == 0:
            return float("inf")

        ang_min = self.sensor_config["min_angle"]
        inc = self.sensor_config["angle_increment"]
        n = self.sensor_config["num_beams"]
        forward_laser = self._wrap(-self.base_to_laser_yaw)
        half = math.radians(self.fwd_slow_half_angle_deg)

        vals = []
        for i in range(n):
            a = ang_min + i * inc
            if abs(self._wrap(a - forward_laser)) <= half:
                vals.append(float(self._ranges_clipped[i]))

        if len(vals) == 0:
            return float("inf")

        return float(np.percentile(vals, 10))

    # =====================================================================
    # Stuck detection
    # =====================================================================
    def update_progress_history(self, now):
        if self.current_pose is None or self.current_goal is None:
            return

        x, y, _ = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        goal_dist = math.hypot(gx - x, gy - y)

        self.progress_history.append((now, x, y, goal_dist))
        t_min = now - self.stuck_window_sec
        self.progress_history = [p for p in self.progress_history if p[0] >= t_min]

    def is_stuck_by_progress(self):
        if len(self.progress_history) < 2:
            return False

        t0, x0, y0, d0 = self.progress_history[0]
        t1, x1, y1, d1 = self.progress_history[-1]

        motion = math.hypot(x1 - x0, y1 - y0)
        progress = d0 - d1
        stuck = (motion < self.stuck_min_motion and progress < self.stuck_min_progress)

        if stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        return self.stuck_counter >= self.stuck_confirm_count

    # =====================================================================
    # Recovery helpers
    # =====================================================================
    def _enter_recovery(self):
        self.recovery_state = "BACKUP"
        self.recovery_enter_t = rospy.get_time()
        self.recovery_backup_start_t = self.recovery_enter_t
        self.recovery_backup_start_xy = self._current_xy()

        self.recovery_scan_direction = 0
        self.recovery_current_offset_deg = 0.0
        self.recovery_last_switch_t = rospy.get_time()

        self.escape_start_xy = self._current_xy()
        self.recovery_goal_xy = None
        self.recovery_follow_mode = "NONE"

        rospy.logwarn(
            "RECOVERY enter: BACKUP %.2fm then SCAN then ESCAPE_FOLLOW",
            self.recovery_backup_dist,
        )

    def _exit_recovery(self):
        if self.recovery_state != "NONE":
            rospy.logwarn(
                "RECOVERY exit – suppressing goal for %d cycles",
                self.recovery_goal_suppress_cycles,
            )
            self.recovery_goal_suppress_counter = self.recovery_goal_suppress_cycles

        self.recovery_state = "NONE"
        self.recovery_backup_start_xy = None
        self.recovery_backup_start_t = 0.0
        self.recovery_scan_direction = 0
        self.recovery_current_offset_deg = 0.0

        self.recovery_goal_xy = None
        self.recovery_follow_mode = "NONE"
        self.escape_follow_enter_t = 0.0
        self.escape_start_xy = None

    def _recovery_step(self, now):
        if self.recovery_state == "BACKUP":
            cmd, done = self._recovery_backup_cmd_or_done(now)
            self.cmd_pub.publish(cmd)
            self._publish_tube_markers()

            if not done:
                return

            self.generate_motion_tubes()
            self.evaluate_tubes_recovery()

            self._decide_scan_direction()
            self.recovery_state = "SCAN"

            _, _, yaw = self._pose_xy_yaw()
            self.recovery_base_yaw = yaw
            self.recovery_current_offset_deg = self.recovery_scan_step_deg
            self.recovery_target_yaw = self._wrap(
                yaw + math.radians(self.recovery_scan_direction * self.recovery_current_offset_deg)
            )
            self.recovery_last_switch_t = now

            rospy.logwarn("RECOVERY BACKUP done -> rotate first, then build escape follow")
            self.cmd_pub.publish(Twist())
            self._publish_tube_markers()
            return

        if self.recovery_state == "SCAN":
            cmd, done = self._recovery_rotate_cmd_or_done()
            self.cmd_pub.publish(cmd)
            self._publish_tube_markers()

            if not done:
                return

            if (now - self.recovery_last_switch_t) < self.recovery_min_pause:
                self.cmd_pub.publish(Twist())
                return

            self.generate_motion_tubes()
            self.evaluate_tubes_recovery()

            rec_goal = self.build_recovery_goal()
            if rec_goal is not None:
                self.recovery_follow_mode = "LEFT" if self.recovery_scan_direction > 0 else "RIGHT"
                self.recovery_state = "ESCAPE_FOLLOW"
                self.escape_follow_enter_t = now
                rospy.logwarn(
                    "RECOVERY: enter ESCAPE_FOLLOW, rec_goal=(%.2f, %.2f), mode=%s",
                    rec_goal[0],
                    rec_goal[1],
                    self.recovery_follow_mode,
                )
                return

            self.recovery_current_offset_deg += self.recovery_scan_step_deg
            if self.recovery_current_offset_deg > self.recovery_scan_max_deg:
                rospy.logwarn(
                    "RECOVERY SCAN: reached %.0f° limit, restarting",
                    self.recovery_scan_max_deg,
                )
                self.recovery_current_offset_deg = self.recovery_scan_step_deg

            self.recovery_target_yaw = self._wrap(
                self.recovery_base_yaw
                + math.radians(self.recovery_scan_direction * self.recovery_current_offset_deg)
            )
            self.recovery_last_switch_t = now
            rospy.logwarn(
                "RECOVERY SCAN: advancing to %.0f° (dir=%+d)",
                self.recovery_current_offset_deg,
                self.recovery_scan_direction,
            )
            self.cmd_pub.publish(Twist())
            self._publish_tube_markers()
            return

        if self.recovery_state == "ESCAPE_FOLLOW":
            self.generate_motion_tubes()
            self.evaluate_tubes_escape()
            feas = [t for t in self.motion_tubes if t.is_feasible]

            if len(feas) == 0:
                self._record_recovery_attempt(success=False)
                rospy.logwarn("ESCAPE_FOLLOW: no feasible tubes -> back to SCAN")
                self.recovery_state = "SCAN"
                _, _, yaw = self._pose_xy_yaw()
                self.recovery_base_yaw = yaw
                self.recovery_current_offset_deg = self.recovery_scan_step_deg
                self.recovery_target_yaw = self._wrap(
                    yaw + math.radians(self.recovery_scan_direction * self.recovery_current_offset_deg)
                )
                self.recovery_last_switch_t = now
                self.cmd_pub.publish(Twist())
                return

            self.select_best_tube()
            self.publish_commands()
            self._publish_tube_markers()

            # exit conditions
            if self.is_recovery_goal_reached() or self.has_left_stuck_region():
                self._record_recovery_attempt(success=True)
                self.no_feasible = False
                self._exit_recovery()
                return

            if (now - self.escape_follow_enter_t) > self.escape_follow_timeout:
                self._record_recovery_attempt(success=False)
                rospy.logwarn("ESCAPE_FOLLOW timeout -> SCAN again")
                self.recovery_state = "SCAN"
                _, _, yaw = self._pose_xy_yaw()
                self.recovery_base_yaw = yaw
                self.recovery_current_offset_deg = self.recovery_scan_step_deg
                self.recovery_target_yaw = self._wrap(
                    yaw + math.radians(self.recovery_scan_direction * self.recovery_current_offset_deg)
                )
                self.recovery_last_switch_t = now
                self.cmd_pub.publish(Twist())
                return

            return

        self._exit_recovery()
        self.cmd_pub.publish(Twist())

    def _decide_scan_direction(self):
        left_clear = self._side_clearance(+1)
        right_clear = self._side_clearance(-1)

        min_safe = self.robot.half_width + 0.08
        left_blocked = left_clear < min_safe
        right_blocked = right_clear < min_safe

        if left_blocked and not right_blocked:
            self.recovery_scan_direction = -1
            rospy.logwarn("RECOVERY: left blocked (%.2fm), scanning RIGHT", left_clear)
            return
        if right_blocked and not left_blocked:
            self.recovery_scan_direction = +1
            rospy.logwarn("RECOVERY: right blocked (%.2fm), scanning LEFT", right_clear)
            return

        left_score, right_score = self._tube_side_escape_score()

        if left_score > right_score + 1e-3:
            self.recovery_scan_direction = +1
            rospy.logwarn(
                "RECOVERY: left escape score better (%.3f vs %.3f), scanning LEFT",
                left_score,
                right_score,
            )
            return
        if right_score > left_score + 1e-3:
            self.recovery_scan_direction = -1
            rospy.logwarn(
                "RECOVERY: right escape score better (%.3f vs %.3f), scanning RIGHT",
                right_score,
                left_score,
            )
            return

        cx, cy, yaw = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        goal_bearing = math.atan2(gy - cy, gx - cx)
        relative = self._wrap(goal_bearing - yaw)

        self.recovery_scan_direction = +1 if relative >= 0 else -1
        rospy.logwarn(
            "RECOVERY: tie-break, goal to %s, scanning %s",
            "LEFT" if relative >= 0 else "RIGHT",
            "LEFT" if relative >= 0 else "RIGHT",
        )

    def _tube_side_escape_score(self):
        left_score = 0.0
        right_score = 0.0

        for t in self.motion_tubes:
            side_weight = 0.0
            if t.w > 0.08:
                side_weight = 1.0
            elif t.w < -0.08:
                side_weight = -1.0
            else:
                continue

            clr = t.min_clearance if np.isfinite(t.min_clearance) else -0.1
            score = max(-0.3, min(0.5, clr)) + 0.15 * min(1.0, t.arc_len)

            if side_weight > 0:
                left_score += score
            else:
                right_score += score

        return left_score, right_score

    def _side_clearance(self, side):
        if self._ranges_clipped is None or self.sensor_config["num_beams"] == 0:
            return float("inf")

        ang_min = self.sensor_config["min_angle"]
        inc = self.sensor_config["angle_increment"]
        n = self.sensor_config["num_beams"]
        forward_laser = self._wrap(-self.base_to_laser_yaw)

        vals = []
        half_fov = math.radians(95.0)

        for i in range(n):
            a = ang_min + i * inc
            diff = self._wrap(a - forward_laser)
            if side > 0 and 0 < diff < half_fov:
                vals.append(float(self._ranges_clipped[i]))
            elif side < 0 and -half_fov < diff < 0:
                vals.append(float(self._ranges_clipped[i]))

        if len(vals) == 0:
            return float("inf")

        return float(np.percentile(vals, 12))

    def _recovery_backup_cmd_or_done(self, now):
        if self.recovery_backup_start_xy is None:
            self.recovery_backup_start_xy = self._current_xy()
        if self.recovery_backup_start_t <= 1e-6:
            self.recovery_backup_start_t = now

        moved = self._dist_xy(self._current_xy(), self.recovery_backup_start_xy)
        elapsed = now - self.recovery_backup_start_t
        done_dist = moved >= self.recovery_backup_dist
        done_time = elapsed >= self.recovery_backup_min_time

        rospy.logwarn_throttle(
            0.5,
            "RECOVERY BACKUP: moved=%.3fm/%.3fm elapsed=%.2fs/%.2fs",
            moved,
            self.recovery_backup_dist,
            elapsed,
            self.recovery_backup_min_time,
        )

        if (now - self.recovery_enter_t) > self.recovery_backup_timeout:
            rospy.logwarn("RECOVERY BACKUP timeout -> SCAN")
            return Twist(), True

        if done_dist and done_time:
            return Twist(), True

        cmd = Twist()
        cmd.linear.x = self.recovery_backup_v
        cmd.angular.z = 0.0
        return cmd, False

    def _recovery_rotate_cmd_or_done(self):
        _, _, yaw = self._pose_xy_yaw()
        err = self._wrap(self.recovery_target_yaw - yaw)
        tol = math.radians(self.recovery_yaw_tol_deg)

        if abs(err) <= tol:
            return Twist(), True

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.recovery_w if err > 0 else -self.recovery_w
        return cmd, False

    def build_recovery_goal(self):
        x, y, yaw = self._pose_xy_yaw()
        theta_escape, open_range = self.select_escape_direction_from_scan()

        if theta_escape is None:
            return None

        d = min(self.recovery_goal_max_dist, max(self.recovery_goal_min_dist, 0.7 * open_range))
        gx = x + d * math.cos(yaw + theta_escape)
        gy = y + d * math.sin(yaw + theta_escape)

        self.recovery_goal_xy = (gx, gy)
        return self.recovery_goal_xy

    def select_escape_direction_from_scan(self):
        if self._ranges_clipped is None or self.sensor_config["num_beams"] == 0:
            return None, 0.0

        ang_min = self.sensor_config["min_angle"]
        inc = self.sensor_config["angle_increment"]
        n = self.sensor_config["num_beams"]

        best_score = -1e9
        best_theta = None
        best_open = 0.0

        step = max(1, n // 60)
        for i in range(0, n, step):
            a = ang_min + i * inc
            r = float(self._ranges_clipped[i])

            front_bias = 1.0 - 0.25 * abs(a) / math.pi
            score = r * front_bias
            score -= self.recovery_direction_memory_penalty(a)

            if score > best_score:
                best_score = score
                best_theta = a
                best_open = r

        return best_theta, best_open

    def recovery_direction_memory_penalty(self, theta):
        now = rospy.get_time()
        penalty = 0.0
        dir_sign = +1 if theta >= 0.0 else -1
        x, y = self._current_xy()

        kept = []
        for item in self.recovery_history:
            if (now - item["t"]) <= self.recovery_memory_decay_sec:
                kept.append(item)
                if self._dist_xy((x, y), item["xy"]) < self.recovery_memory_radius:
                    if item["dir"] == dir_sign and (not item["success"]):
                        penalty += self.w_recovery_memory

        self.recovery_history = kept
        return penalty

    def is_recovery_goal_reached(self):
        if self.recovery_goal_xy is None:
            return False
        x, y = self._current_xy()
        return self._dist_xy((x, y), self.recovery_goal_xy) < self.recovery_goal_radius

    def has_left_stuck_region(self):
        if self.escape_start_xy is None:
            return False
        return self._dist_xy(self._current_xy(), self.escape_start_xy) >= self.escape_min_leave_dist

    def _record_recovery_attempt(self, success):
        self.recovery_history.append(
            {
                "xy": self._current_xy(),
                "dir": self.recovery_scan_direction,
                "success": bool(success),
                "t": rospy.get_time(),
            }
        )

    # =====================================================================
    # Tube generation
    # =====================================================================
    def generate_motion_tubes(self):
        self.motion_tubes = []
        fwd = self._fwd_cache

        for (v, w, T) in self._tube_configs:
            approx_len = abs(v) * T

            if fwd < 0.65 and abs(w) < 0.18 and approx_len > (fwd + 0.20):
                continue

            if fwd < 1.0 and abs(w) < 0.10 and v > 0.9 and T > 1.5:
                continue

            tube = self.create_motion_tube(v, w, T)
            if tube:
                self.motion_tubes.append(tube)

        rospy.loginfo_throttle(
            2.0,
            "Generated %d tubes (from %d configs)",
            len(self.motion_tubes),
            len(self._tube_configs),
        )

    def create_motion_tube(self, v, w, T):
        if abs(w) < 0.05:
            ttype = TemplateType.MOVE_STRAIGHT
        elif w > 0:
            ttype = TemplateType.STEER_LEFT
        else:
            ttype = TemplateType.STEER_RIGHT

        samples, beam_indices = [], []
        arc_len = abs(v) * T
        n_samples = int(np.clip(8 + arc_len * 12.0, 12, 48))

        for i in range(1, n_samples + 1):
            t = T * i / float(n_samples)
            if abs(w) < 1e-3:
                x, y = v * t, 0.0
                heading = 0.0
            else:
                R = v / w
                theta = w * t
                x = R * math.sin(theta)
                y = R * (1.0 - math.cos(theta))
                heading = theta

            samples.append(np.array([x, y], dtype=np.float32))
            ang_base = math.atan2(y, x)
            idx = self.angle_to_beam_idx(ang_base + 0.04 * heading)
            beam_indices.append(idx)

        return MotionTube(ttype, v, w, T, samples, beam_indices, arc_len)

    # =====================================================================
    # Tube evaluation
    # =====================================================================
    def evaluate_tubes(self):
        if not self.motion_tubes:
            return

        yaw = self._pose_xy_yaw()[2]
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        goal_bearing = math.atan2(gy - cy, gx - cx)

        if self.recovery_goal_suppress_counter > 0:
            total = max(1.0, float(self.recovery_goal_suppress_cycles))
            restored = total - float(self.recovery_goal_suppress_counter)
            goal_factor = max(0.0, min(1.0, restored / total))
            rospy.loginfo_throttle(
                1.0,
                "Goal partially suppressed (factor=%.2f, %d cycles remaining)",
                goal_factor,
                self.recovery_goal_suppress_counter,
            )
        else:
            goal_factor = 1.0

        for t in self.motion_tubes:
            t.is_feasible, t.min_clearance, t.obstacle_penalty = self.collision_check_halfwidth_hard(t)
            if not t.is_feasible:
                continue
            t.heading_after = yaw + t.w * t.T
            t.goal_progress = self.goal_progress_along_tube(t)
            t.cost = self.composite_cost(t, goal_bearing, goal_factor)

        feas = [x for x in self.motion_tubes if x.is_feasible]
        if feas:
            self.min_cost = min(x.cost for x in feas)
            self.max_cost = max(x.cost for x in feas)
        else:
            self.min_cost = self.max_cost = 0.0

    def evaluate_tubes_recovery(self):
        if not self.motion_tubes:
            return

        yaw = self._pose_xy_yaw()[2]
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        goal_bearing = math.atan2(gy - cy, gx - cx)

        for t in self.motion_tubes:
            t.is_feasible, t.min_clearance, t.obstacle_penalty = self.collision_check_halfwidth_hard(t)
            t.heading_after = yaw + t.w * t.T
            t.goal_progress = self.goal_progress_along_tube(t)
            if t.is_feasible:
                t.cost = self.composite_cost(t, goal_bearing, goal_factor=0.0)

        feas = [x for x in self.motion_tubes if x.is_feasible]
        if feas:
            self.min_cost = min(x.cost for x in feas)
            self.max_cost = max(x.cost for x in feas)
        else:
            self.min_cost = self.max_cost = 0.0

    def evaluate_tubes_escape(self):
        if not self.motion_tubes or self.recovery_goal_xy is None:
            return

        yaw = self._pose_xy_yaw()[2]
        rx, ry = self.recovery_goal_xy
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        rec_bearing = math.atan2(ry - cy, rx - cx)

        for t in self.motion_tubes:
            t.is_feasible, t.min_clearance, t.obstacle_penalty = self.collision_check_halfwidth_hard(t)
            if not t.is_feasible:
                continue
            t.heading_after = yaw + t.w * t.T
            t.goal_progress = self.escape_progress_along_tube(t)
            t.cost = self.escape_composite_cost(t, rec_bearing)

        feas = [x for x in self.motion_tubes if x.is_feasible]
        if feas:
            self.min_cost = min(x.cost for x in feas)
            self.max_cost = max(x.cost for x in feas)
        else:
            self.min_cost = self.max_cost = 0.0

    def collision_check_halfwidth_hard(self, tube):
        if self._ranges_clipped is None:
            tube.left_clearance = -1.0
            tube.right_clearance = -1.0
            tube.center_balance = 1.0
            return False, -1.0, 0.0

        inc = float(self.sensor_config["angle_increment"])
        n = int(self.sensor_config["num_beams"])
        eff_r = float(self.robot.half_width)

        min_clr = float("inf")
        left_min_clr = float("inf")
        right_min_clr = float("inf")

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(10, len(tube.samples)), dtype=int)

        for k in idxs:
            s = tube.samples[k]
            dist = float(np.linalg.norm(s))
            if dist < 1e-3:
                continue

            half = math.atan2(eff_r, max(0.05, dist))
            center_idx = tube.beam_indices[k]
            beam_span = max(1, int(half / max(abs(inc), 1e-6)))

            i0 = max(0, center_idx - beam_span)
            i1 = min(n - 1, center_idx + beam_span)

            local_min = float("inf")
            local_left_min = float("inf")
            local_right_min = float("inf")

            for j in range(i0, i1 + 1):
                r = float(self._ranges_clipped[j])
                clr = r - dist - eff_r

                if clr < local_min:
                    local_min = clr
                if clr < min_clr:
                    min_clr = clr

                if j < center_idx:
                    if clr < local_right_min:
                        local_right_min = clr
                    if clr < right_min_clr:
                        right_min_clr = clr
                elif j > center_idx:
                    if clr < local_left_min:
                        local_left_min = clr
                    if clr < left_min_clr:
                        left_min_clr = clr
                else:
                    if clr < local_left_min:
                        local_left_min = clr
                    if clr < local_right_min:
                        local_right_min = clr
                    if clr < left_min_clr:
                        left_min_clr = clr
                    if clr < right_min_clr:
                        right_min_clr = clr

            if local_min < 0.0:
                tube.left_clearance = left_min_clr if np.isfinite(left_min_clr) else -1.0
                tube.right_clearance = right_min_clr if np.isfinite(right_min_clr) else -1.0
                tube.center_balance = abs(
                    max(-0.2, tube.left_clearance) - max(-0.2, tube.right_clearance)
                )
                return False, min_clr, 0.0

        if not np.isfinite(min_clr):
            min_clr = -1.0
        if not np.isfinite(left_min_clr):
            left_min_clr = min_clr
        if not np.isfinite(right_min_clr):
            right_min_clr = min_clr

        tube.left_clearance = left_min_clr
        tube.right_clearance = right_min_clr
        tube.center_balance = abs(left_min_clr - right_min_clr)

        return True, min_clr, 0.0

    def goal_progress_along_tube(self, tube):
        cx, cy, yaw = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        now_dist = math.hypot(gx - cx, gy - cy)

        if len(tube.samples) == 0:
            return 0.0

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(5, len(tube.samples)), dtype=int)
        improvements = []

        for k in idxs:
            s = tube.samples[k]
            sx, sy = float(s[0]), float(s[1])

            fx = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
            fy = cy + sx * math.sin(yaw) + sy * math.cos(yaw)

            after_dist = math.hypot(gx - fx, gy - fy)
            improvements.append(now_dist - after_dist)

        if len(improvements) == 0:
            return 0.0

        avg_prog = float(np.mean(improvements))
        best_prog = float(np.max(improvements))
        prog = 0.6 * avg_prog + 0.4 * best_prog
        return max(0.0, prog)

    def escape_progress_along_tube(self, tube):
        if self.recovery_goal_xy is None:
            return 0.0

        cx, cy, yaw = self._pose_xy_yaw()
        gx, gy = self.recovery_goal_xy
        now_dist = math.hypot(gx - cx, gy - cy)

        if len(tube.samples) == 0:
            return 0.0

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(5, len(tube.samples)), dtype=int)
        improvements = []

        for k in idxs:
            s = tube.samples[k]
            sx, sy = float(s[0]), float(s[1])

            fx = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
            fy = cy + sx * math.sin(yaw) + sy * math.cos(yaw)

            after_dist = math.hypot(gx - fx, gy - fy)
            improvements.append(now_dist - after_dist)

        if len(improvements) == 0:
            return 0.0

        avg_prog = float(np.mean(improvements))
        best_prog = float(np.max(improvements))
        prog = 0.6 * avg_prog + 0.4 * best_prog
        return max(0.0, prog)

    def composite_cost(self, t, goal_bearing, goal_factor=1.0):
        w_progress = 14.0
        w_heading = 5.0
        w_curvature = 0.8
        w_length = 2.0    #0.15
        w_speed = 0.8
        w_clearance = self.w_clearance

        c = 0.0
        c -= goal_factor * w_progress * t.goal_progress

        heading_err = abs(self._wrap(t.heading_after - goal_bearing))
        c += goal_factor * w_heading * heading_err

        c += w_curvature * abs(t.w)

        # reward longer tubes instead of penalizing them
        c -= w_length * t.arc_len

        c -= w_speed * t.v

        if t.min_clearance < self.clearance_safe_dist:
            clearance_ratio = max(
                0.0, 1.0 - t.min_clearance / max(1e-6, self.clearance_safe_dist)
            )
            c += w_clearance * clearance_ratio

        if t.min_clearance < 0.08:
            c += 3.0

        side_min = min(t.left_clearance, t.right_clearance)
        if side_min < self.side_clearance_safe_dist:
            side_ratio = max(
                0.0, 1.0 - side_min / max(1e-6, self.side_clearance_safe_dist)
            )
            c += self.w_side_clearance * side_ratio

        c += self.w_center_balance * min(0.25, t.center_balance)

        return c

    def escape_composite_cost(self, t, escape_bearing):
        c = 0.0

        w_progress = 12.0
        w_heading = self.w_escape_heading
        w_curvature = 0.6
        w_length = 2.5
        w_speed = 0.5
        w_clearance = self.w_clearance
        w_open = self.w_escape_open
        w_leave = self.w_escape_leave
        w_wall = self.w_wall_follow

        c -= w_progress * t.goal_progress

        heading_err = abs(self._wrap(t.heading_after - escape_bearing))
        c += w_heading * heading_err

        c -= w_open * max(0.0, min(0.5, t.min_clearance))
        c -= w_leave * min(1.0, t.arc_len)

        c += w_curvature * abs(t.w)

        # reward longer tubes instead of penalizing them
        c -= w_length * t.arc_len

        c -= w_speed * t.v

        if t.min_clearance < self.clearance_safe_dist:
            clearance_ratio = max(
                0.0, 1.0 - t.min_clearance / max(1e-6, self.clearance_safe_dist)
            )
            c += w_clearance * clearance_ratio

        side_min = min(t.left_clearance, t.right_clearance)
        if side_min < self.side_clearance_safe_dist:
            side_ratio = max(
                0.0, 1.0 - side_min / max(1e-6, self.side_clearance_safe_dist)
            )
            c += self.w_side_clearance * side_ratio

        c += self.w_center_balance * min(0.25, t.center_balance)
        c += w_wall * self.wall_follow_cost(t)
        return c
    
    def wall_follow_cost(self, tube):
        if self.recovery_follow_mode == "NONE":
            return 0.0

        if self.recovery_follow_mode == "LEFT":
            d = self._side_clearance(+1)
        else:
            d = self._side_clearance(-1)

        if not np.isfinite(d):
            return 0.0

        return abs(d - self.wall_follow_ref_dist)

    def _filter_longest_available_tubes(self, feasible_tubes, rel_ratio=None):
        if rel_ratio is None:
            rel_ratio = self.long_tube_rel_ratio
        if not feasible_tubes:
            return []

        max_len = max(t.arc_len for t in feasible_tubes)
        return [t for t in feasible_tubes if t.arc_len >= rel_ratio * max_len]
        
    # --------------------- selection & command ---------------------
    def select_best_tube(self):
        feas = [t for t in self.motion_tubes if t.is_feasible]
        self.no_feasible = len(feas) == 0

        if not feas:
            self.selected_tube = None
            rospy.logwarn_throttle(
                1.0, "No feasible tubes (generated %d)", len(self.motion_tubes)
            )
            return

        # Keep only the longest available group.
        # Short tubes are only considered when longer ones are blocked.
        long_feas = self._filter_longest_available_tubes(
            feas, rel_ratio=self.long_tube_rel_ratio
        )

        self.selected_tube = min(long_feas, key=lambda z: z.cost)

        rospy.loginfo_throttle(
            1.0,
            "Selected long-priority tube: arc_len=%.2f cost=%.3f (%d long-group / %d feasible)",
            self.selected_tube.arc_len,
            self.selected_tube.cost,
            len(long_feas),
            len(feas),
        )

    def publish_commands(self):
        cmd = Twist()

        if self.selected_tube:
            v = max(0.0, min(self.max_v, self.selected_tube.v))
            w = max(-self.max_w, min(self.max_w, self.selected_tube.w))

            if abs(w) < self.deadband_w:
                w = 0.0

            if self.use_fwd_slowdown and self.fwd_slow_gain > 1e-6:
                fwd = self._fwd_cache
                scale = max(self.min_forward_scale, min(1.0, fwd / (1.0 + 1e-6)))
                v *= scale ** (1.0 + self.fwd_slow_gain)

            if self.selected_tube.min_clearance < 0.12:
                v *= 0.75
            if abs(w) > 0.8:
                v *= 0.85

            if self.recovery_state == "ESCAPE_FOLLOW":
                v *= 0.90

            cmd.linear.x = v
            cmd.angular.z = w
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)
        rospy.loginfo_throttle(
            1.0,
            "CMD v=%.2f, w=%.2f (suppress=%d, recovery=%s)",
            cmd.linear.x,
            cmd.angular.z,
            self.recovery_goal_suppress_counter,
            self.recovery_state,
        )

    # --------------------- markers ---------------------
    def _init_marker_pose(self, m):
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

    def _tube_color(self, tube):
        if tube is self.selected_tube:
            return ColorRGBA(0.00, 0.95, 1.00, 1.0)
        if not tube.is_feasible:
            return ColorRGBA(0.95, 0.15, 0.15, 0.35)
        try:
            c = max(
                0.0,
                min(
                    1.0,
                    (tube.cost - self.min_cost) / (self.max_cost - self.min_cost + 1e-6),
                ),
            )
        except Exception:
            c = 0.5
        return ColorRGBA(1.0 * c, 1.0, 0.0, 0.9)

    def _publish_tube_markers(self):
        if not self.motion_tubes or not self.current_pose:
            return

        rx, ry, ryaw = self._pose_xy_yaw()
        cy, sy = math.cos(ryaw), math.sin(ryaw)
        ma = MarkerArray()
        frame = self.current_pose.header.frame_id or "odom"
        header = Header(stamp=rospy.Time.now(), frame_id=frame)

        wipe = Marker()
        wipe.header = header
        self._init_marker_pose(wipe)
        wipe.ns = "motion_tubes"
        wipe.id = 0
        wipe.action = Marker.DELETEALL
        ma.markers.append(wipe)

        mid = 1
        Z = 0.03
        BASE_W = 0.05
        SEL_W = 0.09

        for t in self.motion_tubes:
            m = Marker()
            m.header = header
            self._init_marker_pose(m)
            m.ns = "motion_tubes"
            m.id = mid
            mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = BASE_W
            m.color = self._tube_color(t)

            pts = []
            for s in t.samples:
                sx, sy_local = float(s[0]), float(s[1])
                ox = rx + (sx * cy - sy_local * sy)
                oy = ry + (sx * sy + sy_local * cy)
                pts.append(Point(ox, oy, Z))
            m.points = pts
            ma.markers.append(m)

        if self.selected_tube is not None:
            t = self.selected_tube
            m = Marker()
            m.header = header
            self._init_marker_pose(m)
            m.ns = "motion_tubes"
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = SEL_W
            m.color = ColorRGBA(0.00, 0.95, 1.00, 1.0)
            pts = []
            for s in t.samples:
                sx, sy_local = float(s[0]), float(s[1])
                ox = rx + (sx * cy - sy_local * sy)
                oy = ry + (sx * sy + sy_local * cy)
                pts.append(Point(ox, oy, Z))
            m.points = pts
            ma.markers.append(m)

        self.marker_pub.publish(ma)

    # --------------------- diagnostics ---------------------
    def print_diagnostics(self, _):
        feas = [t for t in self.motion_tubes if t.is_feasible]
        rospy.loginfo(
            "=== DIAG v2.5 === tubes=%d feas=%d fwd=%.2fm scan_age=%.3f "
            "no_feasible=%s recovery=%s goal_suppress=%d stuck_counter=%d rec_goal=%s",
            len(self.motion_tubes),
            len(feas),
            self._fwd_cache,
            rospy.get_time() - self.last_scan_time,
            str(self.no_feasible),
            self.recovery_state,
            self.recovery_goal_suppress_counter,
            self.stuck_counter,
            str(self.recovery_goal_xy),
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        planner = GoalOrientedMotionTubePlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
