#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import subprocess
import os
import sys
from os.path import join

import numpy as np
import rospy
import rospkg

from gazebo_simulation import GazeboSimulation

INIT_POSITION = [-2, 3, 1.57]  # world frame (used only for gazebo reset)
GOAL_POSITION = [0, 10]        # relative goal (interpreted in odom after reset)


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def path_coord_to_gazebo_coord(x, y):
    radius = 0.075
    r_shift = -radius - (30 * radius * 2)
    c_shift = radius + 5
    gazebo_x = x * (radius * 2) + r_shift
    gazebo_y = y * (radius * 2) + c_shift
    return (gazebo_x, gazebo_y)


def terminate_process(proc, timeout=5.0):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_for_sim_time(timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout and not rospy.is_shutdown():
        if rospy.get_time() > 0:
            return True
        time.sleep(0.05)
    return False

def wait_for_ros_master(timeout=15.0):
    import rosgraph
    t0 = time.time()
    master = rosgraph.Master('/run.py')
    while time.time() - t0 < timeout and not rospy.is_shutdown():
        try:
            master.getPid()
            return True
        except Exception:
            time.sleep(0.1)
    return False
    
def wait_for_tf(tf_buffer, target_frame, source_frame, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout and not rospy.is_shutdown():
        try:
            tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(0.2)
            )
            return True
        except Exception:
            time.sleep(0.05)
    return False


def get_base_in_odom(tf_buffer, base_frame="base_link"):
    transform = tf_buffer.lookup_transform(
        "odom", base_frame, rospy.Time(0), rospy.Duration(0.5)
    )
    return (
        transform.transform.translation.x,
        transform.transform.translation.y,
    )


def wait_pose_stable_odom(
    tf_buffer,
    base_frame="base_link",
    stable_eps=0.01,
    stable_n=6,
    timeout=5.0,
):
    last = None
    stable_count = 0
    t0 = time.time()

    while time.time() - t0 < timeout and not rospy.is_shutdown():
        try:
            pose = get_base_in_odom(tf_buffer, base_frame=base_frame)
        except Exception:
            time.sleep(0.05)
            continue

        if last is not None and compute_distance(last, pose) < stable_eps:
            stable_count += 1
            if stable_count >= stable_n:
                return True
        else:
            stable_count = 0

        last = pose
        time.sleep(0.05)

    return False


def read_collision_fresh(gazebo_sim, checks=3, dt=0.1):
    collided = False
    for _ in range(checks):
        collided = collided or bool(gazebo_sim.get_hard_collision())
        time.sleep(dt)
    return collided


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test BARN navigation challenge")
    parser.add_argument("--world_idx", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out", type=str, default="out.txt")
    parser.add_argument("--planner", type=str, default="scripts/fixed_granular.py")
    parser.add_argument("--base_frame", type=str, default="base_link")
    args = parser.parse_args()

    gazebo_process = None
    nav_stack_process = None

    try:
        ##########################################################################################
        ## 0. Launch Gazebo Simulation
        ##########################################################################################

        os.environ["JACKAL_LASER"] = "1"
        os.environ["JACKAL_LASER_MODEL"] = "ust10"
        os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"

        if args.world_idx < 300:  # static environment from 0-299
            world_name = "BARN/world_%d.world" % (args.world_idx)
            INIT_POSITION = [-2.25, 3, 1.57]
            GOAL_POSITION = [0, 10]
        elif args.world_idx < 360:  # dynamic environment from 300-359
            world_name = "DynaBARN/world_%d.world" % (args.world_idx - 300)
            INIT_POSITION = [11, 0, 3.14]
            GOAL_POSITION = [-20, 0]
        else:
            raise ValueError("World index %d does not exist" % args.world_idx)

        print(">>>>>>>>>>>>>>>>>> Loading Gazebo Simulation with %s <<<<<<<<<<<<<<<<<<" % world_name)

        rospack = rospkg.RosPack()
        base_path = rospack.get_path("jackal_helper")
        os.environ["GAZEBO_PLUGIN_PATH"] = os.path.join(base_path, "plugins")

        launch_file = join(base_path, "launch", "gazebo_launch.launch")
        world_path = join(base_path, "worlds", world_name)

        gazebo_process = subprocess.Popen([
            "roslaunch",
            launch_file,
            "world_name:=" + world_path,
            "gui:=" + ("true" if args.gui else "false"),
        ])

        if not wait_for_ros_master(timeout=20.0):
            raise RuntimeError("ROS master did not start in time.")

        rospy.init_node("gym", anonymous=True)
        rospy.set_param("/use_sim_time", True)

        if not wait_for_sim_time(timeout=20.0):
            raise RuntimeError("Sim time did not start (/clock not publishing).")

        rospy.wait_for_service("/gazebo/get_model_state", timeout=30.0)

        gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)

        import tf2_ros
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        if not wait_for_tf(tf_buffer, "odom", args.base_frame, timeout=30.0):
            raise RuntimeError(
                "TF not ready: cannot lookup odom -> %s" % args.base_frame
            )

        init_coor = (INIT_POSITION[0], INIT_POSITION[1])
        goal_coor = (
            INIT_POSITION[0] + GOAL_POSITION[0],
            INIT_POSITION[1] + GOAL_POSITION[1],
        )

        print("[run] Resetting until pose stable and no collision...")

        max_resets = 20
        reset_ok = False
        for _ in range(max_resets):
            gazebo_sim.reset()
            time.sleep(0.6)

            wait_pose_stable_odom(
                tf_buffer, base_frame=args.base_frame, timeout=5.0
            )
            collided = read_collision_fresh(gazebo_sim, checks=3, dt=0.12)

            if not collided:
                reset_ok = True
                break

        if not reset_ok:
            raise RuntimeError("Failed to obtain a clean reset (collision keeps triggering).")

        init_odom = get_base_in_odom(tf_buffer, base_frame=args.base_frame)
        goal_odom = (
            init_odom[0] + GOAL_POSITION[0],
            init_odom[1] + GOAL_POSITION[1],
        )

        ##########################################################################################
        ## 1. Launch your navigation stack
        ##########################################################################################

        planner_script = args.planner
        if not os.path.isabs(planner_script):
            planner_script = join(base_path, planner_script)

        if not os.path.exists(planner_script):
            raise RuntimeError("Planner script not found: %s" % planner_script)

        nav_stack_process = subprocess.Popen([sys.executable, planner_script])

        import actionlib
        from geometry_msgs.msg import Quaternion
        from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

        nav_as = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        if not nav_as.wait_for_server(rospy.Duration(30.0)):
            raise RuntimeError("move_base action server not available on /move_base")

        mb_goal = MoveBaseGoal()
        mb_goal.target_pose.header.frame_id = "odom"
        mb_goal.target_pose.header.stamp = rospy.Time.now()
        mb_goal.target_pose.pose.position.x = goal_odom[0]
        mb_goal.target_pose.pose.position.y = goal_odom[1]
        mb_goal.target_pose.pose.position.z = 0.0
        mb_goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        nav_as.send_goal(mb_goal)

        ##########################################################################################
        ## 2. Start navigation
        ##########################################################################################

        curr_time = rospy.get_time()
        curr_odom = get_base_in_odom(tf_buffer, base_frame=args.base_frame)

        while compute_distance(init_odom, curr_odom) < 0.05 and (rospy.get_time() - curr_time) < 10.0:
            time.sleep(0.05)
            curr_time = rospy.get_time()
            curr_odom = get_base_in_odom(tf_buffer, base_frame=args.base_frame)

        start_time = rospy.get_time()
        collided = False
        timeout_s = 100.0

        while (
            compute_distance(goal_odom, curr_odom) > 1.0
            and not collided
            and (rospy.get_time() - start_time) < timeout_s
        ):
            curr_time = rospy.get_time()
            try:
                curr_odom = get_base_in_odom(tf_buffer, base_frame=args.base_frame)
            except Exception:
                time.sleep(0.05)
                continue

            print(
                "Time: %.2f (s), odom x: %.2f (m), odom y: %.2f (m)"
                % (curr_time - start_time, curr_odom[0], curr_odom[1]),
                end="\r",
            )

            collided = read_collision_fresh(gazebo_sim, checks=1, dt=0.0)
            time.sleep(0.1)

        ##########################################################################################
        ## 3. Report metrics and generate log
        ##########################################################################################

        print("\n>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")
        actual_time = rospy.get_time() - start_time

        success = False
        if collided:
            status = "collided"
        elif actual_time >= timeout_s:
            status = "timeout"
        else:
            status = "succeeded"
            success = True

        print("Navigation %s with time %.4f (s)" % (status, actual_time))

        if args.world_idx >= 300:
            path_length = abs(GOAL_POSITION[0]) + abs(GOAL_POSITION[1])
        else:
            path_file_name = join(
                base_path, "worlds/BARN/path_files", "path_%d.npy" % args.world_idx
            )
            path_array = np.load(path_file_name)
            path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
            path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
            path_array = np.insert(
                path_array,
                len(path_array),
                (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]),
                axis=0,
            )

            path_length = 0.0
            for p1, p2 in zip(path_array[:-1], path_array[1:]):
                path_length += compute_distance(p1, p2)

        optimal_time = path_length / 2.0
        nav_metric = int(success) * optimal_time / np.clip(
            actual_time, 2 * optimal_time, 8 * optimal_time
        )
        print("Navigation metric: %.4f" % nav_metric)

        with open(args.out, "a") as f:
            f.write(
                "%d %d %d %d %.4f %.4f\n"
                % (
                    args.world_idx,
                    int(success),
                    int(collided),
                    int(actual_time >= timeout_s),
                    actual_time,
                    nav_metric,
                )
            )

    finally:
        terminate_process(nav_stack_process, timeout=5.0)
        terminate_process(gazebo_process, timeout=5.0)
