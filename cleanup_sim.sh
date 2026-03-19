#!/usr/bin/env bash
set -u

echo "[cleanup] ===== start cleanup ====="

kill_pat() {
  local sig="$1"
  local pat="$2"
  pkill -"${sig}" -f "${pat}" >/dev/null 2>&1 || true
}

have_ros_master() {
  rostopic list >/dev/null 2>&1
}

echo "[cleanup] killing run.py / planner ..."
kill_pat INT  "python3.*run\.py"
kill_pat INT  "python.*run\.py"
kill_pat INT  "fixed_granular\.py"
kill_pat INT  "goal_oriented_motion_tube_planner"
sleep 1
kill_pat KILL "python3.*run\.py"
kill_pat KILL "python.*run\.py"
kill_pat KILL "fixed_granular\.py"
kill_pat KILL "goal_oriented_motion_tube_planner"

echo "[cleanup] killing roslaunch that runs gazebo_launch.launch ..."
kill_pat INT  "roslaunch.*gazebo_launch\.launch"
sleep 1
kill_pat KILL "roslaunch.*gazebo_launch\.launch"

echo "[cleanup] killing gzserver / gzclient ..."
kill_pat INT  "gzserver"
kill_pat INT  "gzclient"
sleep 1
kill_pat KILL "gzserver"
kill_pat KILL "gzclient"

echo "[cleanup] killing spawn_model / controller spawners ..."
kill_pat KILL "spawn_model"
kill_pat KILL "controller_spawner"
kill_pat KILL "spawner"
kill_pat KILL "gazebo_ros_control"

echo "[cleanup] rosnode cleanup ..."
if have_ros_master; then
  timeout 5s rosnode cleanup >/dev/null 2>&1 || true
else
  echo "[cleanup] ROS master not reachable, skip rosnode cleanup."
fi

echo "[cleanup] ===== cleanup done ====="
exit 0
