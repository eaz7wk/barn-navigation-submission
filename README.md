<p align="center">
  <img width="100%" src="res/BARN_Challenge.png" />
</p>

--------------------------------------------------------------------------------

# BARN Navigation Challenge Submission

This repository contains our submission for the ICRA BARN Navigation Challenge.  
It is built upon the official benchmark repository with a custom motion tube–based navigation planner.

--------------------------------------------------------------------------------

## Updates
* Custom navigation stack integrated with motion tube planner
* Optimized trajectory selection for cluttered environments
* Tuned for stable performance in BARN benchmark worlds

--------------------------------------------------------------------------------

## Overview

We implement a **motion tube–based local planner** for autonomous navigation.

The planner:
- Samples multiple forward trajectories (motion tubes)
- Evaluates feasibility, obstacle clearance, and goal alignment
- Strongly prefers longer feasible trajectories
- Falls back to shorter trajectories only when necessary

This design improves navigation stability and success rate in narrow BARN environments.

--------------------------------------------------------------------------------

## Main Modified Components

The following files contain our main contributions:

- `run.py`  
  → Initializes and launches the custom navigation stack

- `jackal_helper/scripts/fixed_granular.py`  
  → Core motion tube planner implementation

- `Singularityfile.def`  
  → Defines all dependencies for reproducible evaluation

- `entrypoint.sh`  
  → Ensures correct ROS environment inside container

--------------------------------------------------------------------------------

## Requirements

### Local Machine (without container)
* ROS version ≥ Kinetic (we use Noetic)
* CMake ≥ 3.0.2
* Python ≥ 3.6
* Python packages:
  - defusedxml
  - rospkg
  - netifaces
  - numpy

### Singularity Container
* Singularity ≥ 3.6.3 and ≤ 4.0.2

--------------------------------------------------------------------------------

## Build Container

```bash
sudo singularity build barn_submit.sif Singularityfile.def
