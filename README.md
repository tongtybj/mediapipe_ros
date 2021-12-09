# mediapipe_ros

ROS package for [Mediapipe](https://google.github.io/mediapipe/) in Linux Env that the python3 is not the default version

## Environment
- Ubuntu 18.04 + Melodic

## Notice
We need `python3.7` to run this package

## Setup

### Install `python3.7` and related dependenceis

```
$ sudo apt-get install python3.7 python3.7-dev python3.7-venv
```

### Workspace build (melodic)
```
source /opt/ros/melodic/setup.bash
mkdir -p ~/mediapipe_ws/src
cd ~/mediapipe_ws/src
git clone https://github.com/tongtybj/mediapipe_ros.git
wstool init
wstool merge mediapipe_ros/fc.rosinstall.melodic
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/mediapipe_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so
catkin build
```
