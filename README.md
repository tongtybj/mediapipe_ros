# mediapipe_ros

ROS package for [Mediapipe](https://google.github.io/mediapipe/) in Linux Env that the python3 is not the default version

## Environment
- Ubuntu 18.04 + Melodic
- Ubuntu 20.04 + Noetic

## Setup

### Workspace build (melodic)
```bash
sudo apt-get install python3.7 python3.7-dev python3.7-venv
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

### Workspace build (Noetic)
```bash
source /opt/ros/noetic/setup.bash
mkdir -p ~/mediapipe_ws/src
cd ~/mediapipe_ws/src
git clone https://github.com/tongtybj/mediapipe_ros.git
wstool init
rosdep install --from-paths . --ignore-src -y -r
cd ~/mediapipe_ws
catkin init
catkin build
```

### Usage


#### In the first terminal:
```
$ roslaunch usb_cam usb_cam-test.launch
```

#### In the second terminal:
```
$ source ~/mediapipe_ws/devel/setup.bash
$ rosrun mediapipe_ros hand_gesture_ros.py
```

#### In the thid terminal:
```
$ rqt_image_view 
```
choose `/cv_hand_sense`
