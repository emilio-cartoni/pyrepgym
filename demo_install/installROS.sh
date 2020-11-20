#!/bin/bash
set -e


cd

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install -y ros-kinetic-desktop-full
source /opt/ros/kinetic/setup.bash

sudo apt-get install -y python-wstool

git clone git@git.ias.informatik.tu-darmstadt.de:ias_ros/ias_ros_core.git ias_ros
cd ias_ros
echo "kinetic" | ./install.sh
sudo rosdep init
rosdep update
echo "source ~/ias_ros/setup.bash" >> ~/.bashrc
source ~/.bashrc

source ~/ias_ros/setup.bash

pip3 install mpi4py
pip3 install rospkg
ias install coppelia_sim_iiwas
ias update
ias make


### ADDITIONAL STEPS FOR TF2 messages
sudo apt-get install -y python3-empy

cd

mkdir -p ~/catkin_ws/src; cd ~/catkin_ws
catkin_make
source devel/setup.bash
wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool up
rosdep install --from-paths src --ignore-src -y -r

catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so

echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc







