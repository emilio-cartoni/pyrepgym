#!/bin/bash
set -e


cd

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
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


