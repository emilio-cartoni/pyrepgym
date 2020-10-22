#!/bin/bash
set -e

sudo apt install -y python-wstool
cd
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install -y ros-kinetic-desktop-full
source /opt/ros/kinetic/setup.bash
git clone git@git.ias.informatik.tu-darmstadt.de:ias_ros/ias_ros_core.git ias_ros
cd ias_ros
echo "kinetic" | ./install.sh
sudo rosdep init
rosdep update
echo "source ~/ias_ros/setup.bash" >> ~/.bashrc
source ~/.bashrc

